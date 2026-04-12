"""
CBM training on TruthfulQA — adapted from CB-LLMs/generation/train_combined_finegrained.py.

Key differences from original:
  - Uses TruthfulQA question-answer pairs instead of classification datasets
  - Concept labels computed inline via MPNet similarity
  - Post-training generates TruthfulQA answers and saves in standard jsonl format
  - Calls evaluate.py for common evaluation with baselines
  - No external config file / no Hydra

Usage:
  python train_cbm.py --seed 42 --epochs 3 --batch_size 4
  python train_cbm.py --seed 42 --epochs 3 --evaluate
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, AutoModel

from config import TRUTHFULQA_CONCEPTS, TRUTHFULQA_SYSTEM_PROMPT, DEFAULT_SEED
from cbm_modules import CBL, elastic_net_penalty, cos_sim_cubed, mean_pooling

RESULTS_DIR = Path(__file__).parent / "results" / "truthfulqa"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_concept_labels(texts, concept_set, sim_model, sim_tokenizer, device, batch_size=32):
    """Compute MPNet cosine similarity between texts and concepts."""
    concept_enc = sim_tokenizer(concept_set, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        concept_out = sim_model(**concept_enc)
        concept_feats = mean_pooling(concept_out.last_hidden_state, concept_enc["attention_mask"])
        concept_feats = F.normalize(concept_feats, p=2, dim=1)

    all_sims = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Computing concept labels"):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        enc = sim_tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = sim_model(**enc)
            feats = mean_pooling(out.last_hidden_state, enc["attention_mask"])
            feats = F.normalize(feats, p=2, dim=1)
            sims = feats @ concept_feats.T
        all_sims.append(sims.cpu().numpy())
    return np.concatenate(all_sims, axis=0)


class TQADataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text, similarity):
        self.encoded_text = encoded_text
        self.similarity = similarity

    def __getitem__(self, idx):
        t = {k: torch.tensor(v[idx]) for k, v in self.encoded_text.items()}
        s = torch.FloatTensor(self.similarity[idx])
        return t, s

    def __len__(self):
        return len(self.encoded_text["input_ids"])


def main():
    parser = argparse.ArgumentParser(description="Train CBM on TruthfulQA")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--concept_loss_weight", type=float, default=1.0)
    parser.add_argument("--word_loss_weight", type=float, default=1.0)
    parser.add_argument("--elastic_net_alpha", type=float, default=1.0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"cbm-tqa-seed{args.seed}-{int(time.time())}"
    checkpoint_dir = CHECKPOINTS_DIR / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    concept_set = TRUTHFULQA_CONCEPTS
    print(f"Concepts: {len(concept_set)}")

    # --- Load TruthfulQA ---
    print("Loading TruthfulQA dataset...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    splits = ds.train_test_split(test_size=0.5, seed=42)
    train_df = splits["train"].to_pandas()
    test_df = splits["test"].to_pandas()

    train_texts = []
    for _, row in train_df.iterrows():
        for ans in row.correct_answers:
            train_texts.append(f"Q: {row.question.strip()}\nA: {ans}")
        for ans in row.incorrect_answers:
            train_texts.append(f"Q: {row.question.strip()}\nA: {ans}")

    test_texts = []
    for _, row in test_df.iterrows():
        for ans in row.correct_answers:
            test_texts.append(f"Q: {row.question.strip()}\nA: {ans}")

    print(f"Train texts: {len(train_texts)}, Test texts: {len(test_texts)}")

    # --- Tokenize ---
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_enc = tokenizer(train_texts, padding=True, truncation=True, max_length=args.max_length)
    test_enc = tokenizer(test_texts, padding=True, truncation=True, max_length=args.max_length)

    # --- Compute concept labels via MPNet ---
    print("Computing concept similarity labels...")
    sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    sim_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    sim_model.eval()

    train_sim = compute_concept_labels(train_texts, concept_set, sim_model, sim_tokenizer, device)
    del sim_model, sim_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Concept labels shape: {train_sim.shape}")

    test_sim = np.zeros((len(test_texts), len(concept_set)), dtype=np.float32)

    # --- Build dataloaders ---
    train_loader = torch.utils.data.DataLoader(
        TQADataset(train_enc, train_sim), batch_size=args.batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TQADataset(test_enc, test_sim), batch_size=args.batch_size, shuffle=False,
    )

    # --- Build model ---
    print("Building model...")
    config = LlamaConfig.from_pretrained(args.model_name)
    lora_config = LoraConfig(
        r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    )
    preLM = LlamaModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    preLM.print_trainable_parameters()

    cbl = CBL(config, len(concept_set), tokenizer).to(device)

    opt_lm = torch.optim.Adam(filter(lambda p: p.requires_grad, preLM.parameters()), lr=args.lr)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in preLM.parameters()) + sum(p.numel() for p in cbl.parameters())
    trainable = sum(p.numel() for p in preLM.parameters() if p.requires_grad) + sum(p.numel() for p in cbl.parameters())
    print(f"Total params: {total_params:,}, Trainable: {trainable:,} ({trainable/total_params:.4f})")

    # --- Training ---
    print("Training...")
    best_loss = float("inf")
    best_epoch = -1
    start_time = time.time()

    for epoch in range(args.epochs):
        preLM.train()
        cbl.train()
        losses = {"concept": [], "word": [], "reg": []}

        for batch, batch_sim in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_sim = batch_sim.to(device)

            word_label = torch.where(
                batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:],
            )
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            concepts, unsup, vocabs, _ = cbl(features.float())

            mask = (batch["attention_mask"][:, :-1] != 0).reshape(-1)
            c_slice = concepts[:, :-1, :].contiguous().view(-1, concepts.shape[-1])
            sim_slice = batch_sim.unsqueeze(1).expand(-1, concepts.shape[1] - 1, -1).contiguous().view(-1, batch_sim.shape[-1])

            concept_loss = -cos_sim_cubed(c_slice[mask], sim_slice[mask])
            word_loss = F.cross_entropy(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
            reg = elastic_net_penalty(cbl.fc.weight[:, : len(concept_set)])

            loss = args.concept_loss_weight * concept_loss + args.word_loss_weight * word_loss + args.elastic_net_alpha * reg

            opt_lm.zero_grad()
            opt_cbl.zero_grad()
            loss.backward()
            opt_lm.step()
            opt_cbl.step()

            losses["concept"].append(concept_loss.item())
            losses["word"].append(word_loss.item())
            losses["reg"].append(reg.item())

        avg = {k: np.mean(v) for k, v in losses.items()}
        total = avg["concept"] + avg["word"]
        print(f"Epoch {epoch + 1}: concept={avg['concept']:.4f} word={avg['word']:.4f} reg={avg['reg']:.4f}")

        preLM.save_pretrained(str(checkpoint_dir / f"llama3_epoch_{epoch + 1}"))
        torch.save(cbl.state_dict(), str(checkpoint_dir / f"cbl_epoch_{epoch + 1}.pt"))

        if total < best_loss:
            best_loss = total
            best_epoch = epoch + 1

    elapsed = (time.time() - start_time) / 3600
    print(f"Training complete in {elapsed:.2f} hours. Best epoch: {best_epoch}")

    # --- Post-training: generate TruthfulQA answers ---
    del preLM, cbl, opt_lm, opt_cbl
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading best checkpoint for generation...")
    preLM = LlamaModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    preLM.load_adapter(str(checkpoint_dir / f"llama3_epoch_{best_epoch}"))
    preLM.eval()

    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(str(checkpoint_dir / f"cbl_epoch_{best_epoch}.pt"), map_location=device))
    cbl.eval()

    # Generate steered responses via CBM intervention (boost truthful concepts)
    output_dir = RESULTS_DIR / "raw_outputs" / "CBM"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"CBM-l0-CBM-seed{args.seed}-TruthfulQA-seed{args.seed}.jsonl"

    print("Generating TruthfulQA responses with CBM steering...")
    intervention = [0] * len(concept_set)
    for i in range(len(concept_set) // 2):
        intervention[i] = 100

    all_prompts, all_outputs = [], []
    for split_idx in [0, 1]:
        from data_prep import load_questions
        questions = load_questions(split_idx)

        for q in tqdm(questions, desc=f"Generating (split {split_idx})"):
            prompt_text = f"{TRUTHFULQA_SYSTEM_PROMPT}\nQ: {q}"
            input_ids = torch.tensor([tokenizer.encode(prompt_text)]).to(device)

            with torch.no_grad():
                gen_ids, _ = cbl.generate_batch(
                    input_ids, preLM, num_samples=1,
                    intervene=intervention, length=50,
                )
            output_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            output_text = output_text.split("\nQ:")[0].strip()
            all_prompts.append(q)
            all_outputs.append(output_text)

    with open(output_dir / filename, "w") as f:
        for prompt, output in zip(all_prompts, all_outputs):
            f.write(json.dumps({
                "prompt": prompt,
                "output": output,
                "generator": f"CBM-seed{args.seed}",
                "dataset": "TruthfulQA",
            }) + "\n")

    print(f"Saved {len(all_outputs)} outputs to {filename}")

    del preLM, cbl
    gc.collect()
    torch.cuda.empty_cache()

    # --- Evaluate ---
    if args.evaluate:
        from evaluate import evaluate_from_jsonl_list
        eval_path = RESULTS_DIR / "eval_results" / "stat_results" / f"CBM-TruthfulQA-seed{args.seed}.csv"
        evaluate_from_jsonl_list([output_dir / filename], eval_path, display=True, seed=args.seed)


if __name__ == "__main__":
    main()
