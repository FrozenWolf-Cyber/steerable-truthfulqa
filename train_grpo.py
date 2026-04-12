"""
GRPO training on TruthfulQA — adapted from CB-LLMs/generation/train_grpo_finegrained_llm.py.

Loads a pretrained CBM checkpoint (from train_cbm.py), then applies Group Relative
Policy Optimization with MPNet-based steerability rewards. After training, generates
TruthfulQA answers and evaluates with the common evaluate.py.

Usage:
  python train_grpo.py --pretrained_path ./checkpoints/cbm-tqa-seed42-... --seed 42
  python train_grpo.py --pretrained_path ... --seed 42 --evaluate
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
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
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


def find_best_epoch(checkpoint_dir: Path) -> int:
    cbl_files = sorted(checkpoint_dir.glob("cbl_epoch_*.pt"))
    if not cbl_files:
        raise FileNotFoundError(f"No CBL checkpoints in {checkpoint_dir}")
    return max(
        int(f.stem.replace("cbl_epoch_", "")) for f in cbl_files
    )


def main():
    parser = argparse.ArgumentParser(description="GRPO training on TruthfulQA")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to pretrained CBM checkpoint directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grpo_epochs", type=int, default=1)
    parser.add_argument("--grpo_num_trajectories", type=int, default=4)
    parser.add_argument("--grpo_gen_length", type=int, default=100)
    parser.add_argument("--grpo_lr", type=float, default=1e-5)
    parser.add_argument("--grpo_kl_weight", type=float, default=0.1)
    parser.add_argument("--grpo_clip_advantage", type=float, default=5.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concept_set = TRUTHFULQA_CONCEPTS
    pretrained_dir = Path(args.pretrained_path)
    best_epoch = find_best_epoch(pretrained_dir)
    peft_path = str(pretrained_dir / f"llama3_epoch_{best_epoch}")
    cbl_path = str(pretrained_dir / f"cbl_epoch_{best_epoch}.pt")
    print(f"Loading pretrained: epoch={best_epoch}")

    run_name = args.run_name or f"grpo-tqa-seed{args.seed}-{int(time.time())}"
    checkpoint_dir = CHECKPOINTS_DIR / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Build training data ---
    print("Loading TruthfulQA data...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    splits = ds.train_test_split(test_size=0.5, seed=42)
    train_df = splits["train"].to_pandas()

    train_texts = []
    for _, row in train_df.iterrows():
        for ans in row.correct_answers:
            train_texts.append(f"Q: {row.question.strip()}\nA: {ans}")

    config = LlamaConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    )

    # --- Reference model (frozen) ---
    print("Loading reference model (frozen)...")
    ref_preLM = LlamaModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    ref_preLM.load_adapter(peft_path)
    ref_preLM.eval()
    for p in ref_preLM.parameters():
        p.requires_grad = False

    ref_cbl = CBL(config, len(concept_set), tokenizer).to(device)
    ref_cbl.load_state_dict(torch.load(cbl_path, map_location=device), strict=False)
    ref_cbl.eval()
    for p in ref_cbl.parameters():
        p.requires_grad = False

    # --- Policy model (trainable) ---
    print("Loading policy model (trainable)...")
    preLM = LlamaModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)

    adapter_path = os.path.join(peft_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        from safetensors.torch import load_file
        adapter_state = load_file(adapter_path, device=str(device))
    else:
        adapter_path = os.path.join(peft_path, "adapter_model.bin")
        adapter_state = torch.load(adapter_path, map_location=device)
    set_peft_model_state_dict(preLM, adapter_state)

    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device), strict=False)

    opt_lm = torch.optim.Adam(filter(lambda p: p.requires_grad, preLM.parameters()), lr=args.grpo_lr)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=args.grpo_lr)

    # --- MPNet for reward scoring ---
    print("Loading MPNet for reward scoring...")
    sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    sim_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    sim_model.eval()
    for p in sim_model.parameters():
        p.requires_grad = False

    concept_enc = sim_tokenizer(concept_set, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        concept_feats = sim_model(**concept_enc)
        concept_feats = mean_pooling(concept_feats.last_hidden_state, concept_enc["attention_mask"])
        concept_feats = F.normalize(concept_feats, p=2, dim=1)

    intervention_value = 100
    special_tokens = torch.tensor([128000, 128001]).to(device)

    # --- GRPO Training ---
    print("Starting GRPO training...")
    start_time = time.time()
    num_concepts = len(concept_set)
    active_indices = list(range(num_concepts))

    for epoch in range(args.grpo_epochs):
        preLM.train()
        cbl.train()
        step_losses = {"policy": [], "kl": [], "total": [], "reward": []}

        for step_idx in tqdm(range(len(active_indices) * 2), desc=f"GRPO Epoch {epoch + 1}"):
            concept_idx = active_indices[step_idx % len(active_indices)]
            intervene = [0] * num_concepts
            intervene[concept_idx] = intervention_value

            gen_input = torch.tensor([tokenizer.encode("")]).to(device)

            # Phase 1: generate trajectories (no grad)
            preLM.eval()
            cbl.eval()
            with torch.no_grad():
                gen_ids, _ = cbl.generate_batch(
                    gen_input, preLM, num_samples=args.grpo_num_trajectories,
                    intervene=intervene, length=args.grpo_gen_length,
                )
                decoded = []
                re_encoded = []
                for g in range(args.grpo_num_trajectories):
                    text = tokenizer.decode(gen_ids[g][~torch.isin(gen_ids[g], special_tokens)])
                    decoded.append(text)
                    re_encoded.append(torch.tensor([tokenizer.encode(text)]).to(device).detach())

                # MPNet reward scoring
                rewards = [0.0] * args.grpo_num_trajectories
                non_empty = [g for g, t in enumerate(decoded) if t.strip()]
                if non_empty:
                    ne_texts = [decoded[g] for g in non_empty]
                    ne_enc = sim_tokenizer(ne_texts, return_tensors="pt", truncation=True, max_length=args.max_length, padding=True).to(device)
                    ne_out = sim_model(**ne_enc)
                    ne_feats = mean_pooling(ne_out.last_hidden_state, ne_enc["attention_mask"])
                    ne_feats = F.normalize(ne_feats, p=2, dim=1)
                    sims = ne_feats @ concept_feats.T
                    v_target = torch.zeros(len(ne_texts), num_concepts, device=device)
                    v_target[:, concept_idx] = 1.0
                    for rank, g in enumerate(non_empty):
                        rewards[g] = cos_sim_cubed(sims[rank:rank+1], v_target[rank:rank+1].float()).item()

            preLM.train()
            cbl.train()

            # Phase 2: advantages
            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            if rewards_t.std() > 1e-8:
                advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            else:
                advantages = torch.zeros_like(rewards_t)
            advantages = advantages.clamp(-args.grpo_clip_advantage, args.grpo_clip_advantage)

            # Phase 3: policy gradient with KL
            valid = [g for g in range(args.grpo_num_trajectories)
                     if re_encoded[g].shape[1] > 1 and advantages[g].abs().item() > 1e-8]

            if not valid:
                continue

            max_len = max(re_encoded[g].shape[1] for g in valid) - 1
            batch_in, batch_tgt, batch_attn, valid_adv = [], [], [], []
            for g in valid:
                seq = re_encoded[g]
                sl = seq.shape[1] - 1
                pad = max_len - sl
                inp = F.pad(seq[:, :-1], (0, pad), value=tokenizer.pad_token_id)
                tgt = F.pad(seq[:, 1:], (0, pad), value=0)
                attn = torch.cat([torch.ones(1, sl, device=device), torch.zeros(1, pad, device=device)], dim=1) if pad > 0 else torch.ones(1, sl, device=device)
                batch_in.append(inp)
                batch_tgt.append(tgt)
                batch_attn.append(attn)
                valid_adv.append(advantages[g])

            batch_in = torch.cat(batch_in, dim=0)
            batch_tgt = torch.cat(batch_tgt, dim=0)
            batch_attn = torch.cat(batch_attn, dim=0).long()
            valid_adv = torch.stack(valid_adv)

            # Policy forward
            feats = preLM(input_ids=batch_in, attention_mask=batch_attn).last_hidden_state
            _, unsup, _, _ = cbl(feats.float())
            intervened_t = torch.zeros(len(valid), batch_in.shape[1], num_concepts, device=device)
            intervened_t[:, :, concept_idx] = intervention_value
            vocab_logits = cbl.intervene(unsup, intervened_t)
            log_probs = F.log_softmax(vocab_logits, dim=-1)
            token_lp = log_probs.gather(2, batch_tgt.unsqueeze(-1)).squeeze(-1) * batch_attn.float()
            mean_lp = token_lp.sum(dim=1) / batch_attn.sum(dim=1).float()

            # Reference forward
            with torch.no_grad():
                ref_feats = ref_preLM(input_ids=batch_in, attention_mask=batch_attn).last_hidden_state
                _, ref_unsup, _, _ = ref_cbl(ref_feats.float())
                ref_intervened = torch.zeros_like(intervened_t)
                ref_intervened[:, :, concept_idx] = intervention_value
                ref_vocab = ref_cbl.intervene(ref_unsup, ref_intervened)
                ref_log_probs = F.log_softmax(ref_vocab, dim=-1)

            # KL divergence
            policy_probs = F.softmax(vocab_logits, dim=-1)
            kl_per_token = (policy_probs * (log_probs - ref_log_probs)).sum(dim=-1) * batch_attn.float()
            kl_loss = (kl_per_token.sum(dim=1) / batch_attn.sum(dim=1).float()).mean()

            policy_loss = (-valid_adv * mean_lp).mean()
            reg = elastic_net_penalty(cbl.fc.weight[:, :num_concepts])
            total_loss = policy_loss + args.grpo_kl_weight * kl_loss + reg

            opt_lm.zero_grad()
            opt_cbl.zero_grad()
            total_loss.backward()
            opt_lm.step()
            opt_cbl.step()

            step_losses["policy"].append(policy_loss.item())
            step_losses["kl"].append(kl_loss.item())
            step_losses["total"].append(total_loss.item())
            step_losses["reward"].append(rewards_t.mean().item())

            del feats, unsup, vocab_logits, log_probs, ref_feats, ref_unsup, ref_vocab, ref_log_probs
            gc.collect()
            torch.cuda.empty_cache()

        avg = {k: np.mean(v) if v else 0.0 for k, v in step_losses.items()}
        print(f"Epoch {epoch + 1}: policy={avg['policy']:.4f} kl={avg['kl']:.4f} reward={avg['reward']:.4f}")

        preLM.save_pretrained(str(checkpoint_dir / f"llama3_epoch_{epoch + 1}"))
        torch.save(cbl.state_dict(), str(checkpoint_dir / f"cbl_epoch_{epoch + 1}.pt"))

    elapsed = (time.time() - start_time) / 3600
    print(f"GRPO training done in {elapsed:.2f} hours")

    # --- Post-training: generate TruthfulQA answers ---
    del preLM, cbl, ref_preLM, ref_cbl, opt_lm, opt_cbl
    gc.collect()
    torch.cuda.empty_cache()

    best_epoch = find_best_epoch(checkpoint_dir)
    preLM = LlamaModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    preLM.load_adapter(str(checkpoint_dir / f"llama3_epoch_{best_epoch}"))
    preLM.eval()

    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(str(checkpoint_dir / f"cbl_epoch_{best_epoch}.pt"), map_location=device))
    cbl.eval()

    output_dir = RESULTS_DIR / "raw_outputs" / "GRPO"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"GRPO-l0-GRPO-seed{args.seed}-TruthfulQA-seed{args.seed}.jsonl"

    print("Generating TruthfulQA responses with GRPO model...")
    intervention = [0] * num_concepts
    for i in range(num_concepts // 2):
        intervention[i] = intervention_value

    all_prompts, all_outputs = [], []
    for split_idx in [0, 1]:
        from data_prep import load_questions
        questions = load_questions(split_idx)
        for q in tqdm(questions, desc=f"Generating (split {split_idx})"):
            prompt_text = f"{TRUTHFULQA_SYSTEM_PROMPT}\nQ: {q}"
            input_ids = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
            with torch.no_grad():
                gen_ids, _ = cbl.generate_batch(
                    input_ids, preLM, num_samples=1, intervene=intervention, length=50,
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
                "generator": f"GRPO-seed{args.seed}",
                "dataset": "TruthfulQA",
            }) + "\n")

    print(f"Saved {len(all_outputs)} outputs to {filename}")

    del preLM, cbl
    gc.collect()
    torch.cuda.empty_cache()

    if args.evaluate:
        from evaluate import evaluate_from_jsonl_list
        eval_path = RESULTS_DIR / "eval_results" / "stat_results" / f"GRPO-TruthfulQA-seed{args.seed}.csv"
        evaluate_from_jsonl_list([output_dir / filename], eval_path, display=True, seed=args.seed)


if __name__ == "__main__":
    main()
