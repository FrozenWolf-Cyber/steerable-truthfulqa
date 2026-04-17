import argparse
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm

import config_finegrained as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBLResidual, CBL, Roberta_classifier
from utils import (
    elastic_net_penalty,
    mean_pooling,
    eos_pooling,
    cos_sim_cubed,
    load_jsonl_as_dataset,
    _normalize_fever_label,
    _apply_fever_concept_mask,
    _align_fever_dataset_to_llamacpp_claims,
    _run_llamacpp_concept_cosine_eval,
    build_intervened_concepts_from_similarity,
)
from steerability_cache import save_all_steerability_texts, steerability_output_root
from eval_metrics import (
    set_seed,
    get_intervention_value,
    generate_steerability_texts,
    run_steerability_mpnet,
    run_concept_accuracy_cosine,
    run_weight_analysis,
    generate_perplexity_texts,
    compute_perplexity,
    load_reward_model,
    run_rm_metrics,
    run_steerability_llamacpp_judge,
    run_truthfulqa_evaluation_for_cbm,
)
import wandb


parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FEVER-only (this script is now wired to annotate_llamacpp.py outputs)
parser.add_argument("--dataset", type=str, default="fever", choices=["fever"])
parser.add_argument(
    "--fever_train_jsonl",
    type=str,
    default="fever_train.jsonl",
    help="Path to FEVER train.jsonl (same source/order as used for annotate_llamacpp.py).",
)
parser.add_argument(
    "--fever_test_jsonl",
    type=str,
    default="fever_paper_test.jsonl",
    help="Path to FEVER paper_test.jsonl (same source/order as used for annotate_llamacpp.py).",
)
parser.add_argument(
    "--fever_max_train_samples",
    type=int,
    default=0,
    help="Optional: truncate FEVER train jsonl to first N rows (0/<=0 disables).",
)
parser.add_argument(
    "--fever_max_test_samples",
    type=int,
    default=0,
    help="Optional: truncate FEVER test jsonl to first N rows (0/<=0 disables).",
)

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epoch_multiplier", type=int, default=1, help="Epoch multiplier to increase total training steps (for debugging).")
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--samples_per_concept",
    type=int,
    default=50,
    help="Steerability evaluation: samples per concept. Default 50.",
)

parser.add_argument("--discrimination_loss", type=float, default=1.0)
parser.add_argument("--neg_entropy_loss", type=float, default=1.0)
parser.add_argument("--concept_loss", type=float, default=1.0)
parser.add_argument("--word_loss", type=float, default=1.0)
parser.add_argument("--elastic_net_alpha", type=float, default=1.0)
parser.add_argument("--residual_dim", type=int, default=768)
parser.add_argument("--orthogonal_loss_weight", type=float, default=0)
parser.add_argument("--residual_penalty_weight", type=float, default=0)
parser.add_argument("--DEBUG", action='store_true', help="If set, use a smaller subset of data for quick debugging.")
parser.add_argument("--intervention_gen_loss", type=float, default=0.0)
parser.add_argument("--intervention_margin", type=float, default=10.0)
parser.add_argument("--no_detach_intervention", action='store_true', help="If set, do not detach unsup during intervention generation loss computation.")
parser.add_argument("--intervention_spread", type=float, default=2.0)
parser.add_argument(
    "--intervention_keep_other_concepts",
    action="store_true",
    help="If set, intervention overwrites only the selected concept(s) and keeps all other concept activations as-is (instead of setting them to 0).",
)
parser.add_argument(
    "--intervention_topk_concepts",
    action="store_true",
    help=(
        "If set, use stochastic top-K selection when constructing the intervention tensor: "
        "take the top-K concepts by similarity, softmax their scores, sample a rank, and intervene on the top-(rank+1) concepts. "
        "If not set, intervene on only the top-1 concept."
    ),
)
parser.add_argument(
    "--intervention_topk_k",
    type=int,
    default=3,
    help="K for --intervention_topk_concepts (default 3).",
)


parser.add_argument("--concept_loss_type", type=str, default="cosine_cubed", help="Type of concept loss to use: 'cosine_cubed' or 'ce'.")

# Label sources
parser.add_argument("--labeling", type=str, default="llamacpp", choices=["llamacpp", "mpnet", "angle", "simcse", "llm"], help="Concept label source")
parser.add_argument(
    "--use_class_concepts",
    action="store_true",
    help=(
        "If set, ignore finegrained concept vectors/labels and instead train with 3 concepts that are the dataset classes "
        "(FEVER: SUPPORTS/REFUTES/NOT ENOUGH INFO), using one-hot supervision derived from the `label` field."
    ),
)
parser.add_argument(
    "--llamacpp_train_vectors",
    type=str,
    default="fever_concept_vectors_llamacpp.npy",
    help="Path to numpy array saved by annotate_llamacpp.py for FEVER train (N, C).",
)
parser.add_argument(
    "--llamacpp_train_claims",
    type=str,
    default="",
    help="Path to *_claims_llamacpp.npy for FEVER train (required when --labeling=llamacpp).",
)
parser.add_argument(
    "--llamacpp_val_vectors",
    type=str,
    default="",
    help="Optional path to numpy array saved by annotate_llamacpp.py for FEVER test/eval (N, C).",
)
parser.add_argument(
    "--llamacpp_val_claims",
    type=str,
    default="",
    help="Path to *_claims_llamacpp.npy for FEVER test/eval (required when --llamacpp_val_vectors is set).",
)
parser.add_argument(
    "--no_zero_out_nonclass_concepts",
    action="store_true",
    help="Disable FEVER label-based masking of non-class concepts in concept/intervention losses.",
)
parser.add_argument(
    "--skip_mpnet_eval",
    action="store_true",
    help="Skip MPNet-based steerability evaluation.",
)
parser.add_argument("--use_last_epoch", action='store_true', help="If set, load the classifier from the last epoch instead of the best epoch based on validation loss.")
parser.add_argument(
    "--add_llama_logits",
    action="store_true",
    help=(
        "If set, add the original Llama vocab projection logits (from the backbone hidden states) to the CBL/CBLResidual logits. "
        "This keeps CBL unchanged (no extra parameters) and acts like a residual-on-logits."
    ),
)
parser.add_argument("--rm_model_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    help="HF id for sequence-classification reward model.")
parser.add_argument("--rm_batch_size", type=int, default=0, help="0 = score all texts per chunk in one forward.")
parser.add_argument("--rm_max_text_len", type=int, default=500)
parser.add_argument("--skip_rm", action="store_true", help="Skip RM reward evaluation after training.")
parser.add_argument(
    "--skip_llamacpp_steer_eval",
    action="store_true",
    help="Skip llama.cpp judge-based steerability evaluation.",
)
parser.add_argument(
    "--llamacpp_eval_model_repo_id",
    type=str,
    default="unsloth/Qwen3.5-27B-GGUF",
    help="HF repo id for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_model_filename",
    type=str,
    default="Qwen3.5-27B-Q8_0.gguf",
    help="GGUF filename for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_n_ctx",
    type=int,
    default=2048,
    help="Context size for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_max_tokens",
    type=int,
    default=64,
    help="Max tokens for llama.cpp judge output.",
)
parser.add_argument(
    "--llamacpp_eval_repeat_penalty",
    type=float,
    default=1.15,
    help="Repeat penalty for llama.cpp steerability judge.",
)
parser.add_argument(
    "--llamacpp_eval_temperature",
    type=float,
    default=0.1,
    help="Temperature for llama.cpp steerability judge.",
)
parser.add_argument(
    "--skip_truthfulqa_final_test",
    action="store_true",
    help="Skip final TruthfulQA generation + judge evaluation.",
)
parser.add_argument(
    "--truthfulqa_data_dir",
    type=str,
    default="",
    help="Optional TruthfulQA data directory override.",
)
parser.add_argument(
    "--truthfulqa_results_root",
    type=str,
    default="",
    help="Optional root directory for TruthfulQA raw outputs and eval CSV.",
)
parser.add_argument(
    "--truthfulqa_max_new_tokens",
    type=int,
    default=50,
    help="Max new tokens for final TruthfulQA generation.",
)
parser.add_argument(
    "--truthfulqa_temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for final TruthfulQA generation.",
)
parser.add_argument(
    "--truthfulqa_top_p",
    type=float,
    default=0.9,
    help="Top-p for final TruthfulQA generation.",
)
parser.add_argument(
    "--truthfulqa_top_k",
    type=int,
    default=100,
    help="Top-k for final TruthfulQA generation.",
)
parser.add_argument(
    "--truthfulqa_repetition_penalty",
    type=float,
    default=1.1,
    help="Repetition penalty for final TruthfulQA generation.",
)


class ClassificationDataset(torch.utils.data.Dataset):
    """Thin wrapper around a HF Dataset + numpy supervision array.

    train_combined_finegrained.py previously assumed `encoded_text` was a dict of lists.
    Here `encoded_text` is a `datasets.Dataset`, so we index per-row.
    """

    def __init__(self, encoded_dataset, s):
        self.encoded_dataset = encoded_dataset
        self.s = s

    def __getitem__(self, idx):
        row = self.encoded_dataset[int(idx)]
        t = {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
        }
        y = torch.tensor(self.s[int(idx)], dtype=torch.float32)
        return t, y

    def __len__(self):
        return len(self.encoded_dataset)


def build_loaders(encoded_dataset, s, mode):
    dataset = ClassificationDataset(encoded_dataset, s)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    set_seed(args.seed)

    wandb.init(project="cbm-generation-new", name=f"finegrained-{args.dataset}-seed{args.seed}",
               config=vars(args))
    
    run_name = wandb.run.id

    # ─────────────────────────────────────────────────────────────
    # FEVER data loading (local jsonl) + llama.cpp alignment
    # ─────────────────────────────────────────────────────────────
    print("loading FEVER jsonl...")

    # Load jsonl exactly like annotate_llamacpp.py (preserve file order; optional truncation).
    train_dataset = load_jsonl_as_dataset(args.fever_train_jsonl, max_samples=args.fever_max_train_samples)
    test_dataset = load_jsonl_as_dataset(args.fever_test_jsonl, max_samples=args.fever_max_test_samples)

    # Ensure FEVER labels are ints: 0=SUPPORTS, 1=REFUTES, 2=NEI.
    train_dataset = train_dataset.map(lambda e: {"label": _normalize_fever_label(e.get("label", 2))})
    test_dataset = test_dataset.map(lambda e: {"label": _normalize_fever_label(e.get("label", 2))})

    # If using llama.cpp labels, align jsonl rows to the exact annotation order using *_claims_llamacpp.npy.
    train_similarity = None
    test_similarity_llamacpp = None
    train_claims_np = None
    test_claims_np = None

    if args.labeling == "llamacpp" and (not args.use_class_concepts):
        print(f"Loading llama.cpp concept vectors from: {args.llamacpp_train_vectors}")
        train_similarity = np.load(args.llamacpp_train_vectors)

        train_claims_path = args.llamacpp_train_claims
        print(f"Loading llama.cpp claims from: {train_claims_path}")
        train_claims_np = np.load(train_claims_path, allow_pickle=True)
        train_dataset, train_similarity = _align_fever_dataset_to_llamacpp_claims(
            train_dataset, train_similarity, train_claims_np, split_name="train"
        )

        if args.llamacpp_val_vectors:
            print(f"Loading llama.cpp eval concept vectors from: {args.llamacpp_val_vectors}")
            test_similarity_llamacpp = np.load(args.llamacpp_val_vectors)

            test_claims_path = args.llamacpp_val_claims
            print(f"Loading llama.cpp eval claims from: {test_claims_path}")
            test_claims_np = np.load(test_claims_path, allow_pickle=True)
            test_dataset, test_similarity_llamacpp = _align_fever_dataset_to_llamacpp_claims(
                test_dataset, test_similarity_llamacpp, test_claims_np, split_name="test"
            )

    print("training data len: ", len(train_dataset))
    print("test data len: ", len(test_dataset))

    print("tokenizing...")

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token

    def _tok(batch):
        # Use fixed-length padding so all rows share the same shape across map batches.
        # This matches the original CB-LLMs assumption that tensors are stackable in the DataLoader.
        return tokenizer(batch["claim"], padding="max_length", truncation=True, max_length=args.max_length)

    encoded_train_dataset = train_dataset.map(_tok, batched=True, batch_size=1024)
    encoded_test_dataset = test_dataset.map(_tok, batched=True, batch_size=1024)

    # Keep only tensors + label
    keep_cols = {"input_ids", "attention_mask", "label"}
    encoded_train_dataset = encoded_train_dataset.remove_columns([c for c in encoded_train_dataset.column_names if c not in keep_cols])
    encoded_test_dataset = encoded_test_dataset.remove_columns([c for c in encoded_test_dataset.column_names if c not in keep_cols])

    concept_set = CFG.concept_set[args.dataset]
    concept_set_for_similarity = concept_set
    if args.use_class_concepts:
        if args.dataset != "fever":
            raise ValueError("--use_class_concepts is only supported for dataset='fever' in this script.")
        # Keep short names for WandB keys / cache bookkeeping, but use descriptive holders
        # for embedding-based similarity evaluation.
        concept_set = CFG.FEVER_LABEL_NAMES
        concept_set_for_similarity = CFG.FEVER_LABEL_CONCEPTS
    print("concept len: ", len(concept_set))  # concept_set: list of strings, len = num_concepts

    d_name = args.dataset.replace('/', '_')
    label_prefix = "./"
    val_similarity = None

    # Concept labels / supervision vectors
    # train_similarity must be shaped (N_train, C) where C == len(concept_set).
    if args.use_class_concepts:
        train_labels = np.asarray(encoded_train_dataset["label"], dtype=np.int64)
        # Safety: clamp into [0, 2] in case of weird labels.
        train_labels = np.clip(train_labels, 0, 2)
        train_similarity = np.zeros((len(train_labels), 3), dtype=np.float32)
        train_similarity[np.arange(len(train_labels)), train_labels] = 1.0
        print("train_similarity shape (class one-hot): ", train_similarity.shape)
    elif args.labeling == "llamacpp":
        # llama.cpp labels are direct concept vectors.
        label_prefix = os.path.dirname(os.path.abspath(args.llamacpp_train_vectors)) or "."
        if train_similarity is None:
            train_similarity = np.load(args.llamacpp_train_vectors)
        print("train_similarity shape: ", train_similarity.shape)

        # Optional llama.cpp eval vectors (for post-training analysis).
        val_similarity = test_similarity_llamacpp
        if val_similarity is not None:
            print("val/test_similarity(llamacpp) shape: ", val_similarity.shape)
    else:
        if args.labeling == 'mpnet':
            label_prefix += "mpnet_acs"
        elif args.labeling == 'simcse':
            label_prefix += "simcse_acs"
        elif args.labeling == 'angle':
            label_prefix += "angle_acs"
        elif args.labeling == 'llm':
            label_prefix += "llm_labeling"

        label_prefix += "/" + d_name + "/"
        print(f"Loading concept labels from: {label_prefix}")
        train_similarity = np.load(label_prefix + "/concept_labels_train.npy")  # (N_train, num_concepts)
        print("train_similarity shape: ", train_similarity.shape)

    # Require exact alignment between concept-label rows and tokenized dataset rows.
    assert int(np.asarray(train_similarity).shape[0]) == len(encoded_train_dataset), (
        f"train: concept-label rows ({int(np.asarray(train_similarity).shape[0])}) != tokenized dataset rows ({len(encoded_train_dataset)})"
    )
    if val_similarity is not None:
        assert int(np.asarray(val_similarity).shape[0]) == len(encoded_test_dataset), (
            f"test: concept-label rows ({int(np.asarray(val_similarity).shape[0])}) != tokenized dataset rows ({len(encoded_test_dataset)})"
        )

    # Basic shape sanity checks.
    if train_similarity.ndim != 2 or train_similarity.shape[1] != len(concept_set):
        raise ValueError(
            f"Unexpected train_similarity shape {train_similarity.shape}; expected (N, {len(concept_set)}). "
            f"Check concept vectors / labels and config_finegrained.concept_set for {args.dataset}."
        )

    if args.dataset == "fever" and (not args.no_zero_out_nonclass_concepts) and (not args.use_class_concepts):
        start = time.time()
        print("Applying FEVER label-based concept masking (zeroing non-class concepts)...")
        train_labels = np.asarray(encoded_train_dataset["label"])
        train_similarity = _apply_fever_concept_mask(train_similarity, train_labels)
        if val_similarity is not None:
            test_labels = np.asarray(encoded_test_dataset["label"])
            val_similarity = _apply_fever_concept_mask(val_similarity, test_labels)
        end = time.time()
        print("time of masking:", (end - start) / 3600, "hours")

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")

    # test_loader is used for post-training analyses; it does not require labels.
    test_similarity = np.zeros((len(encoded_test_dataset["label"]), len(concept_set)), dtype=np.float32)
    test_loader = build_loaders(encoded_test_dataset, test_similarity, mode="test")

    print("preparing backbone")
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=5e-5)

    llama_vocab_weight = None
    if args.add_llama_logits:
        # IMPORTANT: For Llama-3, lm_head weights are not necessarily tied to input embeddings.
        # We therefore grab the *output* projection (lm_head) weights from a CausalLM head.
        # This does not add parameters to CBL; it's just an external tensor used in forward.
        lm_head_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3-8B',
            torch_dtype=torch.bfloat16,
        ).to(device)
        llama_vocab_weight = lm_head_model.get_output_embeddings().weight.detach()
        del lm_head_model
    
    if args.discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=5e-5)
    print("preparing classifier")
    total_params = sum(p.numel() for p in preLM.parameters())
    trainable_params = sum(p.numel() for p in preLM.parameters() if p.requires_grad)
    cbl_params = sum(p.numel() for p in cbl.parameters())
    trainable_params += cbl_params
    total_params += cbl_params
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} = {trainable_params/total_params:.4f} of total")
    wandb.log({"trainable_parameters": trainable_params, "trainable_ratio": trainable_params/total_params})
    
    classifier = torch.nn.Linear(args.residual_dim, len(concept_set)).to(device)
    
    if args.discrimination_loss > 0:
        opt_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)


    intervention_value = 100


    print("start training...")
    best_loss = float('inf')
    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    prefix += "./from_pretained_llama3_lora_cbm_" + run_name
    prefix += "/"
    prefix += d_name
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "llama3"
    cbl_name = "cbl"



    start = time.time()
    best_epoch = -1
    epochs = CFG.epoch[args.dataset]*args.epoch_multiplier
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        cbl.train()
        classifier.train()
        training_losses = {
            "concept_loss": [],
            "word_loss": [],
            "neg_entropy_loss": [],
            "reg_loss": [],
            "orthogonal_loss": [],
            "residual_penalty_loss": [],
            "intervention_gen_loss": [],
        }

        
        for i, (batch, batch_sim) in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_sim = batch_sim.to(device)

            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            llama_logits = F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
            concepts, unsup, vocabs, matched_unsup = cbl(features.float(), llama_logits=llama_logits)
            # print("concepts shape in training loop:", concepts.shape)
            # print("elastic_net_alphaunsup shape in training loop:", unsup.shape)
            # print("vocabs shape in training loop:", vocabs.shape)
            
            mask = (batch["attention_mask"][:, :-1] != 0).reshape(-1) # (B * (seq_len - 1))
            c_slice = concepts[:, :-1, :].contiguous().view(-1, concepts.shape[-1]) # (B * (seq_len - 1), C)
            batch_sim_slice = batch_sim.unsqueeze(1).expand(-1, concepts.shape[1] - 1, -1).contiguous().view(-1, batch_sim.shape[-1])
            
            valid_c = c_slice[mask]          # (N_valid, C)
            valid_sim = batch_sim_slice[mask]  # (N_valid, C)

            if args.concept_loss_type == "cosine_cubed":
                # Cosine-similarity-based concept loss against soft ACS labels
                concept_loss = -cos_sim_cubed(valid_c, valid_sim)
            elif args.concept_loss_type == "ce":
                # Cross-entropy concept loss using hard labels from ACS top concept
                hard_targets = torch.argmax(valid_sim, dim=-1)  # (N_valid,)
                concept_loss = torch.nn.CrossEntropyLoss()(valid_c, hard_targets)
            else:
                raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")
            word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
            loss = args.concept_loss * concept_loss + word_loss*args.word_loss
            reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
            
            if matched_unsup is not None:
                orthogonal_loss = torch.cosine_similarity(concepts, matched_unsup, dim=-1).mean().abs() ## TODO: check shape
                loss += args.orthogonal_loss_weight * orthogonal_loss
                training_losses["orthogonal_loss"].append(orthogonal_loss.detach().cpu().numpy())
            
            if args.residual_penalty_weight > 0:
                residual_contrib = cbl.compute_residual_contrib(unsup)
                residual_penalty = torch.mean(torch.abs(residual_contrib)) ## TODO: check logic
                loss += args.residual_penalty_weight * residual_penalty
                training_losses["residual_penalty_loss"].append(residual_penalty.detach().cpu().numpy())
                
            if args.intervention_gen_loss > 0:
                ### concepts shapes: (B, seq_len, concept_dim)
                intervention_value = 100

                intervened_concept = build_intervened_concepts_from_similarity(
                    concepts=concepts,
                    batch_sim=batch_sim,
                    intervention_value=intervention_value,
                    keep_other_concepts=args.intervention_keep_other_concepts,
                    use_topk=args.intervention_topk_concepts,
                    topk_k=args.intervention_topk_k,
                    log_wandb=args.intervention_topk_concepts,
                    wandb_prefix="train",
                )
                    
                # print("intervened_concept shape: ", intervened_concept.shape, intervened_concept.max(), intervened_concept.min())
                llama_logits_for_intervene = None
                if llama_logits is not None:
                    llama_logits_for_intervene = llama_logits if args.no_detach_intervention else llama_logits.detach()

                if args.no_detach_intervention:
                    vocab = cbl.intervene(unsup, intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
                else:
                    vocab = cbl.intervene(unsup.detach(), intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
                intervention_gen_loss = torch.nn.CrossEntropyLoss()(vocab[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
                loss += args.intervention_gen_loss * intervention_gen_loss
                training_losses["intervention_gen_loss"].append(intervention_gen_loss.detach().cpu().numpy())
                
            loss += args.elastic_net_alpha * reg
            
            
            
            opt_prelm.zero_grad()
            opt_cbl.zero_grad()
            loss.backward()
            opt_prelm.step()
            opt_cbl.step()

            if args.discrimination_loss > 0:
                classification = classifier(mean_pooling(unsup.detach(), batch["attention_mask"]))

                # Probe loss: train the classifier to predict finegrained concept similarities from unsup.
                # This keeps the probe consistent with the concept supervision and avoids class labels.
                if args.concept_loss_type == "cosine_cubed":
                    discrimination_loss = -cos_sim_cubed(classification, batch_sim)
                elif args.concept_loss_type == "ce":
                    hard_targets = torch.argmax(batch_sim, dim=-1)
                    discrimination_loss = torch.nn.CrossEntropyLoss()(classification, hard_targets)
                else:
                    raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")
                opt_classifier.zero_grad()
                (args.discrimination_loss * discrimination_loss).backward(inputs=list(classifier.parameters()))
                opt_classifier.step()

            if args.neg_entropy_loss > 0:
                _, unsup, _, _ = cbl(features.detach().float())
                classification = classifier(mean_pooling(unsup, batch["attention_mask"]))
                p = F.softmax(classification, dim=-1)
                neg_entropy_loss = torch.sum(p * torch.log(p), dim=-1).mean()
                opt_cbl.zero_grad()
                (args.neg_entropy_loss * neg_entropy_loss).backward(inputs=list(cbl.unsup.parameters()))
                opt_cbl.step()
                training_losses["neg_entropy_loss"].append(neg_entropy_loss.detach().cpu().numpy())


            training_losses["concept_loss"].append(concept_loss.detach().cpu().numpy())
            training_losses["word_loss"].append(word_loss.detach().cpu().numpy())
            
            training_losses["reg_loss"].append(reg.detach().cpu().numpy())
            
            log = {}
            for key in training_losses.keys():
                if len(training_losses[key]) > 0:
                    print(f"{key}: {training_losses[key][-1]}", end=" ")
                    log[key] = training_losses[key][-1]
            # print(" | batch ", i+1, " / ", len(train_loader), end="\r")
            
            
            log["epoch"] = e + 1
            log["batch"] = i + 1
            wandb.log(log)
            
            if args.DEBUG and i >= 2:
                break
            
            
        avg_metrics = {}
        for key in training_losses.keys():
            if len(training_losses[key]) > 0:
                avg_metrics[key] = sum(training_losses[key]) / len(training_losses[key])
        print("Epoch ", e + 1, " training losses: ", avg_metrics)
        wandb.log({f"avg_{k}": avg_metrics[k] for k in avg_metrics.keys()})

        print("save model")
        preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
        torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")

        if args.DEBUG:
            break

    end = time.time()
    print("time of training CBM:", (end - start) / 3600, "hours")
    
    ## delete training objects and free GPU before evaluation
    import gc
    if llama_vocab_weight is not None:
        del llama_vocab_weight
        llama_vocab_weight = None
    del preLM, cbl, classifier, opt_prelm, opt_cbl
    
    if args.discrimination_loss > 0:
        del opt_classifier
    gc.collect()
    torch.cuda.empty_cache()
    
    ## lOAD BEST MODEL AND
    if best_epoch == -1:
        best_epoch = epochs
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    peft_path = prefix + model_name + "_epoch_" + str(best_epoch)
    preLM.load_adapter(peft_path)
    preLM.eval()

    llama_vocab_weight = None
    if args.add_llama_logits:
        from eval_metrics import get_llama_vocab_weight
        llama_vocab_weight = get_llama_vocab_weight(device)

    if args.discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    cbl.load_state_dict(torch.load(prefix + cbl_name + "_epoch_" + str(best_epoch) + ".pt", map_location=device))
    cbl.eval()

    # ── Configure evaluation ──
    intervention_value = get_intervention_value(args.dataset)
    num_steerability_samples = (
        max(1, args.samples_per_concept)
        if args.samples_per_concept is not None
        else max(1, 100 // len(concept_set))
    )
    steer_root = steerability_output_root(os.path.normpath(prefix.rstrip("/")), best_epoch, False)
    print(f"Steerability sample cache: {steer_root}")

    # ── Generate steerability texts (cached) ──
    set_seed(args.seed)
    decoded_texts_by_concept = generate_steerability_texts(
        preLM, cbl, tokenizer, concept_set, args.dataset, device,
        samples_per_concept=num_steerability_samples,
        llama_vocab_weight=llama_vocab_weight,
        keep_other_concepts=args.intervention_keep_other_concepts,
        steerability_cache_dir=steer_root,
        steerability_cache_seed=args.seed,
        interventions_per_batch=50,
    )

    # ── Generate perplexity texts (cached) ──
    ppl_texts = generate_perplexity_texts(
        cbl, preLM, tokenizer, args.seed, device,
        cache_dir=prefix, run_name=run_name,
        llama_vocab_weight=llama_vocab_weight,
    )

    # ── Concept accuracy ──
    test_similarity_eval = None
    if args.use_class_concepts:
        test_labels = np.asarray(encoded_test_dataset["label"], dtype=np.int64)
        test_labels = np.clip(test_labels, 0, len(concept_set) - 1)
        test_similarity_eval = np.zeros((len(test_labels), len(concept_set)), dtype=np.float32)
        test_similarity_eval[np.arange(len(test_labels)), test_labels] = 1.0

    run_concept_accuracy_cosine(
        preLM,
        cbl,
        test_loader,
        concept_set,
        label_prefix,
        device,
        test_similarity_np=test_similarity_eval,
        llama_vocab_weight=llama_vocab_weight,
    )

    # If provided, evaluate against llama.cpp vectors directly.
    if args.labeling == "llamacpp" and val_similarity is not None:
        _run_llamacpp_concept_cosine_eval(preLM, cbl, test_loader, val_similarity, device)

    # ── Weight analysis ──
    run_weight_analysis(cbl, concept_set, tokenizer)

    # ── Final test: TruthfulQA generation + judge evaluation ──
    if not args.skip_truthfulqa_final_test:
        try:
            print("Running final TruthfulQA evaluation...")
            run_truthfulqa_evaluation_for_cbm(
                preLM=preLM,
                cbl=cbl,
                tokenizer=tokenizer,
                concept_set=concept_set,
                seed=args.seed,
                batch_size=args.batch_size,
                model_label=f"CBM-Llama3-{args.dataset}",
                layer_idx=best_epoch,
                run_id=run_name,
                use_class_concepts=args.use_class_concepts,
                intervention_goals=None,
                intervention_value=float(intervention_value),
                keep_other_concepts=args.intervention_keep_other_concepts,
                max_new_tokens=args.truthfulqa_max_new_tokens,
                temperature=args.truthfulqa_temperature,
                top_p=args.truthfulqa_top_p,
                top_k=args.truthfulqa_top_k,
                repetition_penalty=args.truthfulqa_repetition_penalty,
                data_dir=(args.truthfulqa_data_dir or None),
                results_root=(args.truthfulqa_results_root or None),
                llama_vocab_weight=llama_vocab_weight,
                display=not args.DEBUG,
            )
        except Exception as truthfulqa_err:
            print(f"TruthfulQA final testing failed (non-fatal): {truthfulqa_err}")
    else:
        print("Skipping final TruthfulQA testing.")

    # ── Free model from GPU ──
    del preLM, cbl
    if llama_vocab_weight is not None:
        from eval_metrics import release_llama_vocab_weight
        release_llama_vocab_weight()
        llama_vocab_weight = None
    gc.collect()
    torch.cuda.empty_cache()

    # ── Steerability scoring (MPNet similarity) ──
    if not args.skip_mpnet_eval and args.labeling != "llamacpp":
        run_steerability_mpnet(
            decoded_texts_by_concept, concept_set_for_similarity,
            intervention_value, args.max_length, device,
        )
    else:
        print("Skipping MPNet steerability evaluation.")

    # ── Steerability scoring (llama.cpp judge) ──
    if not args.skip_llamacpp_steer_eval:
        try:
            run_steerability_llamacpp_judge(
                decoded_texts_by_concept=decoded_texts_by_concept,
                concept_set=concept_set,
                model_repo_id=args.llamacpp_eval_model_repo_id,
                model_filename=args.llamacpp_eval_model_filename,
                n_ctx=args.llamacpp_eval_n_ctx,
                max_tokens=args.llamacpp_eval_max_tokens,
                repeat_penalty=args.llamacpp_eval_repeat_penalty,
                temperature=args.llamacpp_eval_temperature,
            )
        except Exception as llama_eval_err:
            print(f"llama.cpp steerability evaluation failed (non-fatal): {llama_eval_err}")
    else:
        print("Skipping llama.cpp steerability evaluation.")

    # ── Perplexity computation (evaluate library loads its own LLM) ──
    compute_perplexity(ppl_texts)

    # ── RM reward scoring (optional) ──
    if not args.skip_rm:
        try:
            rm_model, rm_tokenizer_rm = load_reward_model(args.rm_model_name, device)
            run_rm_metrics(
                decoded_texts_by_concept, concept_set,
                rm_model, rm_tokenizer_rm, device,
                rm_batch_size=args.rm_batch_size,
                rm_max_text_len=args.rm_max_text_len,
            )
            del rm_model, rm_tokenizer_rm
            torch.cuda.empty_cache()
        except Exception as rm_err:
            print(f"RM evaluation failed (non-fatal): {rm_err}")

    # ── Save steerability text cache ──
    save_all_steerability_texts(steer_root, args.seed, concept_set, decoded_texts_by_concept)
    