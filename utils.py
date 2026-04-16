import torch
import torch.nn.functional as F
import config as CFG

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def eos_pooling(token_embeddings, attention_mask):
    last_index = []
    for i in range(attention_mask.size(0)):
        last_index.append(check_zero(attention_mask[i]))
    last_index = torch.tensor(last_index)
    return token_embeddings[range(len(last_index)), last_index]

def check_zero(mask):
    for i in range(len(mask)):
        if mask[i] == 0:
            return i-1
    return len(mask)-1

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=float('-inf')):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0][indices_to_remove] = filter_value
    return logits

def top_k_top_p_filtering_batched(logits, top_k=0, top_p=0.0, filter_value=float('-inf')):
    """Batched top-k/top-p filtering. logits: (B, vocab_size)"""
    if top_k > 0:
        top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0]
        indices_to_remove = logits < top_k_vals[:, -1:]
        logits = logits.masked_fill(indices_to_remove, filter_value)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

def elastic_net_penalty(param, alpha=0.99):
    return alpha * torch.abs(param).mean() + (1-alpha) * torch.square(param).mean()

def cos_sim_cubed(cbl_features, target, reduce: bool = True):
    """Cosine similarity after centering and cubing.

    Args:
        cbl_features: (..., D)
        target:       (..., D)
        reduce:       If True (default), return mean over the last non-feature dim.
                       If False, return per-sample similarities (no final mean).
    """
    cbl_features = cbl_features - torch.mean(cbl_features, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    cbl_features = F.normalize(cbl_features**3, dim=-1)
    target = F.normalize(target**3, dim=-1)

    sim = torch.sum(cbl_features * target, dim=-1)  # (...,)
    if reduce:
        return sim.mean()
    return sim

def normalize(x, d=-1, mean=None, std=None):
    if mean is not None and std is not None:
        x_mean = mean
        x_std = std
    else:
        x_mean = torch.mean(x, dim=d)
        x_std = torch.std(x, dim=d)
    if d == -1:
        x = x - x_mean.unsqueeze(-1)
        x = x / (x_std.unsqueeze(-1) + 1e-12)
    else:
        x = x - x_mean.unsqueeze(0)
        x = x / (x_std.unsqueeze(0) + 1e-12)
    return x, x_mean, x_std


def load_jsonl_as_dataset(jsonl_path: str, max_samples: int = 0):
    """Load a local JSONL file into a HF Dataset, preserving file order.

    This matches truthful_qa/annotate_llamacpp.py's behavior (read line-by-line json.loads).

    Args:
        jsonl_path: Path to a .jsonl file.
        max_samples: If >0, stop after this many rows.
    """
    import os
    import json

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and max_samples > 0 and len(rows) >= int(max_samples):
                break

    if len(rows) == 0:
        raise ValueError(f"No rows found in JSONL: {jsonl_path}")

    # Lazy import to avoid making `datasets` a hard dependency for every utils consumer.
    from datasets import Dataset

    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────
# FEVER + llama.cpp helpers (train_combined_finegrained.py)
# ─────────────────────────────────────────────────────────────

def _normalize_fever_label(label):
    """Normalize FEVER labels to ints: 0=SUPPORTS, 1=REFUTES, 2=NEI."""
    import numpy as np

    if isinstance(label, (int, np.integer)):
        if int(label) in (0, 1, 2):
            return int(label)
        return 2

    s = str(label).strip().upper()
    if s == "SUPPORTS":
        return 0
    if s == "REFUTES":
        return 1
    return 2


def _apply_fever_concept_mask(similarity, labels, allowed_map=None):
    """Zero-out concept dims not allowed by the FEVER class label, and renormalize rows."""
    import numpy as np

    sim = np.asarray(similarity, dtype=np.float32)
    labels = np.asarray(labels)
    labels = np.vectorize(_normalize_fever_label)(labels).astype(np.int64)

    if allowed_map is None:
        # Lazy import to avoid pulling finegrained config in unrelated scripts.
        import config_finegrained as CFG_FINE
        allowed_map = CFG_FINE.FEVER_LABEL_TO_CONCEPT_INDICES

    mask = np.zeros_like(sim, dtype=np.float32)
    for y in (0, 1, 2):
        idxs = allowed_map.get(int(y), [])
        if len(idxs) == 0:
            continue
        rows = np.flatnonzero(labels == y)
        if rows.size == 0:
            continue
        mask[np.ix_(rows, idxs)] = 1.0

    sim = sim * mask
    row_sums = sim.sum(axis=1, keepdims=True)
    nonzero = row_sums > 0
    sim[nonzero[:, 0]] = sim[nonzero[:, 0]] / row_sums[nonzero]
    return sim






def _align_fever_dataset_to_llamacpp_claims(dataset, vectors, anno_claims, split_name: str):
    """Align FEVER jsonl rows to llama.cpp annotation order."""
    import numpy as np
    from collections import defaultdict

    if "claim" not in dataset.column_names:
        raise ValueError(
            f"Expected a 'claim' column in {split_name} dataset. Found columns={dataset.column_names}"
        )

    ds_claims = dataset["claim"]
    claim_to_indices = defaultdict(list)
    for i, c in enumerate(ds_claims):
        claim_to_indices[str(c)].append(i)

    matched_ds_indices = []
    matched_vec_indices = []

    anno_claims_list = [str(x) for x in np.asarray(anno_claims).tolist()]
    for j, c in enumerate(anno_claims_list):
        q = claim_to_indices.get(c)
        if q:
            matched_ds_indices.append(q.pop(0))
            matched_vec_indices.append(j)

    if len(matched_ds_indices) == 0:
        raise ValueError(
            f"Could not match any llama.cpp claims to {split_name} dataset claims. "
            f"Check that --fever_*_jsonl corresponds to the same source used during annotation."
        )

    if len(matched_ds_indices) != len(anno_claims_list):
        print(
            f"WARNING: Only matched {len(matched_ds_indices)}/{len(anno_claims_list)} llama.cpp annotated claims "
            f"to {split_name} jsonl rows. Unmatched annotations will be dropped."
        )
        raise ValueError(
            f"Could not match all llama.cpp claims to {split_name} dataset claims. "
            f"Check that --fever_*_jsonl corresponds to the same source used during annotation."
        )

    dataset = dataset.select(matched_ds_indices)
    vectors = np.asarray(vectors)[matched_vec_indices]
    return dataset, vectors


def _run_llamacpp_concept_cosine_eval(preLM, cbl, test_loader, eval_vectors, device):
    """Cosine-sim eval between predicted concepts and llama.cpp concept vectors (N, C)."""
    import numpy as np
    import wandb
    from tqdm.auto import tqdm

    if eval_vectors is None:
        return {}

    eval_vectors = np.asarray(eval_vectors, dtype=np.float32)
    preds = []
    for batch, _ in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            concepts, _, _, _ = cbl(features.float())
        pooled = eos_pooling(concepts, batch["attention_mask"]).detach().cpu()
        preds.append(pooled)

    preds = torch.cat(preds, dim=0)
    labels = torch.tensor(eval_vectors, dtype=torch.float32)

    # Align by length (we already aligned earlier when possible; this is a safe fallback).
    n = min(preds.size(0), labels.size(0))
    preds = preds[:n]
    labels = labels[:n]

    pred_norm = F.normalize(preds, p=2, dim=-1)
    lab_norm = F.normalize(labels, p=2, dim=-1)
    cos = (pred_norm * lab_norm).sum(dim=-1).mean().item()

    print(f"Test concept cosine similarity vs llama.cpp vectors (raw): {cos:.4f}")
    wandb.log({"test_concept_cosine_raw_llamacpp": float(cos)})
    return {"test_concept_cosine_raw_llamacpp": float(cos)}


def build_intervened_concepts_from_similarity(
    concepts: torch.Tensor,
    batch_sim: torch.Tensor,
    intervention_value: float,
    keep_other_concepts: bool,
    use_topk: bool,
    topk_k: int,
    log_wandb: bool = False,
    wandb_prefix: str = "train",
) -> torch.Tensor:
    """Construct an intervention concept tensor for `cbl.intervene`.

    Defaults:
      - top-1 concept per example (argmax over batch_sim)
      - set that concept to `intervention_value` for all time steps
      - set all other concepts to 0

    Optional:
      - keep_other_concepts=True: start from `concepts` and only overwrite targeted dims
      - use_topk=True: stochastic top-K cutoff selection per example
    """
    import wandb

    if concepts.dim() != 3:
        raise ValueError(f"Expected concepts to have shape (B, T, C); got {tuple(concepts.shape)}")
    if batch_sim.dim() != 2:
        raise ValueError(f"Expected batch_sim to have shape (B, C); got {tuple(batch_sim.shape)}")
    if concepts.size(0) != batch_sim.size(0) or concepts.size(-1) != batch_sim.size(-1):
        raise ValueError(
            f"Shape mismatch: concepts {tuple(concepts.shape)} vs batch_sim {tuple(batch_sim.shape)}"
        )

    if keep_other_concepts:
        intervened = concepts.detach().clone()
    else:
        intervened = torch.zeros_like(concepts)

    value = float(intervention_value)
    B = concepts.size(0)
    C = concepts.size(-1)

    if not use_topk:
        indices = torch.argmax(batch_sim, dim=-1)  # (B,)
        for b in range(B):
            intervened[b, :, int(indices[b].item())] = value
        return intervened

    k = int(topk_k)
    if k <= 0:
        k = 1
    k = min(k, C)

    sampled_ranks = []
    sampled_probs = []
    for b in range(B):
        topk = torch.topk(batch_sim[b], k=k, dim=-1)
        topk_scores = topk.values  # (k,)
        topk_indices = topk.indices  # (k,)

        probs = F.softmax(topk_scores, dim=-1)
        sampled_rank = torch.multinomial(probs, num_samples=1).item()  # int in [0, k-1]
        cutoff = int(sampled_rank) + 1
        selected = topk_indices[:cutoff].tolist()

        sampled_ranks.append(int(sampled_rank))
        sampled_probs.append(float(probs[int(sampled_rank)].item()))

        for idx in selected:
            intervened[b, :, int(idx)] = value

    if log_wandb and len(sampled_ranks) > 0 and wandb.run is not None:
        mean_rank = float(sum(sampled_ranks) / len(sampled_ranks))
        mean_prob = float(sum(sampled_probs) / len(sampled_probs))
        wandb.log(
            {
                f"{wandb_prefix}_intervention_topk_sampled_rank_b0": sampled_ranks[0],
                f"{wandb_prefix}_intervention_topk_sampled_prob_b0": sampled_probs[0],
                f"{wandb_prefix}_intervention_topk_sampled_rank_mean": mean_rank,
                f"{wandb_prefix}_intervention_topk_sampled_prob_mean": mean_prob,
            },
            commit=False,
        )

    return intervened
