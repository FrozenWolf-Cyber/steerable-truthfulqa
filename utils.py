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
