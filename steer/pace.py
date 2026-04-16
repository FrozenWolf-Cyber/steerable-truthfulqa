"""
PaCE — Parsimonious Concept Engineering (NeurIPS 2024).

Adapted from ODESteer's pace.py with:
  - tqdm progress bar during initial concept embedding generation
  - No Hydra dependency — plain Python config dict

How this file works
-------------------
1. **ConceptDictionary** — Reads PaCE-1M-style `concept_index.txt` and per-concept
   `*.txt` files of short context strings that *illustrate* each concept.

2. **ConceptPartitioner** — Splits concept indices into "benign" vs "undesirable"
   (e.g. for TruthfulQA, keywords suggestive of falsehood → undesirable). Only
   undesirable directions are removed at inference.

3. **ActivationConceptEncoder** — For each concept, runs the frozen LM on its
   context strings, takes the last-token hidden state at `layer_idx`, averages
   across contexts, and caches a single **CPU** vector per concept (disk cache).

4. **PaCESteerer** — Registers a **forward hook** on that transformer block. On
   each forward, for every token position it expresses the hidden state as a
   linear mix of concept vectors (`decompose_sparse`: SVD-reduced least squares),
   rebuilds only the **undesirable** part of that mix, and **subtracts** it
   (scaled by `alpha`) from the activation — suppressing those directions in
   residual space.

Dictionary math stays on **CPU** (NumPy `lstsq`); activations are moved CPU for
the solve and moved back to the model device/dtype (see `_steer_activation`).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sparse coding helpers
# ---------------------------------------------------------------------------

def _svd_embed(X: torch.Tensor, q: int = 500) -> torch.Tensor:
    q = min(q, min(X.shape) - 1)
    U, S, V = torch.svd_lowrank(X, q=q)
    return (V * S).T


def decompose_sparse(
    target: torch.Tensor,
    dictionary: List[torch.Tensor],
    normalize: bool = True,
    use_gpu: bool = False,
) -> torch.Tensor:
    if use_gpu:
        data = torch.stack([target.view(-1)] + [a.view(-1) for a in dictionary], dim=0).float()
        embedded = _svd_embed(data.T).T

        y = embedded[0:1].clone()
        D = embedded[1:].clone()

        if normalize:
            eps = 1e-12
            norm_y = torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(eps)
            norm_D = torch.linalg.norm(D, dim=1, keepdim=True).clamp_min(eps)
            y = y / norm_y
            D = D / norm_D
        else:
            norm_y = torch.ones((1, 1), device=embedded.device, dtype=embedded.dtype)
            norm_D = torch.ones((D.shape[0], 1), device=embedded.device, dtype=embedded.dtype)

        c = torch.linalg.lstsq(D.T, y.T).solution.squeeze(-1)

        if normalize:
            c = c / norm_D.squeeze(-1) * norm_y.squeeze()

        return c.to(dtype=torch.float32)

    data = torch.stack([target.view(-1)] + [a.view(-1) for a in dictionary], dim=0)
    embedded = _svd_embed(data.T).T

    data_np = embedded.numpy().astype(np.float64)
    y = data_np[0:1].copy()
    D = data_np[1:].copy()

    if normalize:
        norm_y = np.linalg.norm(y) + 1e-12
        norm_D = np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
        y = y / norm_y
        D = D / norm_D
    else:
        norm_y, norm_D = 1.0, np.ones((D.shape[0], 1))

    c, *_ = np.linalg.lstsq(D.T, y.T, rcond=None)
    c = c.T[0]

    if normalize:
        c = c / norm_D.squeeze() * norm_y

    return torch.tensor(c, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Concept dictionary loader
# ---------------------------------------------------------------------------

class ConceptDictionary:
    def __init__(
        self,
        index_path: str,
        representation_path: str,
        max_concepts: int = 5000,
    ):
        self.representation_path = Path(representation_path)
        with open(index_path, "r") as f:
            all_concepts: List[str] = ast.literal_eval(f.read())

        if max_concepts > 0:
            all_concepts = all_concepts[:max_concepts]

        self.concepts: List[str] = []
        self.representations: List[List[str]] = []
        skipped_empty = 0

        for concept in all_concepts:
            rep_file = self.representation_path / f"{concept}.txt"
            if rep_file.exists():
                with open(rep_file, "r") as rf:
                    rep = ast.literal_eval(rf.read())
                valid_rep = [r for r in rep if isinstance(r, str) and r.strip()]
                if not valid_rep:
                    skipped_empty += 1
                    continue
                self.concepts.append(concept)
                self.representations.append(valid_rep)

        logger.info(
            "ConceptDictionary: loaded %d / %d concepts (skipped_empty=%d)",
            len(self.concepts),
            len(all_concepts),
            skipped_empty,
        )

    def __len__(self) -> int:
        return len(self.concepts)


# ---------------------------------------------------------------------------
# Concept partitioner
# ---------------------------------------------------------------------------

class ConceptPartitioner:
    _HALLUCINATION_KEYWORDS = {
        "false", "myth", "fake", "fiction", "hoax", "rumor", "rumour",
        "incorrect", "misinformation", "disinformation", "fabricat",
        "conspiracy", "pseudoscience", "debunked", "wrong", "error",
        "lie", "lies", "lying", "untrue", "misleading", "misconception",
        "superstition", "legend", "folklore",
    }

    def __init__(self, mode: str = "heuristic", partition_file: Optional[str] = None):
        self.mode = mode
        self._cache: dict[str, bool] = {}

        if mode == "file":
            if partition_file is None or not Path(partition_file).exists():
                raise FileNotFoundError(f"partition_file={partition_file!r} not found.")
            import json
            with open(partition_file) as f:
                self._cache = json.load(f)

    def is_benign(self, concept: str) -> bool:
        if concept in self._cache:
            return self._cache[concept]
        if self.mode == "file":
            return True
        concept_lower = concept.lower()
        for kw in self._HALLUCINATION_KEYWORDS:
            if kw in concept_lower:
                self._cache[concept] = False
                return False
        self._cache[concept] = True
        return True

    def partition(self, concepts: List[str]) -> Tuple[List[int], List[int]]:
        benign, undesirable = [], []
        for i, c in enumerate(concepts):
            (benign if self.is_benign(c) else undesirable).append(i)
        return benign, undesirable


# ---------------------------------------------------------------------------
# Activation-space concept encoder — with tqdm progress
# ---------------------------------------------------------------------------

class ActivationConceptEncoder:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer_idx: int,
        cache_path: str,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    def _encode_contexts(self, contexts: List[str]) -> torch.Tensor:
        contexts = [c for c in contexts if isinstance(c, str) and c.strip()]
        if not contexts:
            raise ValueError("PaCE concept has no valid contexts after filtering.")

        vectors = []
        for i in range(0, len(contexts), self.batch_size):
            batch = contexts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[self.layer_idx + 1]
            seq_lens = (enc["attention_mask"].sum(dim=1) - 1).clamp_min(0)
            for b_idx, s_len in enumerate(seq_lens):
                vectors.append(hs[b_idx, s_len].cpu())
        if not vectors:
            raise RuntimeError("PaCE failed to encode any vectors from non-empty contexts.")
        return torch.stack(vectors).mean(dim=0)

    def get_concept_vector(self, concept: str, contexts: List[str]) -> torch.Tensor:
        cache_file = self.cache_path / f"{concept}.pt"
        if cache_file.exists():
            return torch.load(cache_file, map_location="cpu")
        vec = self._encode_contexts(contexts)
        torch.save(vec, cache_file)
        return vec

    def encode_dictionary(
        self,
        concept_dict: ConceptDictionary,
        max_concepts: int = -1,
    ) -> List[torch.Tensor]:
        concepts = concept_dict.concepts
        if max_concepts > 0:
            concepts = concepts[:max_concepts]

        vecs = []
        pairs = list(zip(concept_dict.concepts, concept_dict.representations))
        if max_concepts > 0:
            pairs = pairs[:max_concepts]

        for c, reps in tqdm(pairs, desc="Encoding PaCE concepts", unit="concept"):
            vecs.append(self.get_concept_vector(c, reps))
        return vecs


# ---------------------------------------------------------------------------
# PaCE Steerer
# ---------------------------------------------------------------------------

class PaCESteerer:
    """
    Drop-in PaCE steerer. No Hydra — pass a plain dict as `cfg`.

    Expected cfg keys:
        index_path, representation_path, max_concepts (int),
        partition_mode (str), partition_file (str|None),
        vector_cache_path (str), encode_batch_size (int),
        alpha (float), layer_idx (int)
    """

    def __init__(self, cfg: dict, model: nn.Module, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx: int = cfg["layer_idx"]
        self._hook_handle = None
        self.pace_gpu: bool = bool(cfg.get("pace_gpu", False))

        concept_dict = ConceptDictionary(
            index_path=cfg["index_path"],
            representation_path=cfg["representation_path"],
            max_concepts=cfg.get("max_concepts", 5000),
        )

        partitioner = ConceptPartitioner(
            mode=cfg.get("partition_mode", "heuristic"),
            partition_file=cfg.get("partition_file"),
        )
        self.benign_idx, self.undesirable_idx = partitioner.partition(concept_dict.concepts)
        logger.info("PaCESteerer: %d benign, %d undesirable", len(self.benign_idx), len(self.undesirable_idx))

        encoder = ActivationConceptEncoder(
            model=model, tokenizer=tokenizer, layer_idx=self.layer_idx,
            cache_path=cfg.get("vector_cache_path", f"./pace_cache/layer{self.layer_idx}"),
            batch_size=cfg.get("encode_batch_size", 8),
        )
        self.concept_vectors: List[torch.Tensor] = encoder.encode_dictionary(
            concept_dict, max_concepts=cfg.get("max_concepts", 5000),
        )

        self.alpha: float = cfg.get("alpha", 1.0)
        self._concept_vectors_gpu: Optional[List[torch.Tensor]] = None
        if self.pace_gpu:
            self._concept_vectors_gpu = []

    def fit(self, *args, **kwargs):
        return self

    def _steer_activation(self, activation: torch.Tensor) -> torch.Tensor:
        if not self.concept_vectors:
            return activation

        dev, dtype = activation.device, activation.dtype
        if self.pace_gpu:
            if self._concept_vectors_gpu is None or len(self._concept_vectors_gpu) != len(self.concept_vectors):
                self._concept_vectors_gpu = []
            if len(self._concept_vectors_gpu) == 0 or self._concept_vectors_gpu[0].device != dev:
                self._concept_vectors_gpu = [v.to(device=dev, dtype=torch.float32) for v in self.concept_vectors]

            act_gpu = activation.detach().float()
            coeffs = decompose_sparse(
                target=act_gpu,
                dictionary=self._concept_vectors_gpu,
                normalize=True,
                use_gpu=True,
            )
            correction = torch.zeros_like(act_gpu)
            for idx in self.undesirable_idx:
                if idx < len(coeffs):
                    correction += coeffs[idx] * self._concept_vectors_gpu[idx]
            steered = act_gpu - self.alpha * correction
            return steered.to(dtype=dtype)

        # CPU reference path.
        act_cpu = activation.detach().float().cpu()
        coeffs = decompose_sparse(
            target=act_cpu, dictionary=self.concept_vectors, normalize=True,
        )
        correction = torch.zeros_like(act_cpu)
        for idx in self.undesirable_idx:
            if idx < len(coeffs):
                correction += coeffs[idx] * self.concept_vectors[idx]
        steered = act_cpu - self.alpha * correction
        return steered.to(device=dev, dtype=dtype)

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        B, T, D = hidden.shape
        steered = hidden.clone()
        for b in range(B):
            for t in range(T):
                steered[b, t] = self._steer_activation(hidden[b, t])
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    def register_hook(self):
        layer = self._get_layer(self.layer_idx)
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _get_layer(self, idx: int) -> nn.Module:
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        for attr in ("model", "transformer", "base_model"):
            if hasattr(model, attr):
                model = getattr(model, attr)
                break
        for layers_attr in ("layers", "h", "blocks"):
            if hasattr(model, layers_attr):
                return getattr(model, layers_attr)[idx]
        raise AttributeError(f"Cannot locate transformer layers in {type(model)}")

    def __enter__(self):
        self.register_hook()
        return self

    def __exit__(self, *args):
        self.remove_hook()
