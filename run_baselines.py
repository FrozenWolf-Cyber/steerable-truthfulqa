"""
Run probing-based baseline steering methods on TruthfulQA.

Replicates the exact same pipeline as ODESteer's truthfulqa_generate.py:
  - 2-fold cross-validation (train on split 0, test on split 1, then swap)
  - Fit steering model on train-split activations
  - Generate answers to test-split questions with steering applied
  - Save outputs in standard jsonl format for evaluate.py

Usage:
  # Run a single method:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer CAA --T 1.0

  # Run all baselines:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer all

  # Run all baselines + evaluate:
  python run_baselines.py -m Llama3.1-8B-Base -l 13 --steer all --evaluate
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import GenerationConfig

from config import (
    TRUTHFULQA_SYSTEM_PROMPT, DEFAULT_SEED, STEER_METHODS,
)
from data_prep import load_questions, load_activations
from lm import HuggingFaceLM, batch_chat

RESULTS_DIR = Path(__file__).parent / "results" / "truthfulqa"


def run_single_method(
    model_name: str,
    layer_idx: int,
    steer_name: str,
    T: float,
    batch_size: int,
    seed: int,
    pace_cfg: dict | None = None,
):
    output_dir = RESULTS_DIR / "raw_outputs" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    steer_label = f"{steer_name}-T{T}" if steer_name != "NoSteer" else "NoSteer"
    filename = f"{model_name}-l{layer_idx}-{steer_label}-TruthfulQA-seed{seed}.jsonl"

    if (output_dir / filename).exists():
        print(f"Output exists: {filename} — skipping")
        return output_dir / filename

    all_prompts, all_outputs = [], []
    print(f"\n{'='*80}")
    print(f"Running {model_name} / {steer_label} on TruthfulQA (2-fold CV)")
    print(f"{'='*80}")

    for test_split in [0, 1]:
        train_split = 1 - test_split
        print(f"\nFold {test_split + 1}: train={train_split}, test={test_split}")

        gen_config = GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=0.7,
            top_p=0.9, repetition_penalty=1.1, seed=seed,
        )

        model = HuggingFaceLM(
            model_name, steer_name,
            default_generation_config=gen_config,
            steer_layer_idx=layer_idx,
            device="auto", dtype=torch.float32,
            pace_cfg=pace_cfg if steer_name == "PaCE" else None,
        )

        if steer_name not in ("NoSteer", "PaCE"):
            pos_train, neg_train = load_activations(model_name, layer_idx, train_split)
            model.fit_steer_model(pos_train, neg_train)

        questions = load_questions(test_split)
        messages = [[
            {"role": "system", "content": TRUTHFULQA_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ] for q in questions]

        print(f"Generating {len(questions)} responses (T={T}) ...")
        outputs = batch_chat(model, messages, T=T, batch_size=batch_size)

        all_prompts.extend(questions)
        all_outputs.extend(outputs)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(output_dir / filename, "w") as f:
        for prompt, output in zip(all_prompts, all_outputs):
            f.write(json.dumps({
                "prompt": prompt,
                "output": output,
                "generator": f"{model_name}-{steer_label}",
                "dataset": "TruthfulQA",
                "T": T,
            }) + "\n")

    print(f"Saved {len(all_outputs)} outputs to {filename}")
    return output_dir / filename


def build_pace_cfg(layer_idx: int, args) -> dict:
    return {
        "index_path": args.pace_index_path,
        "representation_path": args.pace_representation_path,
        "max_concepts": args.pace_max_concepts,
        "partition_mode": "heuristic",
        "partition_file": None,
        "vector_cache_path": f"./pace_cache/layer{layer_idx}",
        "encode_batch_size": 8,
        "alpha": args.pace_alpha,
        "layer_idx": layer_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline steering on TruthfulQA")
    parser.add_argument("-m", "--model", type=str, default="Llama3.1-8B-Base")
    parser.add_argument("-l", "--layer_idx", type=int, default=13)
    parser.add_argument("-b", "--batch_size", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steer", type=str, default="all",
                        help="Steering method name or 'all'. Choices: " + ", ".join(STEER_METHODS))
    parser.add_argument("--T", type=float, default=1.0, help="Steering strength T")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after generation")

    parser.add_argument("--pace_index_path", type=str, default="./pace_data/concept_index.txt")
    parser.add_argument("--pace_representation_path", type=str, default="./pace_data/concept/")
    parser.add_argument("--pace_max_concepts", type=int, default=5000)
    parser.add_argument("--pace_alpha", type=float, default=1.0)
    args = parser.parse_args()

    methods = STEER_METHODS if args.steer == "all" else [args.steer]

    for method in methods:
        pace_cfg = build_pace_cfg(args.layer_idx, args) if method == "PaCE" else None
        run_single_method(
            model_name=args.model,
            layer_idx=args.layer_idx,
            steer_name=method,
            T=args.T,
            batch_size=args.batch_size,
            seed=args.seed,
            pace_cfg=pace_cfg,
        )

    if args.evaluate:
        from evaluate import evaluate_outputs
        raw_dir = RESULTS_DIR / "raw_outputs"
        eval_path = (
            RESULTS_DIR / "eval_results" / "stat_results"
            / f"{args.model}-l{args.layer_idx}-TruthfulQA-seed{args.seed}.csv"
        )
        evaluate_outputs(raw_dir, eval_path, args.model, args.layer_idx, args.seed, args.batch_size, display=True)


if __name__ == "__main__":
    main()
