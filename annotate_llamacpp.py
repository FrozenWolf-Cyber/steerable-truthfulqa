import argparse
import json
import os
import re
import pickle
import urllib.request
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama


def parse_args():
    p = argparse.ArgumentParser(description="FEVER concept annotation with llama.cpp")
    p.add_argument(
        "--restart",
        action="store_true",
        help="Delete existing checkpoint pickle and start from scratch (does not delete fever_train.jsonl).",
    )
    return p.parse_args()


ARGS = parse_args()

# =========================
# 1. Llama.cpp setup (A100)
# =========================
MODEL_REPO_ID = "unsloth/Qwen3.5-27B-GGUF"
MODEL_FILENAME = "Qwen3.5-27B-Q8_0.gguf"
CHECKPOINT_PATH = "fever_progress_llamacpp.pkl"
MAX_SAMPLES = 20000

if ARGS.restart and os.path.isfile(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print(f"Removed checkpoint (--restart): {CHECKPOINT_PATH}")

llm = Llama.from_pretrained(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILENAME,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)


# =========================
# 2. FEVER concept sets
# =========================
FEVER_CONCEPTS_ALL = [
    "claim directly supported by verifiable documented evidence",
    "claim with explicit attribution to a named source or study",
    "claim asserting certainty on a contested or ambiguous question",
    "claim reflecting a widespread popular myth or misconception",
    "claim that contradicts established scientific or historical consensus",
    "claim generalized from anecdotal or single-case evidence",
    "claim presented as fact but lacking sufficient evidential basis",
    "claim under genuine empirical uncertainty with appropriate hedging",
]

SUPPORTS_CONCEPTS = [
    FEVER_CONCEPTS_ALL[0],
    FEVER_CONCEPTS_ALL[1],
    FEVER_CONCEPTS_ALL[2],
]

REFUTES_CONCEPTS = [
    FEVER_CONCEPTS_ALL[3],
    FEVER_CONCEPTS_ALL[4],
    FEVER_CONCEPTS_ALL[5],
    FEVER_CONCEPTS_ALL[2],
]

NEI_CONCEPTS = [
    FEVER_CONCEPTS_ALL[5],
    FEVER_CONCEPTS_ALL[6],
    FEVER_CONCEPTS_ALL[7],
    FEVER_CONCEPTS_ALL[2],
]


# =========================
# 3. FEVER concept routing
# =========================
def normalize_label(label):
    if isinstance(label, (int, np.integer)):
        if label == 0:
            return "SUPPORTS"
        if label == 1:
            return "REFUTES"
        return "NOT ENOUGH INFO"

    label_str = str(label).strip().upper()
    if label_str in ["SUPPORTS", "REFUTES"]:
        return label_str
    return "NOT ENOUGH INFO"


def get_concepts(label):
    normalized = normalize_label(label)
    if normalized == "SUPPORTS":
        return SUPPORTS_CONCEPTS
    if normalized == "REFUTES":
        return REFUTES_CONCEPTS
    return NEI_CONCEPTS


# =========================
# 4. Prompt builder (strict, machine-parseable last line)
# =========================
def build_prompt(claim, concepts):
    opts_block = "\n".join(f"- {c}" for c in concepts)
    return f"""You are a strict multi-label classifier for fact-checking concepts.

TASK:
Given the CLAIM below, select ALL applicable labels from OPTIONS.

OUTPUT RULES (mandatory):
1. You may think step by step in earlier lines if needed.
2. The VERY LAST non-empty line of your entire reply MUST be your only machine-readable answer.
3. That final line MUST contain NOTHING except labels taken verbatim from OPTIONS (copy the full text exactly as written under OPTIONS).
4. Separate multiple labels with a comma followed by a space: ", "
5. Do NOT number labels, do NOT use "Option 1/2", do NOT add quotes, bullets, or extra words on that final line.
6. Regex target: ^(<exact option text>(, <exact option text>)*)$

OPTIONS:
{opts_block}

CLAIM:
{claim}

End your reply so the last line is only comma-separated labels copied from OPTIONS.""".strip()


# =========================
# 5. Llama.cpp call
# =========================
_THINK_RE = re.compile(r"<redacted_thinking>.*?</redacted_thinking>\s*", re.DOTALL)


def call_model(prompt):
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=512,
    )
    raw = response["choices"][0]["message"]["content"]
    if raw is None:
        raw = ""
    raw = raw if isinstance(raw, str) else str(raw)
    return raw


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    return text


def last_non_empty_line(text: str) -> str:
    body = strip_thinking(text)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# =========================
# 6. Parser (last line only; regex-friendly)
# =========================
def parse_output(output, concepts):
    last = last_non_empty_line(output)
    if not last:
        return [concepts[0]]

    parts = [p.strip() for p in last.split(",") if p.strip()]
    labels = []
    concept_set = {c: c for c in concepts}

    for p in parts:
        if p in concept_set:
            labels.append(p)
            continue
        matched = None
        for c in concepts:
            if c.lower() == p.lower():
                matched = c
                break
        if matched:
            if matched not in labels:
                labels.append(matched)
            continue
        for c in concepts:
            if p.lower() in c.lower() or c.lower() in p.lower():
                if c not in labels:
                    labels.append(c)
                break

    if len(labels) == 0:
        labels = [concepts[0]]

    return labels


# =========================
# 7. Convert to vector
# =========================
def to_vector(labels, concepts):
    vec = np.zeros(len(FEVER_CONCEPTS_ALL), dtype=np.float32)
    selected = [c for c in labels if c in concepts]
    if len(selected) == 0:
        selected = [concepts[0]]

    for c in selected:
        try:
            idx = FEVER_CONCEPTS_ALL.index(c)
        except ValueError:
            continue
        vec[idx] = 1.0

    s = float(vec.sum())
    if s > 0:
        vec = vec / s
    return vec


def log_sample(claim, prompt, raw_output, idx: int):
    print("\n============")
    print(f"sample_index={idx}")
    print("[claim]")
    print(claim)
    print("------------")
    print("[prompt]")
    print(prompt)
    print("------------")
    print("[raw_model_output]")
    print(raw_output)
    print("============\n", flush=True)


# =========================
# 8. Load FEVER + resume checkpoint
# =========================
FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"
FEVER_LOCAL_PATH = "fever_train.jsonl"

if not os.path.exists(FEVER_LOCAL_PATH):
    print(f"Downloading FEVER training set from {FEVER_TRAIN_URL} ...")
    urllib.request.urlretrieve(FEVER_TRAIN_URL, FEVER_LOCAL_PATH)
    print("Download complete.")

with open(FEVER_LOCAL_PATH) as f:
    dataset = [json.loads(line) for line in f]
dataset = dataset[:MAX_SAMPLES]

all_vectors = []
all_claims = []
all_labels = []
all_prompts = []
all_outputs_raw = []
all_parse_errors = []

try:
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    all_vectors = ckpt.get("all_vectors", [])
    all_claims = ckpt.get("all_claims", [])
    all_labels = ckpt.get("all_labels", [])
    all_prompts = ckpt.get("all_prompts", [])
    all_outputs_raw = ckpt.get("all_outputs_raw", ckpt.get("all_outputs", []))
    all_parse_errors = ckpt.get("all_parse_errors", [])
    if len(all_prompts) < len(all_claims):
        all_prompts = (all_prompts + [""] * len(all_claims))[: len(all_claims)]
    print(f"Resuming from checkpoint: {len(all_claims)} examples already processed")
except FileNotFoundError:
    print("No checkpoint found. Starting fresh.")

start_idx = len(all_claims)


# =========================
# 9. Annotation loop (checkpoint every response)
# =========================
for i, ex in enumerate(tqdm(dataset[start_idx:], initial=start_idx, total=len(dataset))):
    global_idx = start_idx + i
    claim = ex["claim"]
    label = normalize_label(ex["label"])
    concepts = get_concepts(label)
    prompt = build_prompt(claim, concepts)
    parse_error = ""

    try:
        raw_output = call_model(prompt)
    except Exception as e:
        print(f"[model-error] {type(e).__name__}: {str(e)[:200]}")
        raw_output = ""

    log_sample(claim, prompt, raw_output, global_idx)

    try:
        labels = parse_output(raw_output, concepts)
        vec = to_vector(labels, concepts)
    except Exception as e:
        parse_error = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[parse-error] {parse_error}")
        labels = [concepts[0]]
        vec = to_vector(labels, concepts)

    all_vectors.append(vec)
    all_claims.append(claim)
    all_labels.append(label)
    all_prompts.append(prompt)
    all_outputs_raw.append(raw_output)
    all_parse_errors.append(parse_error)

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(
            {
                "all_vectors": all_vectors,
                "all_claims": all_claims,
                "all_labels": all_labels,
                "all_prompts": all_prompts,
                "all_outputs_raw": all_outputs_raw,
                "all_parse_errors": all_parse_errors,
            },
            f,
        )


# =========================
# 10. Save outputs
# =========================
if len(all_vectors):
    np.save("fever_concept_vectors_llamacpp.npy", np.stack(all_vectors, axis=0))
    np.save("fever_claims_llamacpp.npy", np.array(all_claims))
else:
    print("No vectors to save (empty run).")

payload = {
    "claims": all_claims,
    "fever_labels": all_labels,
    "prompts": all_prompts,
    "outputs_raw": all_outputs_raw,
    "parse_errors": all_parse_errors,
}

with open("fever_raw_outputs_llamacpp.json", "w") as f:
    json.dump(payload, f, ensure_ascii=False)

print("DONE")
print(
    "Saved: fever_concept_vectors_llamacpp.npy, fever_claims_llamacpp.npy, "
    "fever_raw_outputs_llamacpp.json, fever_progress_llamacpp.pkl"
)
