import json
import re
import pickle
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from llama_cpp import Llama

# =========================
# 1. Llama.cpp setup (A100)
# =========================
MODEL_REPO_ID = "unsloth/Qwen3.5-27B-GGUF"
MODEL_FILENAME = "Qwen3.5-27B-Q8_0.gguf"
CHECKPOINT_PATH = "fever_progress_llamacpp.pkl"
MAX_SAMPLES = 20000

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
# IMPORTANT:
# - Prompting uses a label-specific option set (conservative).
# - Saved vectors are always over the *global* union concept set (fixed width).

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
# 4. Prompt builder (STRICT)
# =========================
def build_prompt(claim, concepts):
    return f"""
You are a strict multi-label classifier.

TASK:
Given a claim, select ALL applicable labels.

RULES:
- Only choose from the provided options
- Output ONLY a comma-separated list of labels
- Do NOT explain anything
- Do NOT add extra text
- Multiple labels are allowed
- Be precise and conservative


CLAIM:
{claim}

OPTIONS:
{", ".join(concepts)}

ANSWER:
""".strip()


# =========================
# 5. Llama.cpp call
# =========================
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def call_model(prompt):
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=512,
    )
    text = response["choices"][0]["message"]["content"].strip()
    text = _THINK_RE.sub("", text).strip()
    if not text:
        raise RuntimeError("Empty response text.")
    return text


# =========================
# 6. Safe parser
# =========================
def parse_output(output, concepts):
    output_lower = output.lower()
    labels = []

    for c in concepts:
        if c.lower() in output_lower:
            labels.append(c)

    if len(labels) == 0:
        labels = [concepts[0]]

    return labels


# =========================
# 7. Convert to vector
# =========================
def to_vector(labels, concepts):
    """Convert selected labels to a fixed-width vector over FEVER_CONCEPTS_ALL."""
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


# =========================
# 8. Load FEVER + resume checkpoint
# =========================
dataset = load_dataset("fever", "v1.0", split=f"train[:{MAX_SAMPLES}]")

all_vectors = []
all_claims = []
all_labels = []
all_outputs = []
all_parse_errors = []

try:
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)
    all_vectors = ckpt.get("all_vectors", [])
    all_claims = ckpt.get("all_claims", [])
    all_labels = ckpt.get("all_labels", [])
    all_outputs = ckpt.get("all_outputs", [])
    all_parse_errors = ckpt.get("all_parse_errors", [])
    print(f"Resuming from checkpoint: {len(all_claims)} examples already processed")
except FileNotFoundError:
    print("No checkpoint found. Starting fresh.")

start_idx = len(all_claims)
logged_first_sample = False


# =========================
# 9. Annotation loop (checkpoint every response)
# =========================
for ex in tqdm(dataset.select(range(start_idx, len(dataset))), initial=start_idx, total=len(dataset)):
    claim = ex["claim"]
    label = normalize_label(ex["label"])
    concepts = get_concepts(label)
    prompt = build_prompt(claim, concepts)
    parse_error = ""

    try:
        output = call_model(prompt)
    except Exception as e:
        print(f"[model-error] {type(e).__name__}: {str(e)[:200]}")
        output = ""

    try:
        labels = parse_output(output, concepts)
        vec = to_vector(labels, concepts)
    except Exception as e:
        parse_error = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[parse-error] {parse_error}")
        labels = [concepts[0]]
        vec = to_vector(labels, concepts)

    if not logged_first_sample:
        print("\n[first-sample-prompt]")
        print(prompt)
        print("\n[first-sample-output]")
        print(output)
        logged_first_sample = True

    all_vectors.append(vec)
    all_claims.append(claim)
    all_labels.append(label)
    all_outputs.append(output)
    all_parse_errors.append(parse_error)

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(
            {
                "all_vectors": all_vectors,
                "all_claims": all_claims,
                "all_labels": all_labels,
                "all_outputs": all_outputs,
                "all_parse_errors": all_parse_errors,
            },
            f,
        )


# =========================
# 10. Save outputs
# =========================
np.save("fever_concept_vectors_llamacpp.npy", np.stack(all_vectors, axis=0))
np.save("fever_claims_llamacpp.npy", np.array(all_claims))

with open("fever_raw_outputs_llamacpp.json", "w") as f:
    json.dump(
        {
            "claims": all_claims,
            "fever_labels": all_labels,
            "outputs": all_outputs,
            "parse_errors": all_parse_errors,
        },
        f,
    )

print("DONE")
print(
    "Saved: fever_concept_vectors_llamacpp.npy, fever_claims_llamacpp.npy, "
    "fever_raw_outputs_llamacpp.json, fever_progress_llamacpp.pkl"
)
