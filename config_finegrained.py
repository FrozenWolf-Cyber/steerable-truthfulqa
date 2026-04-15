"""Finegrained CBM config for training scripts in this folder.

This file is intentionally minimal and self-contained so that running
`truthful_qa/train_combined_finegrained.py` does not accidentally import
configs from other subprojects (e.g., CB-LLMs).
"""

# FEVER label convention (HF "fever" v1.0):
# 0 = SUPPORTS, 1 = REFUTES, 2 = NOT ENOUGH INFO

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

# Which concepts are considered applicable/eligible per FEVER class.
# These are used for masking (zero-out) during training losses.
FEVER_LABEL_TO_CONCEPT_INDICES = {
    0: [
        0,  # directly supported
        1,  # explicit attribution
        2,  # asserting certainty (can happen for supports too)
    ],
    1: [
        3,  # popular myth
        4,  # contradicts consensus
        5,  # anecdotal
        2,  # asserting certainty
    ],
    2: [
        5,  # anecdotal
        6,  # lacking evidential basis
        7,  # genuine uncertainty / hedging
        2,  # asserting certainty
    ],
}


# Training script expects these dicts.
example_name = {
    "fever": "claim",
}

class_num = {
    "fever": 3,
}

# Keep short by default; FEVER is large.
epoch = {
    "fever": 1,
}

concept_set = {
    "fever": FEVER_CONCEPTS_ALL,
}
