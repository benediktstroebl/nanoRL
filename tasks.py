"""
Tasks for nanoRL.

A task is two functions: a `dataset(split)` that returns a list of
{messages, answer} dicts and a `reward(response_text, answer)` that
returns a scalar in [0, 1]. Add a new task by writing both and
registering them in TASKS at the bottom.

The default task is GSM8K (grade-school math). The model is asked to
reason step by step and put the final answer in \\boxed{}; the reward
is 1.0 iff the boxed value matches the ground truth numerically.
"""
import re
from datasets import load_dataset

# -----------------------------------------------------------------------------
# Reward functions

# Be liberal: accept \boxed{42}, \boxed{ 42 }, \boxed{42.0}, \boxed{4,200}, etc.
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

def _extract_boxed(text):
    """Return the *last* \\boxed{...} content, or None."""
    m = BOXED_RE.findall(text)
    return m[-1].strip() if m else None

def gsm8k_reward(response, answer):
    pred = _extract_boxed(response)
    if pred is None:
        return 0.0
    pred = pred.replace(",", "").replace("$", "").strip()
    try:
        return float(abs(float(pred) - float(answer)) < 1e-4)
    except ValueError:
        return float(pred == answer)

# -----------------------------------------------------------------------------
# Dataset loaders

GSM8K_SYSTEM = (
    "You are a careful math assistant. Reason step by step. "
    "Put your final numeric answer in \\boxed{}."
)

def gsm8k_dataset(split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for ex in ds:
        # GSM8K answers look like "<reasoning>\n#### 42"
        ans = ex["answer"].split("####")[-1].strip().replace(",", "")
        out.append({
            "messages": [
                {"role": "system", "content": GSM8K_SYSTEM},
                {"role": "user",   "content": ex["question"]},
            ],
            "answer": ans,
        })
    return out

# -----------------------------------------------------------------------------
# Registry

TASKS = {
    "gsm8k": (gsm8k_dataset, gsm8k_reward),
}

def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task '{name}', valid: {list(TASKS)}")
    return TASKS[name]
