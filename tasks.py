"""Tasks for nanoRL.

A task = `dataset(split) -> [{messages, answer}, ...]` plus
`reward(text, answer) -> float in [0,1]`, both registered in TASKS below.
Default is GSM8K, scored by extracting the last \\boxed{...} value.
"""
import re
from datasets import load_dataset

# Match the *last* \boxed{...}; accept commas / $ in the value.
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

def gsm8k_reward(response, answer):
    m = BOXED_RE.findall(response)
    if not m:
        return 0.0
    pred = m[-1].replace(",", "").replace("$", "").strip()
    try:
        return float(abs(float(pred) - float(answer)) < 1e-4)
    except ValueError:
        return float(pred == answer)

GSM8K_SYSTEM = ("You are a careful math assistant. Reason step by step. "
                "Put your final numeric answer in \\boxed{}.")

def gsm8k_dataset(split="train"):
    return [
        dict(
            messages=[{"role": "system", "content": GSM8K_SYSTEM},
                     {"role": "user",   "content": ex["question"]}],
            answer=ex["answer"].split("####")[-1].strip().replace(",", ""),
        )
        for ex in load_dataset("openai/gsm8k", "main", split=split)
    ]

TASKS = {"gsm8k": (gsm8k_dataset, gsm8k_reward)}

def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task '{name}', valid: {list(TASKS)}")
    return TASKS[name]
