"""Tasks for nanoRL.

A task = `dataset(split) -> [{question, answer}, ...]` plus an `env_factory()` that
returns a fresh `Env`. The Env owns the system prompt, message construction, and the
reward computation. See env.py for built-in envs.
"""
import re
from datasets import load_dataset

from env import CalculatorEnv, PythonEnv, SingleTurnEnv

# --- shared scoring + dataset --------------------------------------------------

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
            question=ex["question"],
            answer=ex["answer"].split("####")[-1].strip().replace(",", ""),
        )
        for ex in load_dataset("openai/gsm8k", "main", split=split)
    ]

# --- registry: name -> (dataset_fn, env_factory) ------------------------------

TASKS = {
    "gsm8k":      (gsm8k_dataset, lambda: SingleTurnEnv(GSM8K_SYSTEM, gsm8k_reward)),
    "gsm8k_calc": (gsm8k_dataset, CalculatorEnv),
    "gsm8k_py":   (gsm8k_dataset, PythonEnv),
}

def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task '{name}', valid: {list(TASKS)}")
    return TASKS[name]
