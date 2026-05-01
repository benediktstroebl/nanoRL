"""
nanoRL eval. Hits a running inference server on the test split and reports
pass@1 and pass@k (k = samples per problem). Run after `run.sh` has started
serve.py — the trainer doesn't need to be running.

    python eval.py --task gsm8k --n 200 --k 4
"""
import argparse
import random

import requests
from transformers import AutoTokenizer

from tasks import get_task


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="for the tokenizer; the inference server holds the actual weights")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=256, help="number of problems")
    p.add_argument("--k", type=int, default=4, help="samples per problem")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--infer-url", default="http://localhost:8000")
    p.add_argument("--batch", type=int, default=32, help="prompts per HTTP call")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    dataset_fn, reward_fn = get_task(args.task)
    data = dataset_fn(args.split)

    rng = random.Random(args.seed)
    rng.shuffle(data)
    data = data[: args.n]

    prompt_ids = [
        tok.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=True)[
            -args.max_prompt_tokens :
        ]
        for d in data
    ]

    all_responses = []
    for i in range(0, len(prompt_ids), args.batch):
        chunk = prompt_ids[i : i + args.batch]
        r = requests.post(f"{args.infer_url}/generate", json={
            "prompts": chunk, "n": args.k,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "stop_token_ids": [tok.eos_token_id] if tok.eos_token_id else None,
        }, timeout=600)
        r.raise_for_status()
        all_responses.extend(r.json()["response_ids"])
        print(f"  generated {min(i + args.batch, len(prompt_ids))}/{len(prompt_ids)}")

    p_at_k = 0
    p_at_1 = 0.0
    for d, samples in zip(data, all_responses):
        texts = [tok.decode(ids) for ids in samples]
        rewards = [reward_fn(t, d["answer"]) for t in texts]
        p_at_1 += rewards[0]
        if max(rewards) > 0.0:
            p_at_k += 1

    n = len(data)
    print(f"task={args.task} split={args.split} n={n} k={args.k}")
    print(f"  pass@1  = {p_at_1 / n:.3f}")
    print(f"  pass@{args.k} = {p_at_k / n:.3f}")


if __name__ == "__main__":
    main()
