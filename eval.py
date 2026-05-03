"""nanoRL eval. Hits a running serve.py on the test split and reports pass@1
and pass@k via the same multi-turn rollout as training. Run after `./run.sh`
has started serve.py — the trainer doesn't need to be running.

    uv run python eval.py --task gsm8k --n 200 --k 4
    uv run python eval.py --task gsm8k_calc --n 200 --k 4
"""
import argparse
import random
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from env import rollout_one
from tasks import get_task


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="for the tokenizer; the inference server has the weights")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=256, help="number of problems")
    p.add_argument("--k", type=int, default=4, help="samples per problem")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--infer-url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=32, help="parallel rollouts in flight")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    dataset_fn, env_factory = get_task(args.task)
    data = dataset_fn(args.split)
    random.Random(args.seed).shuffle(data)
    data = data[: args.n]

    rollout_kwargs = dict(
        max_response_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p,
        max_total_len=args.max_prompt_tokens + args.max_tokens,
    )

    def k_rewards(sample):
        return [rollout_one(env_factory(), tok, args.infer_url, sample, **rollout_kwargs)["reward"]
                for _ in range(args.k)]

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        all_rewards = []
        for i, rs in enumerate(ex.map(k_rewards, data), 1):
            all_rewards.append(rs)
            if i % 10 == 0 or i == len(data):
                print(f"  {i}/{len(data)}")

    p1 = sum(rs[0] for rs in all_rewards)
    pk = sum(float(max(rs) > 0) for rs in all_rewards)
    n = len(data)
    print(f"task={args.task} split={args.split} n={n}: "
          f"pass@1 = {p1/n:.3f} | pass@{args.k} = {pk/n:.3f}")


if __name__ == "__main__":
    main()
