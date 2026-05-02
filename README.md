# nanoRL

A minimal, modern, async RL framework for language models, in the spirit of
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[nanochat](https://github.com/karpathy/nanochat). Four Python files,
**~530 lines** — read it in 30 minutes.

It does the things modern RL frameworks do:

- **Two processes:** an FSDP2 trainer and a vLLM inference server.
- **Async rollouts:** a single background thread fills a bounded queue;
  the trainer pops batches and never waits on inference until the queue
  is empty.
- **Off-policy correction:** the GRPO loss uses a PPO-style importance
  ratio against the rollout-time logprobs returned by vLLM, so trainer
  and sampler can drift by `--max-async-steps` steps without bias.
- **In-place weight sync:** trainer rank 0 forms a side NCCL group with
  all vLLM workers and broadcasts parameters every step — no checkpoint
  files, no process restart.

It does *not* do the things that double the line count without changing
the science: no Ray, no orchestrator process, no ZMQ, no value head, no
reference model, no KL by default, no LoRA, no multi-turn agent loop, no
checkpoint conversion pipeline. Single-node tool for short-horizon RL on
math and similar verifiable tasks.

## The whole thing

```
nanoRL/
├── train.py        # FSDP trainer + GRPO + rollout queue + weight push  (286)
├── serve.py        # vLLM server: /generate + weight-sync endpoints     (134)
├── tasks.py        # GSM8K dataset + reward                              (41)
├── eval.py         # pass@k on the test split                            (66)
├── run.sh          # split GPUs, launch both, trap-kill on exit          (53)
├── pyproject.toml  # deps (managed by uv)
└── README.md
```

## Architecture

```
                       prompts (HTTP /generate)
        ┌─────────────────────────────────────────────────┐
        │     ▼  rollouts (token ids + logprobs)          │
   ┌────┴───────┐                                  ┌──────┴───────┐
   │  train.py  │                                  │  serve.py    │
   │  rank 0    │  ◀───── /update_weights (HTTP) ──┤  vLLM        │
   │  rank 1    │                                  │  workers     │
   │  ...       │  ─────  NCCL broadcast  ────────▶│  0..M-1      │
   │  rank N-1  │                                  │              │
   └────────────┘                                  └──────────────┘
       FSDP2                                          continuous
                                                      batching
```

`train.py` is launched with `torchrun --nproc-per-node=N`. All ranks shard
the model with FSDP2; only rank 0 talks to vLLM (HTTP for rollouts, NCCL
for weights). Other ranks just do their share of forward/backward.

`serve.py` is a FastAPI server wrapping `vllm.AsyncLLMEngine`. The two
RL endpoints (`/init_weight_sync`, `/update_weights`) sit alongside
`/generate`. Weight sync uses vLLM's `worker_extension_cls`: every worker
gets methods that join an NCCL group and `broadcast()`-receive a tensor.

`run.sh` partitions GPUs via `CUDA_VISIBLE_DEVICES`, starts vLLM on the
inference half, waits for `/health`, then `torchrun`s the trainer on
the training half. Killing the trainer kills the server via `trap`.

## Algorithm

For each prompt q sample G responses {o_1, …, o_G} from vLLM, score each
r_i = reward(text(o_i), answer), and compute

    A_i  = r_i - mean(r_*)                            # mean-centered (Dr.GRPO)
    ratio = exp(logπ_θ(o_i) - logπ_gen(o_i))           # IS against rollout-time
    Lclip = min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)      # PPO clip
    loss  = -mean over response tokens of Lclip        # DAPO normalization

That's the whole thing. No value head, no reference model, no KL by
default. The PPO clip handles the off-policy correction implicitly when
rollouts are generated against a stale policy.

## How to run

```bash
# One-time setup.
curl -LsSf https://astral.sh/uv/install.sh | sh    # skip if uv is installed
uv sync                                             # add --extra wandb for wandb
huggingface-cli login                               # if your model is gated

# 4 GPUs: 2 trainer + 2 inference, default model (Qwen2.5-0.5B-Instruct), GSM8K.
./run.sh

# 8 GPUs, 4+4, longer run, smaller LR.
./run.sh --train-gpus 4 --infer-gpus 4 -- --total-steps 5000 --lr 5e-7

# Eval (serve.py must still be running):
uv run python eval.py --task gsm8k --n 200 --k 4
```

You should see something like:

```
step    1 | loss +0.0021 | reward 0.062 (max 1.00) | kl 0.0000 | clipfrac 0.00 | gnorm 0.21 | qdepth 1 | dt 8.4s
step   10 | loss -0.0143 | reward 0.158 (max 1.00) | kl 0.0021 | clipfrac 0.02 | gnorm 0.45 | qdepth 1 | dt 7.9s
step  100 | loss -0.0298 | reward 0.392 (max 1.00) | kl 0.0148 | clipfrac 0.07 | gnorm 0.61 | qdepth 2 | dt 7.6s
```

`qdepth` is the rollout queue depth at step start: queue full = inference
faster than training; queue empty = trainer starved. With `--max-async-
steps 2` (default), rollouts are at most 2 trainer-steps stale.

## Hacking

**Add a task.** Write `dataset(split) -> [{messages, answer}]` and
`reward(text, answer) -> float in [0,1]` and register both in `TASKS`
in `tasks.py`.

**Add a loss variant.** `grpo_loss` in `train.py` is 15 lines. Want
asymmetric clip (DAPO clip-higher)? Replace `(1-eps, 1+eps)`. Want KL?
Add `(logp_ref - logp).exp() - (logp_ref - logp) - 1` and load a
reference model. Want original GRPO sequence-level normalization?
Replace the global token-mean with a per-sequence mean.

**Switch model.** Pass `--model Qwen/Qwen2.5-7B-Instruct` to `run.sh`.
FSDP2 shards transformer blocks; nothing else changes. (Bump
`--max-model-len` and lower `--prompts-per-step` until it fits.)

**Multi-turn / tool use.** Out of scope. Look at
[SkyRL](https://github.com/NovaSky-AI/SkyRL).

## References

- [GRPO](https://arxiv.org/abs/2402.03300), [Dr.GRPO](https://arxiv.org/abs/2503.20783),
  [DAPO](https://arxiv.org/abs/2503.14476) — the loss this implements.
- [vLLM RLHF example](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf.html)
  — `worker_extension_cls` + `StatelessProcessGroup` weight-sync pattern.
- [nanochat chat_rl.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py)
  — synchronous, on-policy ancestor.
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) /
  [SkyRL](https://github.com/NovaSky-AI/SkyRL) — production-scale
  versions of what's in this folder.
