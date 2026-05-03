# nanoRL

A minimal, modern, async RL framework for language models, in the spirit of
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[nanochat](https://github.com/karpathy/nanochat). Five Python files,
**~780 lines** — read it in 30 minutes.

It does the things modern RL frameworks do:

- **Two processes:** an FSDP2 trainer and a vLLM inference server.
- **Per-prompt streaming async:** `--num-rollout-workers` (default 4) parallel
  threads on rank 0 each run multi-turn rollouts and feed a bounded
  per-prompt-group queue. The trainer dequeues B groups per step. Rollouts
  are at most `--max-async-steps` trainer-steps stale.
- **Off-policy correction:** the GRPO loss uses a PPO importance ratio against
  the rollout-time logprobs returned by vLLM, so trainer and sampler can drift
  by the staleness budget without bias.
- **In-place weight sync, race-free:** trainer rank 0 forms a side NCCL group
  with all vLLM workers and broadcasts every parameter. Wrapped in `/pause`
  → broadcast → `/resume` so vLLM can't schedule a generation step on a
  partially-updated model.
- **Multi-turn agent RL:** a tiny `Env` ABC with a tool registry. Built-in
  envs: `SingleTurnEnv` (classic), `CalculatorEnv` (`<calc>`), `PythonEnv`
  (`<python>`). Loss masks observation tokens so credit only flows to tokens
  the model generated.

It does *not* do the things that double the line count without changing the
science: no Ray, no orchestrator process, no ZMQ, no value head, no reference
model, no KL by default, no LoRA, no checkpoint conversion pipeline, no
multi-node. Single-node tool for short-horizon RL on math and similar
verifiable tasks.

## The whole thing

```
nanoRL/
├── train.py        # FSDP trainer + N-worker per-group queue + GRPO + weight push  (309)
├── serve.py        # vLLM server: /generate, /pause, /resume, weight-sync          (154)
├── env.py          # Env ABC, SingleTurnEnv, ToolEnv, CalculatorEnv, PythonEnv +
│                   # the multi-turn rollout loop (used by train + eval)            (200)
├── tasks.py        # GSM8K dataset; registry maps name -> (dataset, env_factory)    (50)
├── eval.py         # pass@k via the same multi-turn rollout                         (69)
├── run.sh          # split GPUs, launch both, trap-kill on exit                     (53)
├── pyproject.toml  # deps (managed by uv)
└── README.md
```

## Architecture

```
                       prompts (HTTP /generate)
        ┌─────────────────────────────────────────────────┐
        │     ▼  rollouts (token ids + logprobs)          │
   ┌────┴───────┐                                  ┌──────┴───────┐
   │  train.py  │   ─── /pause ── broadcast ── ───▶│  serve.py    │
   │  rank 0    │   ─── /update_weights (HTTP) ───▶│  vLLM        │
   │  rank 1    │   ─── /resume ──────────────────▶│  workers     │
   │  ...       │   ─── NCCL broadcast ───────────▶│  0..M-1      │
   │  rank N-1  │                                  │              │
   └────────────┘                                  └──────────────┘
       FSDP2                                          continuous
                                                      batching
```

`train.py` is launched with `torchrun --nproc-per-node=N`. All ranks shard
the model with FSDP2; only rank 0 talks to vLLM. Other ranks just do their
share of forward/backward.

`serve.py` is a FastAPI server wrapping `vllm.AsyncLLMEngine`. Custom endpoints:
`/generate`, `/init_weight_sync`, `/update_weights`, `/pause`, `/resume`. The
pause gate is a single `asyncio.Event` plus an in-flight counter — `/pause`
returns once the counter hits zero, so the trainer is guaranteed exclusive
access for the broadcast loop.

`env.py` defines the `Env` ABC and the `rollout_one` function used by both
the trainer (per worker thread) and `eval.py` (per `ThreadPoolExecutor` slot).
Each call drives one trajectory: `env.reset` → loop {`/generate` → `env.step`
→ append observation} until `done` or the turn budget is exhausted.

`run.sh` partitions GPUs via `CUDA_VISIBLE_DEVICES`, starts vLLM on the
inference half, waits for `/health`, then `torchrun`s the trainer.

## Algorithm

For each prompt q sample G trajectories from vLLM, score each
r_i = env's terminal reward, and compute

    A_i  = r_i - mean(r_*)                              # mean-centered (Dr.GRPO)
    ratio = exp(logπ_θ - logπ_gen)                       # IS against rollout-time
    Lclip = min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)        # PPO clip
    loss  = -mean over generated tokens of Lclip         # DAPO normalization

A single `loss_mask` (1 on tokens the model generated, 0 on prompt + observations)
covers single-turn and multi-turn the same way: in single-turn it's just the
response; in multi-turn it's the union of all assistant turns across the
trajectory.

That's the whole thing. No value head, no reference model, no KL by default.
The PPO clip handles the off-policy correction implicitly when rollouts are
generated against a stale policy.

## How to run

```bash
# One-time setup.
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is installed
uv sync                                            # add --extra wandb for wandb
huggingface-cli login                              # if your model is gated

# 4 GPUs: 2 trainer + 2 inference, default model, classic single-turn GSM8K.
./run.sh

# 8 GPUs, 4+4, longer run, smaller LR.
./run.sh --train-gpus 4 --infer-gpus 4 -- --total-steps 5000 --lr 5e-7

# Multi-turn agent RL on GSM8K with a calculator tool.
./run.sh -- --task gsm8k_calc --rollouts-per-prompt 8

# Or with a tiny python sandbox.
./run.sh -- --task gsm8k_py

# Eval (serve.py must still be running):
uv run python eval.py --task gsm8k --n 200 --k 4
uv run python eval.py --task gsm8k_calc --n 200 --k 4
```

You should see something like:

```
step    1 | loss +0.0021 | reward 0.062 (max 1.00) | kl 0.0000 | clipfrac 0.00 | gnorm 0.21 | qdepth 8 | stale 0 | dt 8.4s
step   10 | loss -0.0143 | reward 0.158 (max 1.00) | kl 0.0021 | clipfrac 0.02 | gnorm 0.45 | qdepth 6 | stale 1 | dt 7.9s
step  100 | loss -0.0298 | reward 0.392 (max 1.00) | kl 0.0148 | clipfrac 0.07 | gnorm 0.61 | qdepth 4 | stale 2 | dt 7.6s
```

- `qdepth` — rollout queue depth at step start. Full = inference faster than
  training; empty = trainer starved.
- `stale` — max staleness in this step's batch (number of trainer steps
  between when each rollout started and now). Bounded by `--max-async-steps`.

## Adding a tool

Subclass `ToolEnv` in `env.py`. Add `_tool_<name>(args)` for each tool you
want; the framework parses `<name>...</name>` tags and dispatches automatically.
Implement `_score(answer_text)` for terminal scoring on the model's
`<answer>...</answer>`. Wire it into `tasks.py`'s `TASKS` registry.

```python
class WikiEnv(ToolEnv):
    SYSTEM = "Search Wikipedia with <wiki>query</wiki>; answer with <answer>X</answer>."
    MAX_TURNS = 6
    def _tool_wiki(self, query): return wikipedia.summary(query, sentences=2)
    def _score(self, ans):       return float(ans.strip().lower() == self._answer.lower())
```

```python
TASKS["nq_wiki"] = (nq_dataset, WikiEnv)
```

## Hacking

**Add a single-turn task.** `dataset(split) -> [{question, answer}]` plus a
`reward(text, answer)` function, then register
`(dataset, lambda: SingleTurnEnv(SYSTEM_STR, reward))` in `TASKS`.

**Add a loss variant.** `grpo_loss` in `train.py` is 15 lines. Asymmetric
clip (DAPO clip-higher): replace `(1-eps, 1+eps)`. KL term: add
`(logp_ref - logp).exp() - (logp_ref - logp) - 1` and load a reference
model. Original GRPO per-sequence normalization: replace the global
token-mean with a per-sequence mean.

**Switch model.** Pass `--model Qwen/Qwen2.5-7B-Instruct` to `run.sh`. FSDP2
shards transformer blocks; nothing else changes. (Bump `--max-model-len`
and lower `--prompts-per-step` until it fits.)

**Tune throughput.** Increase `--num-rollout-workers` until vLLM saturates
(check the logged `qdepth` — if it stays at 0, generators are the
bottleneck; if it pegs at the cap, training is the bottleneck). Bump
`--max-async-steps` to allow more pipelining at the cost of more stale
rollouts.

## What this is *not*

There is no graceful shutdown, no checkpoint resumption, no eval during
training, no curriculum, no multi-node, no LoRA, no quantization, no MoE,
no reward-model training, no DPO/IPO/KTO, no reference model + KL term, no
DAPO clip-higher / dynamic sampling. Adding any is a project; the point of
this repo is to be the smallest thing you can read end-to-end and that
*actually does* async RL with multi-turn tools.

## References

- [GRPO](https://arxiv.org/abs/2402.03300), [Dr.GRPO](https://arxiv.org/abs/2503.20783),
  [DAPO](https://arxiv.org/abs/2503.14476) — the loss this implements.
- [vLLM RLHF example](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf.html)
  — `worker_extension_cls` + `StatelessProcessGroup` weight-sync pattern.
- [SkyRL](https://github.com/NovaSky-AI/SkyRL) — pause/resume around weight
  updates, fully-async generator/trainer split, multi-turn agent loop with
  tool registry. The structure of nanoRL's three new features mirrors
  SkyRL's at minimum scale.
- [nanochat chat_rl.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py)
  — synchronous, on-policy ancestor.
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — production
  three-process trainer/orchestrator/inference design.
