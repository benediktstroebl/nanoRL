"""
nanoRL trainer. The whole trainer process lives in this file.

Launch
------
    # 1. Start serve.py on its own GPUs (see run.sh)
    # 2. Then:
    torchrun --nproc-per-node=N train.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --infer-url http://localhost:8000 --infer-tp M

Architecture
------------
This is one of two processes; the other is `serve.py` (vLLM, on a separate
GPU set). Communication:

    HTTP /generate       every step,    fetches a fresh batch of rollouts
    NCCL broadcast       every K steps, pushes new weights to vLLM workers

Async-ness comes from a single background thread on rank 0 that fills a
bounded rollout queue. The main loop never blocks on inference: it pops
the next batch and trains. Queue full = generator throttled. Queue empty
= trainer stalls (the only stall point). With max-async-steps=K, rollouts
are at most K trainer-steps stale; the importance-sampling ratio handles
the off-policy correction.

Loss
----
GRPO with PPO-style clipping. Per prompt we sample G responses, compute
advantage A_i = r_i - mean(r_*) (Dr.GRPO style — drop std; it amplifies
small reward differences and biases low-variance groups). Per-token loss
is averaged across all response tokens in the batch (DAPO-style; no
per-sequence normalization that would over-weight short responses).

No reference model. No value head. No KL by default. Modulo PPO clip
and IS correction, this is REINFORCE with a per-group baseline — the
same as nanochat's chat_rl.py, with the minimum modern trimmings to make
it correct under async (off-policy) rollouts.
"""
import argparse
import json
import os
import queue
import random
import threading
import time

import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks import get_task

# -----------------------------------------------------------------------------
# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--infer-url", default="http://localhost:8000")
    p.add_argument("--infer-tp", type=int, default=1, help="vllm world size (for NCCL group)")
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--rollouts-per-prompt", type=int, default=8, help="GRPO group size G")
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-response-tokens", type=int, default=1024)
    p.add_argument("--microbatch-size", type=int, default=0,
                   help="seqs per fwd/bwd microbatch on each rank; 0 = all-at-once")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--betas", default="0.9,0.95")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO ratio clip")
    p.add_argument("--max-async-steps", type=int, default=2, help="rollout queue depth")
    p.add_argument("--weight-sync-interval", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="out")
    p.add_argument("--save-interval", type=int, default=0, help="0 = never")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--run", default=None, help="wandb run name; disabled if not set")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Distributed init + helpers

def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world, local_rank, device

def is_master():
    return dist.get_rank() == 0

def print0(*a, **k):
    if is_master():
        print(*a, **k, flush=True)

class _DummyWandb:
    def log(self, *a, **k): pass
    def finish(self): pass

def init_wandb(args):
    if args.run is None or not is_master():
        return _DummyWandb()
    import wandb
    return wandb.init(project="nanoRL", name=args.run, config=vars(args))

# -----------------------------------------------------------------------------
# Model wrapping (FSDP2)

def setup_fsdp(model):
    """Apply FSDP2: shard each transformer block and the whole module."""
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    blocks = (
        getattr(getattr(model, "model", None), "layers", None)
        or getattr(model, "transformer_blocks", None)
        or getattr(model, "h", None)
        or []
    )
    for blk in blocks:
        fully_shard(blk, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model

# -----------------------------------------------------------------------------
# Weight sync: trainer rank 0 + all vLLM workers form a NCCL group on the side.

class WeightSync:
    """Broadcasts trainer params to vLLM workers. Trainer rank 0 only; other
    ranks no-op except for a barrier so they don't race the next training step."""

    def __init__(self, infer_url, infer_tp, device):
        self.infer_url, self.infer_tp, self.device = infer_url, infer_tp, device
        self.comm = None

    def init(self):
        if not is_master():
            dist.barrier()
            return

        master_addr = os.environ.get("WEIGHT_SYNC_HOST", "localhost")
        master_port = int(os.environ.get("WEIGHT_SYNC_PORT", "29600"))
        world_size = 1 + self.infer_tp

        # Tell vLLM workers to join. Their broadcast.recv will block waiting for
        # the side group to form, so we need to do this concurrently with our own join.
        post_done = threading.Event()
        def _post():
            r = requests.post(f"{self.infer_url}/init_weight_sync", json={
                "master_addr": master_addr, "master_port": master_port,
                "world_size": world_size,
            }, timeout=300)
            r.raise_for_status()
            post_done.set()
        threading.Thread(target=_post, daemon=True).start()

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
        pg = StatelessProcessGroup.create(
            host=master_addr, port=master_port, rank=0, world_size=world_size,
        )
        self.comm = PyNcclCommunicator(pg, device=self.device)
        post_done.wait(timeout=300)
        dist.barrier()

    def push(self, model):
        if not is_master():
            dist.barrier()
            return

        # DTensor.shape gives the logical (full) shape, so we can build the manifest
        # without materializing any full tensors. We then gather + broadcast one
        # param at a time to keep peak memory low (only one full param resident).
        named = list(model.named_parameters())
        manifest = [(n, str(p.dtype).removeprefix("torch."), list(p.shape)) for n, p in named]

        post_done = threading.Event()
        def _post():
            r = requests.post(f"{self.infer_url}/update_weights",
                              json={"manifest": manifest}, timeout=600)
            r.raise_for_status()
            post_done.set()
        threading.Thread(target=_post, daemon=True).start()

        for _, p in named:
            full = p.full_tensor() if hasattr(p, "full_tensor") else p.data
            self.comm.broadcast(full, src=0, stream=torch.cuda.current_stream())
            del full

        post_done.wait(timeout=600)
        dist.barrier()

# -----------------------------------------------------------------------------
# Rollouts

def fetch_rollouts(infer_url, prompt_ids, *, n, max_tokens, temperature, top_p,
                   stop_token_ids):
    r = requests.post(f"{infer_url}/generate", json={
        "prompts": prompt_ids, "n": n, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p,
        "stop_token_ids": stop_token_ids,
    }, timeout=600)
    r.raise_for_status()
    return r.json()

# -----------------------------------------------------------------------------
# Advantages + GRPO loss

def compute_advantages(rewards):
    """Group-relative, mean-centered (Dr.GRPO style — drop std).
    rewards: [B, G] -> advantages: [B, G]."""
    return rewards - rewards.mean(dim=-1, keepdim=True)

def grpo_loss(logits, target_ids, response_mask, advantages, old_logp, *, clip_eps):
    """
    logits        [B, T, V]  full-sequence model output
    target_ids    [B, T]     input ids
    response_mask [B, T]     1 on response positions, 0 elsewhere
    advantages    [B]        broadcast across T
    old_logp      [B, T]     vLLM rollout logprob, populated only at response positions
    """
    # Predict tokens[1..T-1] from logits[0..T-2]; align all tensors to that frame.
    sl = logits[:, :-1]
    tg = target_ids[:, 1:]
    rm = response_mask[:, 1:].float()
    olp = old_logp[:, 1:]

    new_logp = -F.cross_entropy(
        sl.reshape(-1, sl.size(-1)).float(), tg.reshape(-1),
        reduction="none",
    ).view_as(tg)

    # log_ratio is 0 outside response positions (mask zeros it), so exp() = 1 there.
    # Clamp guards against early-step blow-up if rollouts and trainer drift fast.
    log_ratio = ((new_logp - olp) * rm).clamp(-20, 20)
    ratio = log_ratio.exp()

    A = advantages.unsqueeze(-1)
    surr1 = ratio * A
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * A
    token_loss = -torch.minimum(surr1, surr2)

    denom = rm.sum().clamp(min=1)
    loss = (token_loss * rm).sum() / denom

    with torch.no_grad():
        clipfrac = (((ratio - 1).abs() > clip_eps).float() * rm).sum() / denom
        # k1 KL estimator on response tokens
        kl = ((-log_ratio) * rm).sum() / denom
    return loss, {"loss": loss.detach(), "kl": kl, "clipfrac": clipfrac}

# -----------------------------------------------------------------------------
# Batch building

def build_batch(batch, tok, rank, world, max_total_len, device):
    """Flatten the rollout batch, take this rank's equal-sized slice, and pad.

    Caller asserts that B*G is divisible by world, so every rank gets the same
    number of sequences and the FSDP collectives stay in lockstep across ranks.
    """
    B = len(batch["prompt_ids"])
    G = len(batch["response_ids"][0])
    total = B * G
    n_per_rank = total // world

    # group-relative advantages, mean-centered (Dr.GRPO)
    rewards = torch.tensor(batch["rewards"], dtype=torch.float)
    advs = compute_advantages(rewards).flatten().tolist()

    # flatten to (full_ids, prompt_len, resp_logp, advantage), then truncate
    seqs = []
    for b in range(B):
        pids = batch["prompt_ids"][b]
        for g in range(G):
            full = pids + batch["response_ids"][b][g]
            rlp = batch["response_logprobs"][b][g]
            if len(full) > max_total_len:
                full = full[:max_total_len]
                rlp = rlp[: max_total_len - len(pids)]
            seqs.append((full, len(pids), rlp, advs[b * G + g]))

    my = seqs[rank * n_per_rank : (rank + 1) * n_per_rank]

    T = max(len(s[0]) for s in my)
    n = len(my)
    input_ids = torch.full((n, T), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((n, T), dtype=torch.long)
    rmask = torch.zeros((n, T), dtype=torch.long)
    olp = torch.zeros((n, T), dtype=torch.float)
    A = torch.zeros((n,), dtype=torch.float)

    for i, (full, plen, rlp, a) in enumerate(my):
        L = len(full)
        input_ids[i, :L] = torch.tensor(full)
        attn[i, :L] = 1
        rmask[i, plen:L] = 1
        olp[i, plen:L] = torch.tensor(rlp[: L - plen])
        A[i] = a

    return (input_ids.to(device), attn.to(device), rmask.to(device),
            olp.to(device), A.to(device))

def microbatches(tensors, mb_size):
    n = tensors[0].size(0)
    if mb_size <= 0 or mb_size >= n:
        yield tensors
        return
    for i in range(0, n, mb_size):
        yield tuple(t[i : i + mb_size] for t in tensors)

# -----------------------------------------------------------------------------
# Rollout worker (runs only on rank 0; pulls prompts, hits vLLM, computes rewards)

def rollout_worker(args, tok, train_data, reward_fn, rollout_q, eos_token_id):
    rng = random.Random(args.seed + 1)
    while True:
        try:
            samples = rng.sample(train_data, args.prompts_per_step)
            prompt_ids = [
                tok.apply_chat_template(s["messages"], add_generation_prompt=True, tokenize=True)
                for s in samples
            ]
            prompt_ids = [p[-args.max_prompt_tokens :] for p in prompt_ids]
            answers = [s["answer"] for s in samples]

            resp = fetch_rollouts(
                args.infer_url, prompt_ids,
                n=args.rollouts_per_prompt,
                max_tokens=args.max_response_tokens,
                temperature=args.temperature, top_p=args.top_p,
                stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
            )
            rewards = [
                [reward_fn(tok.decode(ids), ans) for ids in row]
                for row, ans in zip(resp["response_ids"], answers)
            ]
            rollout_q.put({
                "prompt_ids": prompt_ids,
                "response_ids": resp["response_ids"],
                "response_logprobs": resp["response_logprobs"],
                "rewards": rewards,
            })
        except Exception as e:
            print0(f"[rollout-worker] {type(e).__name__}: {e}; retrying in 5s")
            time.sleep(5)

# -----------------------------------------------------------------------------
# Save/load (single-file pickle, simple but lossy: re-init optimizer on resume)

def save_state(model, save_dir, step):
    sd = {}
    for name, p in model.named_parameters():
        full = p.full_tensor() if hasattr(p, "full_tensor") else p.data
        if is_master():
            sd[name] = full.detach().cpu()
    if is_master():
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"step_{step:06d}.pt")
        torch.save(sd, path)
        print0(f"[save] {path}")
    dist.barrier()

# -----------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    rank, world, local_rank, device = setup_distributed()
    torch.manual_seed(args.seed + rank)

    total_seqs = args.prompts_per_step * args.rollouts_per_prompt
    assert total_seqs % world == 0, (
        f"prompts-per-step * rollouts-per-prompt ({total_seqs}) must be divisible "
        f"by world size ({world}); pick a B*G that divides evenly."
    )

    print0(f"nanoRL trainer | rank {rank}/{world} on {device}")
    print0(f"args: {json.dumps(vars(args), indent=2)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print0(f"loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = setup_fsdp(model)
    model.train()

    betas = tuple(float(x) for x in args.betas.split(","))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=betas,
                              weight_decay=args.weight_decay, fused=False)

    sync = WeightSync(args.infer_url, args.infer_tp, device)
    sync.init()
    print0("[weight-sync] established")
    sync.push(model)
    print0("[weight-sync] initial push done")

    dataset_fn, reward_fn = get_task(args.task)
    train_data = dataset_fn("train") if is_master() else None
    print0(f"[data] {len(train_data) if is_master() else '?'} train examples")

    wandb_run = init_wandb(args)

    rollout_q = queue.Queue(maxsize=args.max_async_steps)
    if is_master():
        threading.Thread(
            target=rollout_worker, daemon=True,
            args=(args, tok, train_data, reward_fn, rollout_q, tok.eos_token_id),
        ).start()

    max_total_len = args.max_prompt_tokens + args.max_response_tokens
    t_last = time.time()

    for step in range(args.total_steps):
        # 1. Fetch a rollout batch (rank 0 from queue, broadcast to others).
        if is_master():
            batch = rollout_q.get()
            qsize = rollout_q.qsize()
        else:
            batch, qsize = None, None
        ob = [batch]
        dist.broadcast_object_list(ob, src=0)
        batch = ob[0]

        # 2. Build my slice of the batch.
        input_ids, attn, rmask, olp, A = build_batch(
            batch, tok, rank, world, max_total_len, device,
        )

        # 3. Forward + backward, possibly split into microbatches.
        optim.zero_grad(set_to_none=True)
        mb_size = args.microbatch_size or input_ids.size(0)
        n_mb = max(1, (input_ids.size(0) + mb_size - 1) // mb_size)
        agg = {"loss": 0.0, "kl": 0.0, "clipfrac": 0.0}
        for ids_, attn_, rm_, olp_, A_ in microbatches(
            (input_ids, attn, rmask, olp, A), mb_size,
        ):
            out = model(input_ids=ids_, attention_mask=attn_, use_cache=False)
            loss, info = grpo_loss(
                out.logits, ids_, rm_, A_, olp_, clip_eps=args.clip_eps,
            )
            (loss / n_mb).backward()
            for k in agg:
                agg[k] += float(info[k]) / n_mb

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        # 4. Push weights periodically.
        if (step + 1) % args.weight_sync_interval == 0:
            sync.push(model)

        # 5. Logging.
        if (step + 1) % args.log_interval == 0 and is_master():
            now = time.time(); dt = now - t_last; t_last = now
            r_mean = torch.tensor(batch["rewards"]).mean().item()
            r_max = torch.tensor(batch["rewards"]).max().item()
            print0(f"step {step+1:5d} | loss {agg['loss']:+.4f} | "
                   f"reward {r_mean:.3f} (max {r_max:.2f}) | "
                   f"kl {agg['kl']:.4f} | clipfrac {agg['clipfrac']:.3f} | "
                   f"gnorm {float(grad_norm):.2f} | "
                   f"qdepth {qsize} | dt {dt:.1f}s")
            wandb_run.log({
                "loss": agg["loss"], "reward_mean": r_mean, "reward_max": r_max,
                "kl": agg["kl"], "clipfrac": agg["clipfrac"],
                "grad_norm": float(grad_norm), "qdepth": qsize,
                "step_time_s": dt, "step": step + 1,
            })

        # 6. Save.
        if args.save_interval and (step + 1) % args.save_interval == 0:
            save_state(model, args.save_dir, step + 1)

    if args.save_interval:
        save_state(model, args.save_dir, args.total_steps)
    wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
