"""nanoRL trainer: one trainer process for an N-GPU torchrun launch.

Two processes total (this + serve.py). Communication: HTTP /generate for
rollouts (each step), NCCL broadcast for weights (every K steps). Async-ness
is a background thread on rank 0 that fills a bounded rollout queue; with
max-async-steps=K, rollouts are at most K trainer-steps stale, and the
importance ratio in the loss handles off-policy correction.

GRPO with PPO clip + IS correction. A_i = r_i - mean(r_*) (Dr.GRPO; no std).
Per-token loss averaged over batch tokens (DAPO). No ref model, no value
head, no KL by default.

Launch: ./run.sh, or `uv run torchrun --nproc-per-node=N train.py
--infer-url http://localhost:8000 --infer-tp M`.
"""
import argparse, json, os, queue, random, threading, time

import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks import get_task

SYNC_HOST, SYNC_PORT = "localhost", 29600  # NCCL rendezvous for weight sync

# --- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--infer-url", default="http://localhost:8000")
    p.add_argument("--infer-tp", type=int, default=1, help="vllm world size (NCCL group)")
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--rollouts-per-prompt", type=int, default=8, help="GRPO group size G")
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-response-tokens", type=int, default=1024)
    p.add_argument("--microbatch-size", type=int, default=0, help="0 = all-at-once")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO ratio clip")
    p.add_argument("--max-async-steps", type=int, default=2, help="rollout queue depth")
    p.add_argument("--weight-sync-interval", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="out")
    p.add_argument("--save-interval", type=int, default=0, help="0 = never")
    p.add_argument("--run", default=None, help="wandb run name; disabled if not set")
    return p.parse_args()

# --- distributed + logging helpers --------------------------------------------

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return dist.get_rank(), dist.get_world_size(), device

def is_master(): return dist.get_rank() == 0
def print0(*a, **k):
    if is_master(): print(*a, **k, flush=True)

class _Null:  # stub for wandb when --run is unset; any method call is a no-op
    def __getattr__(self, _): return lambda *a, **k: None

def init_wandb(args):
    if args.run is None or not is_master(): return _Null()
    import wandb
    return wandb.init(project="nanoRL", name=args.run, config=vars(args))

# --- FSDP2: shard each transformer block, then the whole module ---------------

def setup_fsdp(model):
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.model.layers:  # standard HF transformer convention (Qwen, Llama, ...)
        fully_shard(blk, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model.train()

# --- weight sync: trainer rank 0 + vLLM workers form a side NCCL group --------

def _post_async(url, body, timeout=300):
    """POST in a daemon thread; returns Event the caller waits on."""
    done = threading.Event()
    def go(): requests.post(url, json=body, timeout=timeout).raise_for_status(); done.set()
    threading.Thread(target=go, daemon=True).start()
    return done

def init_weight_sync(infer_url, infer_tp, device):
    """Forms NCCL group; returns the comm (None on non-master ranks)."""
    if not is_master():
        dist.barrier(); return None
    world = 1 + infer_tp
    done = _post_async(f"{infer_url}/init_weight_sync",
                       dict(master_addr=SYNC_HOST, master_port=SYNC_PORT, world_size=world))
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=SYNC_HOST, port=SYNC_PORT, rank=0, world_size=world)
    comm = PyNcclCommunicator(pg, device=device)
    done.wait(timeout=300); dist.barrier()
    return comm

def push_weights(comm, model, infer_url):
    """Broadcast every param to vLLM. comm=None ⇒ non-master (just barrier)."""
    if comm is None:
        dist.barrier(); return
    named = list(model.named_parameters())
    # DTensor.shape is logical (full), so we build the manifest without materializing.
    manifest = [(n, str(p.dtype).removeprefix("torch."), list(p.shape)) for n, p in named]
    done = _post_async(f"{infer_url}/update_weights", dict(manifest=manifest), timeout=600)
    for _, p in named:
        full = p.full_tensor() if hasattr(p, "full_tensor") else p.data
        comm.broadcast(full, src=0, stream=torch.cuda.current_stream())
    done.wait(timeout=600); dist.barrier()

# --- GRPO loss: PPO clip on importance ratio = π_θ / π_gen --------------------

def grpo_loss(logits, ids, response_mask, advantages, old_logp, *, clip_eps):
    sl, tg = logits[:, :-1], ids[:, 1:]
    rm = response_mask[:, 1:].float()
    new_logp = -F.cross_entropy(
        sl.reshape(-1, sl.size(-1)).float(), tg.reshape(-1), reduction="none",
    ).view_as(tg)
    log_ratio = ((new_logp - old_logp[:, 1:]) * rm).clamp(-20, 20)
    ratio, A = log_ratio.exp(), advantages[:, None]
    token_loss = -torch.minimum(ratio * A, ratio.clamp(1 - clip_eps, 1 + clip_eps) * A)
    denom = rm.sum().clamp(min=1)
    loss = (token_loss * rm).sum() / denom
    with torch.no_grad():
        kl = (-log_ratio * rm).sum() / denom
        clipfrac = (((ratio - 1).abs() > clip_eps).float() * rm).sum() / denom
    return loss, dict(loss=loss.detach(), kl=kl, clipfrac=clipfrac)

# --- build a padded tensor batch for this rank's slice of the rollouts --------

def build_batch(batch, tok, rank, world, max_total_len, device):
    B, G = len(batch["prompt_ids"]), len(batch["response_ids"][0])
    n = (B * G) // world
    rewards = torch.tensor(batch["rewards"], dtype=torch.float)
    advs = (rewards - rewards.mean(-1, keepdim=True)).flatten().tolist()  # Dr.GRPO

    # Build only this rank's slice: tuples of (full_seq, prompt_len, rollout_logp, adv).
    my = []
    for k in range(rank * n, (rank + 1) * n):
        b, g = divmod(k, G)
        plen = len(batch["prompt_ids"][b])
        full = (batch["prompt_ids"][b] + batch["response_ids"][b][g])[:max_total_len]
        my.append((full, plen,
                   batch["response_logprobs"][b][g][: len(full) - plen], advs[k]))

    T = max(len(s[0]) for s in my)
    ids = torch.full((n, T), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((n, T), dtype=torch.long)
    rmask = torch.zeros((n, T), dtype=torch.long)
    olp = torch.zeros((n, T), dtype=torch.float)
    A = torch.tensor([s[3] for s in my], dtype=torch.float)
    for i, (full, plen, rlp, _) in enumerate(my):
        L = len(full)
        ids[i, :L] = torch.tensor(full); attn[i, :L] = 1
        rmask[i, plen:L] = 1; olp[i, plen:L] = torch.tensor(rlp)
    return tuple(t.to(device) for t in (ids, attn, rmask, olp, A))

# --- rollout worker (rank 0 only): sample, generate, score, enqueue -----------

def rollout_worker(args, tok, train_data, reward_fn, rollout_q):
    rng = random.Random(args.seed + 1)
    while True:
        try:
            samples = rng.sample(train_data, args.prompts_per_step)
            prompts = [tok.apply_chat_template(s["messages"], add_generation_prompt=True,
                                               tokenize=True)[-args.max_prompt_tokens:]
                       for s in samples]
            r = requests.post(f"{args.infer_url}/generate", json=dict(
                prompts=prompts, n=args.rollouts_per_prompt,
                max_tokens=args.max_response_tokens,
                temperature=args.temperature, top_p=args.top_p,
                stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
            ), timeout=600).json()
            rewards = [[reward_fn(tok.decode(ids), s["answer"]) for ids in row]
                       for s, row in zip(samples, r["response_ids"])]
            rollout_q.put(dict(prompt_ids=prompts, rewards=rewards,
                               response_ids=r["response_ids"],
                               response_logprobs=r["response_logprobs"]))
        except Exception as e:
            print0(f"[rollout] {type(e).__name__}: {e}; retry in 5s"); time.sleep(5)

# --- save: gathered HF-format state_dict on rank 0 (optimizer not saved) ------

def save_state(model, save_dir, step):
    sd = {n: (p.full_tensor() if hasattr(p, "full_tensor") else p.data).detach().cpu()
          for n, p in model.named_parameters()}  # full_tensor() is a collective; all ranks call
    if is_master():
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"step_{step:06d}.pt")
        torch.save(sd, path); print0(f"[save] {path}")
    dist.barrier()

# --- main ---------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world, device = setup_distributed()
    torch.manual_seed(args.seed + rank)

    total = args.prompts_per_step * args.rollouts_per_prompt
    assert total % world == 0, f"B*G ({total}) must be divisible by world size ({world})"

    print0(f"nanoRL | rank {rank}/{world} on {device}")
    print0(f"args: {json.dumps(vars(args), indent=2)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

    print0(f"loading {args.model}")
    model = setup_fsdp(AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    comm = init_weight_sync(args.infer_url, args.infer_tp, device)
    push_weights(comm, model, args.infer_url)
    print0("[weight-sync] ready")

    dataset_fn, reward_fn = get_task(args.task)
    train_data = dataset_fn("train")
    print0(f"[data] {len(train_data)} train examples")

    wandb_run = init_wandb(args)
    rollout_q = queue.Queue(maxsize=args.max_async_steps)
    if is_master():
        threading.Thread(target=rollout_worker, daemon=True,
                         args=(args, tok, train_data, reward_fn, rollout_q)).start()

    max_total_len = args.max_prompt_tokens + args.max_response_tokens
    t_last = time.time()

    for step in range(args.total_steps):
        # Fetch + broadcast rollouts.
        qsize = rollout_q.qsize() if is_master() else 0
        ob = [rollout_q.get() if is_master() else None]
        dist.broadcast_object_list(ob, src=0)
        batch = ob[0]

        # Build my slice as tensors.
        ids, attn, rmask, olp, A = build_batch(batch, tok, rank, world, max_total_len, device)

        # Forward + backward, optional microbatching.
        optim.zero_grad(set_to_none=True)
        mb = args.microbatch_size or ids.size(0)
        n_mb = (ids.size(0) + mb - 1) // mb
        agg = dict(loss=0.0, kl=0.0, clipfrac=0.0)
        for ids_, attn_, rm_, olp_, A_ in zip(*(t.split(mb) for t in (ids, attn, rmask, olp, A))):
            out = model(input_ids=ids_, attention_mask=attn_, use_cache=False)
            loss, info = grpo_loss(out.logits, ids_, rm_, A_, olp_, clip_eps=args.clip_eps)
            (loss / n_mb).backward()
            for k in agg: agg[k] += float(info[k]) / n_mb

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (step + 1) % args.weight_sync_interval == 0:
            push_weights(comm, model, args.infer_url)

        if is_master():
            dt = time.time() - t_last; t_last += dt
            r = torch.tensor(batch["rewards"])
            rmean, rmax, gn = r.mean().item(), r.max().item(), float(gnorm)
            print0(f"step {step+1:5d} | loss {agg['loss']:+.4f} | reward {rmean:.3f} (max {rmax:.2f}) | "
                   f"kl {agg['kl']:.4f} | clipfrac {agg['clipfrac']:.3f} | gnorm {gn:.2f} | "
                   f"qdepth {qsize} | dt {dt:.1f}s")
            wandb_run.log(dict(step=step+1, **agg, reward_mean=rmean, reward_max=rmax,
                               grad_norm=gn, qdepth=qsize, dt=dt))

        if args.save_interval and (step + 1) % args.save_interval == 0:
            save_state(model, args.save_dir, step + 1)

    if args.save_interval: save_state(model, args.save_dir, args.total_steps)
    wandb_run.finish(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
