"""
nanoRL inference server. Wraps vLLM with two RL-specific endpoints.

Endpoints
---------
  GET  /health
  POST /generate          -> token ids + per-token logprobs (the rollout)
  POST /init_weight_sync  -> have all vLLM workers join an NCCL group with
                             the trainer (rank 0 in the group)
  POST /update_weights    -> for each (name, dtype, shape) in the manifest,
                             every worker calls broadcast() to receive a
                             tensor and load it into the model in place.
                             The trainer is sending the matching params on
                             the same NCCL group at the same time.

Why a custom server (instead of vLLM's stock OpenAI server)? We need the
sampled token's logprob (for the importance ratio in the trainer), and we
need a hook to receive weights without a process restart. Both are simple
to add on top of vLLM's primitives: SamplingParams(logprobs=1) gives the
chosen token's logprob, and `worker_extension_cls` lets us mix new methods
into every worker, callable via `engine.collective_rpc`.

This file is *also* imported (not run) by every vLLM worker process — vLLM
needs to resolve `worker_extension_cls="serve.NanoRLWorker"`. Therefore
module-level code must be safe to import: only the class lives at module
scope; engine + app are built inside main().
"""
import argparse
import asyncio
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt

# -----------------------------------------------------------------------------
# Worker extension: methods mixed into each vLLM worker, called via collective_rpc.
# Every method here runs once per worker (e.g. once per TP rank).

class NanoRLWorker:
    def init_weight_sync(self, master_addr: str, master_port: int, world_size: int):
        # Trainer rank 0 of the side group runs at rank 0; vLLM workers fill 1..N-1.
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup
        rank = get_world_group().rank + 1
        pg = StatelessProcessGroup.create(
            host=master_addr, port=master_port, rank=rank, world_size=world_size,
        )
        self._weight_pg = PyNcclCommunicator(pg, device=self.device)

    def update_weight(self, name: str, dtype: str, shape: list):
        dt = getattr(torch, dtype)
        buf = torch.empty(tuple(shape), dtype=dt, device=self.device)
        self._weight_pg.broadcast(buf, src=0, stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=[(name, buf)])
        del buf

# -----------------------------------------------------------------------------
# Request / response schemas

class GenReq(BaseModel):
    prompts: list[list[int]]
    n: int = 1
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    stop_token_ids: Optional[list[int]] = None

class GenResp(BaseModel):
    response_ids: list[list[list[int]]]         # [B][n][T]
    response_logprobs: list[list[list[float]]]  # [B][n][T]
    finish_reasons: list[list[str]]             # [B][n]

class InitSyncReq(BaseModel):
    master_addr: str
    master_port: int
    world_size: int

class UpdateWeightsReq(BaseModel):
    manifest: list[tuple[str, str, list[int]]]  # (name, torch dtype name, shape)

# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--cuda-graphs", action="store_true",
                   help="enable CUDA graphs (faster decode, but rebuilds after each weight sync)")
    return p.parse_args()


def main():
    args = parse_args()
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.mem,
        max_model_len=args.max_model_len,
        enforce_eager=not args.cuda_graphs,
        enable_prefix_caching=False,  # KV would go stale across weight syncs
        worker_extension_cls="serve.NanoRLWorker",
        dtype="bfloat16",
    ))

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/generate")
    async def generate(req: GenReq) -> GenResp:
        sp = SamplingParams(
            n=req.n, max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, stop_token_ids=req.stop_token_ids, logprobs=1,
        )

        async def gen_one(pids):
            rid = uuid.uuid4().hex
            last = None
            async for out in engine.generate(
                TokensPrompt(prompt_token_ids=pids), sp, rid,
            ):
                last = out
            return last

        outs = await asyncio.gather(*(gen_one(p) for p in req.prompts))

        rids, rlps, frs = [], [], []
        for o in outs:
            i_n, l_n, f_n = [], [], []
            for c in o.outputs:
                tok = list(c.token_ids)
                # logprobs=1 returns top-1 + chosen-if-different, so the chosen
                # token is always present in the per-step dict.
                lps = [c.logprobs[t][tid].logprob for t, tid in enumerate(tok)]
                i_n.append(tok); l_n.append(lps); f_n.append(c.finish_reason or "stop")
            rids.append(i_n); rlps.append(l_n); frs.append(f_n)
        return GenResp(response_ids=rids, response_logprobs=rlps, finish_reasons=frs)

    @app.post("/init_weight_sync")
    async def init_weight_sync(req: InitSyncReq):
        await engine.collective_rpc(
            "init_weight_sync",
            args=(req.master_addr, req.master_port, req.world_size),
        )
        return {"ok": True}

    @app.post("/update_weights")
    async def update_weights(req: UpdateWeightsReq):
        # One collective_rpc per param ⇒ one NCCL broadcast per param.
        # The trainer is sending the matching params concurrently (see push_weights).
        for name, dtype, shape in req.manifest:
            await engine.collective_rpc("update_weight", args=(name, dtype, shape))
        return {"ok": True}

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
