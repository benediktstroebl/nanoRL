"""
nanoRL inference server. Wraps vLLM with three RL-specific endpoints:

  POST /generate          -> token ids + per-token logprobs (the rollout)
  POST /init_weight_sync  -> have all vLLM workers join an NCCL group with
                             the trainer (rank 0 in the group)
  POST /update_weights    -> receive each (name, dtype, shape) via broadcast()
                             and load it into the model in place. The trainer
                             sends matching params on the same NCCL group.

vLLM's `worker_extension_cls` lets us mix new methods into every worker; they
are dispatched by `engine.collective_rpc(method, args=...)`. SamplingParams
with `logprobs=1` always returns the chosen token's logprob in the per-step
dict (top-1 + chosen-if-different).

This file is also imported (not run) by every vLLM worker process to resolve
`worker_extension_cls="serve.NanoRLWorker"`. So only the class lives at module
scope; engine + app are built inside main().
"""
import argparse, asyncio, uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt

# --- worker extension: methods mixed into every vLLM worker ------------------

class NanoRLWorker:
    def init_weight_sync(self, master_addr: str, master_port: int, world_size: int):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup
        rank = get_world_group().rank + 1  # trainer is rank 0; workers are 1..N-1.
        pg = StatelessProcessGroup.create(host=master_addr, port=master_port,
                                          rank=rank, world_size=world_size)
        self._weight_pg = PyNcclCommunicator(pg, device=self.device)

    def update_weight(self, name: str, dtype: str, shape: list):
        buf = torch.empty(tuple(shape), dtype=getattr(torch, dtype), device=self.device)
        self._weight_pg.broadcast(buf, src=0, stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=[(name, buf)])

# --- request / response schemas ----------------------------------------------

class GenReq(BaseModel):
    prompts: list[list[int]]
    n: int = 1
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    stop_token_ids: list[int] | None = None

class GenResp(BaseModel):
    response_ids: list[list[list[int]]]         # [B][n][T]
    response_logprobs: list[list[list[float]]]  # [B][n][T]

class InitSyncReq(BaseModel):
    master_addr: str; master_port: int; world_size: int

class UpdateWeightsReq(BaseModel):
    manifest: list[tuple[str, str, list[int]]]  # (name, dtype name, shape)

# --- entry point -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--cuda-graphs", action="store_true",
                   help="enable CUDA graphs (faster decode; rebuilds after weight sync)")
    return p.parse_args()


def main():
    args = parse_args()
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model, tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.mem, max_model_len=args.max_model_len,
        enforce_eager=not args.cuda_graphs,
        enable_prefix_caching=False,  # KV would go stale across weight syncs
        worker_extension_cls="serve.NanoRLWorker", dtype="bfloat16",
    ))
    app = FastAPI()

    @app.get("/health")
    async def health(): return {"ok": True}

    @app.post("/generate")
    async def generate(req: GenReq) -> GenResp:
        sp = SamplingParams(n=req.n, max_tokens=req.max_tokens,
                            temperature=req.temperature, top_p=req.top_p,
                            stop_token_ids=req.stop_token_ids, logprobs=1)
        async def gen_one(pids):
            last = None
            async for out in engine.generate(TokensPrompt(prompt_token_ids=pids),
                                             sp, uuid.uuid4().hex):
                last = out
            return last
        outs = await asyncio.gather(*(gen_one(p) for p in req.prompts))
        rids, rlps = [], []
        for o in outs:
            ids_n, lps_n = [], []
            for c in o.outputs:
                tok = list(c.token_ids)
                ids_n.append(tok)
                lps_n.append([c.logprobs[t][tid].logprob for t, tid in enumerate(tok)])
            rids.append(ids_n); rlps.append(lps_n)
        return GenResp(response_ids=rids, response_logprobs=rlps)

    @app.post("/init_weight_sync")
    async def init_weight_sync(req: InitSyncReq):
        await engine.collective_rpc("init_weight_sync",
                                    args=(req.master_addr, req.master_port, req.world_size))
        return {"ok": True}

    @app.post("/update_weights")
    async def update_weights(req: UpdateWeightsReq):
        # One collective_rpc per param ⇒ one NCCL broadcast per param. Trainer is
        # sending matching params concurrently (see push_weights in train.py).
        for name, dtype, shape in req.manifest:
            await engine.collective_rpc("update_weight", args=(name, dtype, shape))
        return {"ok": True}

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
