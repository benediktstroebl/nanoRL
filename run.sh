#!/bin/bash
# nanoRL launcher. Splits GPUs between trainer (FSDP) and inference (vLLM),
# starts both, and tears the inference server down when training exits.
#
# Usage:
#   uv sync                                   # one-time: create venv with deps
#   ./run.sh                                  # 2 train + 2 infer GPUs (default)
#   ./run.sh --train-gpus 4 --infer-gpus 4    # 8-GPU node, even split
#   ./run.sh -- --total-steps 5000 --lr 5e-7  # forwards anything after `--` to train.py
set -eo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_GPUS=${TRAIN_GPUS:-2}
INFER_GPUS=${INFER_GPUS:-2}
PORT=${PORT:-8000}

TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-gpus) TRAIN_GPUS="$2"; shift 2;;
    --infer-gpus) INFER_GPUS="$2"; shift 2;;
    --model)      MODEL="$2";      shift 2;;
    --port)       PORT="$2";       shift 2;;
    --) shift; TRAIN_ARGS+=("$@"); break;;
    *)  TRAIN_ARGS+=("$1"); shift;;
  esac
done

TRAIN_DEVS=$(seq -s, 0 $((TRAIN_GPUS-1)))
INFER_DEVS=$(seq -s, $TRAIN_GPUS $((TRAIN_GPUS+INFER_GPUS-1)))

echo "[run] $TRAIN_GPUS train ($TRAIN_DEVS) + $INFER_GPUS infer ($INFER_DEVS) GPUs"
echo "[run] model=$MODEL"

CUDA_VISIBLE_DEVICES=$INFER_DEVS uv run python serve.py \
    --model "$MODEL" --tp $INFER_GPUS --port $PORT &
SERVE_PID=$!
trap "echo '[run] stopping serve ($SERVE_PID)'; kill $SERVE_PID 2>/dev/null || true" EXIT

echo "[run] waiting for serve.py on :$PORT"
until curl -sf http://localhost:$PORT/health > /dev/null; do
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "[run] serve.py died" >&2; exit 1
  fi
  sleep 2
done
echo "[run] serve.py up"

CUDA_VISIBLE_DEVICES=$TRAIN_DEVS uv run torchrun --nproc-per-node=$TRAIN_GPUS train.py \
    --model "$MODEL" \
    --infer-url "http://localhost:$PORT" \
    --infer-tp $INFER_GPUS \
    "${TRAIN_ARGS[@]}"
