#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
B12X_DIR="${B12X_DIR:-$HOME/projects/b12x}"

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR/python:$B12X_DIR:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export VIRTUAL_TP_SHARDING="${VIRTUAL_TP_SHARDING:-b12x-padded}"

export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_OPT_USE_B12X_MHC=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_USE_TILELANG_INDEXER=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_OPT_FP8_WO_A_GEMM="${SGLANG_OPT_FP8_WO_A_GEMM:-1}"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}"
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=high

TP_SIZE="${TP_SIZE:-3}"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ "$TP_SIZE" != "3" ]]; then
  echo "serve-dsv4f-tp3.sh expects TP_SIZE=3, got TP_SIZE=$TP_SIZE" >&2
  exit 1
fi
if [[ "${#VISIBLE_GPU_LIST[@]}" -ne 3 ]]; then
  echo "serve-dsv4f-tp3.sh expects exactly 3 CUDA_VISIBLE_DEVICES, got '$CUDA_VISIBLE_DEVICES'" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  if [[ -d /data/models/DeepSeek-V4-Flash ]]; then
    MODEL_PATH=/data/models/DeepSeek-V4-Flash
  else
    MODEL_PATH="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136"
  fi
fi

SPEC_ARGS=()
if [[ "${SGLANG_DSV4F_DISABLE_MTP:-0}" != "1" ]]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK:-1}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
  )
fi

"$VENV_DIR/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name DeepSeek-V4-Flash \
  --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" \
  --attention-backend dsv4 \
  --moe-runner-backend b12x \
  --fp8-gemm-backend b12x \
  --fp4-gemm-backend b12x \
  --virtual-tp-sharding "$VIRTUAL_TP_SHARDING" \
  --moe-a2a-backend none \
  --reasoning-parser deepseek-v4 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.9 \
  --ep-size 1 \
  --enable-pcie-oneshot-allreduce \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-4}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-4}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  "${SPEC_ARGS[@]}" \
  "$@"
