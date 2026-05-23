#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
B12X_DIR="${B12X_DIR:-$HOME/projects/b12x}"

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR/python:$B12X_DIR:${PYTHONPATH:-}"

TP_SIZE="${TP_SIZE:-10}"
if [[ "$TP_SIZE" != "9" && "$TP_SIZE" != "10" ]]; then
  echo "serve-dsv4p-virtual-tp.sh expects TP_SIZE=9 or TP_SIZE=10, got TP_SIZE=$TP_SIZE" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$TP_SIZE" == "9" ]]; then
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
  else
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
  fi
fi
export CUDA_VISIBLE_DEVICES

IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#VISIBLE_GPU_LIST[@]}" -ne "$TP_SIZE" ]]; then
  echo "serve-dsv4p-virtual-tp.sh expects $TP_SIZE CUDA_VISIBLE_DEVICES, got '$CUDA_VISIBLE_DEVICES'" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export VIRTUAL_TP_SHARDING="${VIRTUAL_TP_SHARDING:-b12x-padded}"
export VIRTUAL_TP_MOE_ALIGNMENT="${VIRTUAL_TP_MOE_ALIGNMENT:-16}"

if ! [[ "$VIRTUAL_TP_MOE_ALIGNMENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "VIRTUAL_TP_MOE_ALIGNMENT must be a positive integer, got '$VIRTUAL_TP_MOE_ALIGNMENT'" >&2
  exit 1
fi

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

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  if [[ -d /data/models/DeepSeek-V4-Pro ]]; then
    MODEL_PATH=/data/models/DeepSeek-V4-Pro
  else
    MODEL_PATH="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/89d501aed998d33fa4f4702102ec1bb2331e10f6"
  fi
fi

SPEC_ARGS=()
if [[ "${SGLANG_DSV4P_DISABLE_MTP:-${SGLANG_DSV4_DISABLE_MTP:-0}}" != "1" ]]; then
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK:-1}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
  )
fi

"$VENV_DIR/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name DeepSeek-V4-Pro \
  --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" \
  --attention-backend dsv4 \
  --moe-runner-backend b12x \
  --fp8-gemm-backend b12x \
  --fp4-gemm-backend b12x \
  --virtual-tp-sharding "$VIRTUAL_TP_SHARDING" \
  --virtual-tp-moe-alignment "$VIRTUAL_TP_MOE_ALIGNMENT" \
  --moe-a2a-backend none \
  --reasoning-parser deepseek-v4 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.9}" \
  --ep-size 1 \
  --enable-pcie-oneshot-allreduce \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-2}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-2}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  "${SPEC_ARGS[@]}" \
  "$@"
