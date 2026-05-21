#!/bin/bash
#SBATCH -J llama3-awq-vllm
#SBATCH -o ./log/llama3_awq_vllm.out
#SBATCH --gres=gpu:1 #Number of GPU devices to use [0-2]
#SBATCH --nodelist=leon05 #YOUR NODE OF PREFERENCE


set -euo pipefail

# Set Hugging Face token

#export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
ACTION="${1:-all}"
RAW_MODEL_ID="${RAW_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"

COMPRESS_PY="${COMPRESS_PY:-$REPO_DIR/.conda/llm-compressor/bin/python}"
VLLM_PY="${VLLM_PY:-$REPO_DIR/.conda/vllm/bin/python}"
COMPRESS_BIN="$(dirname "$COMPRESS_PY")"
VLLM_BIN="$(dirname "$VLLM_PY")"

MODEL_ROOT="$REPO_DIR/artifacts/models"
RESULT_ROOT="$REPO_DIR/artifacts/results"
LOG_ROOT="$REPO_DIR/log"
CACHE_ROOT="$REPO_DIR/artifacts/cache"
PROMPT_ROOT="$REPO_DIR/artifacts/prompts"
HF_CACHE_ROOT="$CACHE_ROOT/huggingface"
DATASET_CACHE_ROOT="$CACHE_ROOT/datasets"

AWQ_MODEL_DIR="${AWQ_MODEL_DIR:-$MODEL_ROOT/llama-3-8b-instruct-awq-w4a16}"
AWQ_FP8_KV_MODEL_DIR="${AWQ_FP8_KV_MODEL_DIR:-$MODEL_ROOT/llama-3-8b-instruct-awq-w4a16-fp8-kv}"
BENCHMARK_RESULT_DIR="${BENCHMARK_RESULT_DIR:-$RESULT_ROOT/llama-3-8b-instruct}"
KV_BENCHMARK_RESULT_DIR="${KV_BENCHMARK_RESULT_DIR:-$RESULT_ROOT/llama-3-8b-instruct-kv}"
CONCURRENCY_RESULT_DIR="${CONCURRENCY_RESULT_DIR:-$RESULT_ROOT/llama-3-8b-instruct-concurrency}"
KV_PROMPT_FILE="${KV_PROMPT_FILE:-$PROMPT_ROOT/kv_long_prompts.txt}"

NUM_CALIBRATION_SAMPLES="${NUM_CALIBRATION_SAMPLES:-256}"
CALIBRATION_MAX_SEQ_LEN="${CALIBRATION_MAX_SEQ_LEN:-512}"
SKIP_SANITY_GENERATION="${SKIP_SANITY_GENERATION:-0}"

BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-128}"
BENCH_MAX_MODEL_LEN="${BENCH_MAX_MODEL_LEN:-2048}"
BENCH_GPU_MEMORY_UTILIZATION="${BENCH_GPU_MEMORY_UTILIZATION:-0.9}"
BENCH_MAX_NUM_SEQS="${BENCH_MAX_NUM_SEQS:-}"
BENCH_WARMUP_PROMPTS="${BENCH_WARMUP_PROMPTS:-1}"
BENCH_REPEATS="${BENCH_REPEATS:-1}"
PROBE_TARGET_INPUT_TOKENS="${PROBE_TARGET_INPUT_TOKENS:-7600}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-64}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
PROMPT_FILE="${PROMPT_FILE:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

mkdir -p "$MODEL_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$PROMPT_ROOT" "$AWQ_MODEL_DIR" \
    "$AWQ_FP8_KV_MODEL_DIR" "$BENCHMARK_RESULT_DIR" "$HF_CACHE_ROOT" \
    "$DATASET_CACHE_ROOT" "$KV_BENCHMARK_RESULT_DIR" "$CONCURRENCY_RESULT_DIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_TOKEN="$HF_TOKEN"
elif [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
    export HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_TOKEN"
fi

export HF_HOME="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"
export HF_DATASETS_CACHE="$DATASET_CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$VLLM_BIN:$COMPRESS_BIN:$PATH"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

cd "$REPO_DIR"

require_token() {
    if [[ -z "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ]]; then
        echo "HUGGINGFACE_TOKEN or HF_TOKEN must be set for gated Llama models."
        exit 1
    fi
}

common_benchmark_args() {
    local label="$1"
    local model="$2"
    shift 2

    local args=(
        scripts/benchmark_vllm.py
        --model "$model"
        --label "$label"
        --result-dir "$BENCHMARK_RESULT_DIR"
        --max-model-len "$BENCH_MAX_MODEL_LEN"
        --max-tokens "$BENCH_MAX_TOKENS"
        --gpu-memory-utilization "$BENCH_GPU_MEMORY_UTILIZATION"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --dtype "$VLLM_DTYPE"
        --warmup-prompts "$BENCH_WARMUP_PROMPTS"
        --repeats "$BENCH_REPEATS"
    )

    if [[ -n "$BENCH_MAX_NUM_SEQS" ]]; then
        args+=(--max-num-seqs "$BENCH_MAX_NUM_SEQS")
    fi
    if [[ -n "$PROMPT_FILE" ]]; then
        args+=(--prompt-file "$PROMPT_FILE")
    fi
    if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
        args+=(--enforce-eager)
    fi

    "$VLLM_PY" "${args[@]}" "$@"
}

quantize_awq() {
    require_token
    local args=(
        scripts/quantize_llama_awq.py
        --model-id "$RAW_MODEL_ID"
        --output-dir "$AWQ_MODEL_DIR"
        --num-calibration-samples "$NUM_CALIBRATION_SAMPLES"
        --max-seq-length "$CALIBRATION_MAX_SEQ_LEN"
    )
    if [[ "$SKIP_SANITY_GENERATION" == "1" || "$SKIP_SANITY_GENERATION" == "true" ]]; then
        args+=(--skip-sanity-generation)
    fi
    "$COMPRESS_PY" "${args[@]}"
}

quantize_awq_fp8_kv() {
    require_token
    local args=(
        scripts/quantize_llama_awq.py
        --model-id "$RAW_MODEL_ID"
        --with-fp8-kv
        --output-dir "$AWQ_FP8_KV_MODEL_DIR"
        --num-calibration-samples "$NUM_CALIBRATION_SAMPLES"
        --max-seq-length "$CALIBRATION_MAX_SEQ_LEN"
    )
    if [[ "$SKIP_SANITY_GENERATION" == "1" || "$SKIP_SANITY_GENERATION" == "true" ]]; then
        args+=(--skip-sanity-generation)
    fi
    "$COMPRESS_PY" "${args[@]}"
}

benchmark_raw() {
    require_token
    common_benchmark_args "llama3_raw" "$RAW_MODEL_ID"
}

benchmark_awq() {
    common_benchmark_args "llama3_awq_w4a16" "$AWQ_MODEL_DIR"
}

benchmark_awq_fp8_kv() {
    common_benchmark_args "llama3_awq_w4a16_fp8_kv" "$AWQ_FP8_KV_MODEL_DIR" \
        --kv-cache-dtype fp8
}

ensure_kv_prompt_file() {
    if [[ -s "$KV_PROMPT_FILE" ]]; then
        return
    fi

    "$COMPRESS_PY" - "$KV_PROMPT_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)

base_sections = [
    "System design note: compare raw fp16 inference, AWQ W4A16 weights, and AWQ with fp8 KV cache for a production chat service.",
    "Workload assumptions: prompts are long, requests include retrieved documents, generation continues for hundreds of tokens, and peak memory matters more than initial model load.",
    "Measurement priorities: steady-state decode throughput, time to first token after warmup, post-load GPU allocation, measured peak GPU allocation during generation, and repeatability across prompts.",
    "Analysis request: explain which memory components are affected by weight quantization, which are affected by KV cache quantization, and why short prompts hide KV benefits.",
]

prompts = []
for idx in range(4):
    paragraphs = []
    for repeat in range(80):
        section = base_sections[(idx + repeat) % len(base_sections)]
        paragraphs.append(
            f"{section} Scenario {idx + 1}.{repeat + 1}: include concrete reasoning about "
            "prefill length, decode length, batch capacity, vLLM block allocation, CUDA graph or eager behavior, "
            "and how a GPU memory snapshot can be distorted by reserved cache space."
        )
    prompts.append(
        "You are evaluating KV cache compression for Llama 3 8B. "
        + " ".join(paragraphs)
        + " Provide a concise technical recommendation with caveats."
    )

path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
PY
}

benchmark_kv() {
    ensure_kv_prompt_file

    local previous_result_dir="$BENCHMARK_RESULT_DIR"
    local previous_prompt_file="$PROMPT_FILE"
    local previous_max_model_len="$BENCH_MAX_MODEL_LEN"
    local previous_max_tokens="$BENCH_MAX_TOKENS"
    local previous_gpu_memory_utilization="$BENCH_GPU_MEMORY_UTILIZATION"
    local previous_warmup_prompts="$BENCH_WARMUP_PROMPTS"
    local previous_repeats="$BENCH_REPEATS"

    BENCHMARK_RESULT_DIR="$KV_BENCHMARK_RESULT_DIR"
    PROMPT_FILE="$KV_PROMPT_FILE"
    BENCH_MAX_MODEL_LEN="${KV_BENCH_MAX_MODEL_LEN:-8192}"
    BENCH_MAX_TOKENS="${KV_BENCH_MAX_TOKENS:-1024}"
    BENCH_GPU_MEMORY_UTILIZATION="${KV_BENCH_GPU_MEMORY_UTILIZATION:-0.75}"
    BENCH_WARMUP_PROMPTS="${KV_BENCH_WARMUP_PROMPTS:-1}"
    BENCH_REPEATS="${KV_BENCH_REPEATS:-1}"

    common_benchmark_args "llama3_awq_w4a16_kvstress" "$AWQ_MODEL_DIR"
    common_benchmark_args "llama3_awq_w4a16_fp8_kv_kvstress" "$AWQ_FP8_KV_MODEL_DIR" \
        --kv-cache-dtype fp8

    BENCHMARK_RESULT_DIR="$previous_result_dir"
    PROMPT_FILE="$previous_prompt_file"
    BENCH_MAX_MODEL_LEN="$previous_max_model_len"
    BENCH_MAX_TOKENS="$previous_max_tokens"
    BENCH_GPU_MEMORY_UTILIZATION="$previous_gpu_memory_utilization"
    BENCH_WARMUP_PROMPTS="$previous_warmup_prompts"
    BENCH_REPEATS="$previous_repeats"
}

benchmark_kv_memory() {
    ensure_kv_prompt_file

    local previous_result_dir="$BENCHMARK_RESULT_DIR"
    local previous_prompt_file="$PROMPT_FILE"
    local previous_max_model_len="$BENCH_MAX_MODEL_LEN"
    local previous_max_tokens="$BENCH_MAX_TOKENS"
    local previous_gpu_memory_utilization="$BENCH_GPU_MEMORY_UTILIZATION"
    local previous_warmup_prompts="$BENCH_WARMUP_PROMPTS"
    local previous_repeats="$BENCH_REPEATS"

    BENCHMARK_RESULT_DIR="${KV_MEMORY_BENCHMARK_RESULT_DIR:-$RESULT_ROOT/llama-3-8b-instruct-kv-memory}"
    mkdir -p "$BENCHMARK_RESULT_DIR"
    PROMPT_FILE="$KV_PROMPT_FILE"
    BENCH_MAX_MODEL_LEN="${KV_BENCH_MAX_MODEL_LEN:-8192}"
    BENCH_MAX_TOKENS="${KV_BENCH_MAX_TOKENS:-1024}"
    BENCH_WARMUP_PROMPTS="${KV_BENCH_WARMUP_PROMPTS:-1}"
    BENCH_REPEATS="${KV_BENCH_REPEATS:-1}"

    BENCH_GPU_MEMORY_UTILIZATION="${KV_AWQ_BENCH_GPU_MEMORY_UTILIZATION:-0.75}"
    common_benchmark_args "llama3_awq_w4a16_kvmem" "$AWQ_MODEL_DIR"

    BENCH_GPU_MEMORY_UTILIZATION="${KV_FP8_BENCH_GPU_MEMORY_UTILIZATION:-0.55}"
    common_benchmark_args "llama3_awq_w4a16_fp8_kv_kvmem" "$AWQ_FP8_KV_MODEL_DIR" \
        --kv-cache-dtype fp8

    BENCHMARK_RESULT_DIR="$previous_result_dir"
    PROMPT_FILE="$previous_prompt_file"
    BENCH_MAX_MODEL_LEN="$previous_max_model_len"
    BENCH_MAX_TOKENS="$previous_max_tokens"
    BENCH_GPU_MEMORY_UTILIZATION="$previous_gpu_memory_utilization"
    BENCH_WARMUP_PROMPTS="$previous_warmup_prompts"
    BENCH_REPEATS="$previous_repeats"
}

common_concurrency_probe_args() {
    local label="$1"
    local model="$2"
    local levels="$3"
    shift 3

    local args=(
        scripts/probe_vllm_concurrency.py
        --model "$model"
        --label "$label"
        --result-dir "$CONCURRENCY_RESULT_DIR"
        --concurrency-levels "$levels"
        --target-input-tokens "$PROBE_TARGET_INPUT_TOKENS"
        --max-model-len "${PROBE_MAX_MODEL_LEN:-8192}"
        --max-tokens "$PROBE_MAX_TOKENS"
        --gpu-memory-utilization "$BENCH_GPU_MEMORY_UTILIZATION"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --dtype "$VLLM_DTYPE"
        --swap-space "${PROBE_SWAP_SPACE:-0}"
    )

    if [[ -n "$BENCH_MAX_NUM_SEQS" ]]; then
        args+=(--max-num-seqs "$BENCH_MAX_NUM_SEQS")
    fi
    if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
        args+=(--enforce-eager)
    fi

    "$VLLM_PY" "${args[@]}" "$@"
}

probe_concurrency_awq() {
    BENCH_GPU_MEMORY_UTILIZATION="${PROBE_AWQ_GPU_MEMORY_UTILIZATION:-0.75}"
    common_concurrency_probe_args \
        "llama3_awq_w4a16_concurrency" \
        "$AWQ_MODEL_DIR" \
        "${PROBE_AWQ_LEVELS:-10,11,12}"
}

probe_concurrency_fp8_kv() {
    BENCH_GPU_MEMORY_UTILIZATION="${PROBE_FP8_GPU_MEMORY_UTILIZATION:-0.55}"
    common_concurrency_probe_args \
        "llama3_awq_w4a16_fp8_kv_concurrency" \
        "$AWQ_FP8_KV_MODEL_DIR" \
        "${PROBE_FP8_LEVELS:-12,13,14}" \
        --kv-cache-dtype fp8
}

probe_concurrency() {
    probe_concurrency_awq
    probe_concurrency_fp8_kv
}

check_gpu_mode() {
    echo "repo: $REPO_DIR"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "compress python: $COMPRESS_PY"
    echo "vllm python: $VLLM_PY"
    echo "raw model: $RAW_MODEL_ID"
    echo "awq model dir: $AWQ_MODEL_DIR"
    echo "awq+fp8 kv model dir: $AWQ_FP8_KV_MODEL_DIR"
    echo "benchmark result dir: $BENCHMARK_RESULT_DIR"
    if [[ -n "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ]]; then
        echo "Hugging Face token: set"
    else
        echo "Hugging Face token: missing"
    fi
    nvidia-smi || true
}

case "$ACTION" in
    check)
        check_gpu_mode
        ;;
    quantize)
        quantize_awq
        quantize_awq_fp8_kv
        ;;
    benchmark)
        benchmark_raw
        benchmark_awq
        benchmark_awq_fp8_kv
        ;;
    benchmark_kv)
        benchmark_kv
        ;;
    benchmark_kv_memory)
        benchmark_kv_memory
        ;;
    probe_concurrency)
        probe_concurrency
        ;;
    probe_concurrency_awq)
        probe_concurrency_awq
        ;;
    probe_concurrency_fp8_kv)
        probe_concurrency_fp8_kv
        ;;
    benchmark_raw)
        benchmark_raw
        ;;
    benchmark_awq)
        benchmark_awq
        ;;
    benchmark_awq_fp8_kv)
        benchmark_awq_fp8_kv
        ;;
    all)
        quantize_awq
        quantize_awq_fp8_kv
        benchmark_raw
        benchmark_awq
        benchmark_awq_fp8_kv
        ;;
    *)
        echo "Usage: bash run.sh [check|quantize|benchmark|benchmark_kv|benchmark_kv_memory|probe_concurrency|probe_concurrency_awq|probe_concurrency_fp8_kv|benchmark_raw|benchmark_awq|benchmark_awq_fp8_kv|all]"
        exit 1
        ;;
esac
