import argparse
import asyncio
import csv
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

# vLLM may JIT/compile kernels via subprocesses that call tools such as
# `ninja`. When this script is invoked by absolute Python path without
# activating the env, make the env's bin directory visible to subprocesses.
python_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ.setdefault("VLLM_USE_V1", "0")

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


DEFAULT_PROMPTS = [
    "Summarize why quantization is useful for large language model deployment.",
    "Write a short explanation of activation-aware weight quantization for a machine learning engineer.",
    "List three tradeoffs between lower memory usage and generation quality in quantized inference.",
    "Explain the difference between model weights and the KV cache during autoregressive decoding.",
]

BYTES_PER_GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a local vLLM checkpoint for TTFT, ms/token, and GPU memory."
    )
    parser.add_argument("--model", required=True, help="Path to the local model directory.")
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory where benchmark json/csv results will be saved.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Human-friendly label for this benchmark run.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length for vLLM.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization target.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use with tensor parallelism.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Optional vLLM max number of active sequences.",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=0,
        help="Number of prompts to run before recording latency measurements.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of measured passes over the prompt set.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Model dtype to pass to vLLM.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
        help="KV cache dtype to pass to vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for easier debugging.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional text file containing one prompt per line.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Store full prompts and generated text in the JSON result.",
    )
    return parser.parse_args()


def load_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def format_model_ref(model: str) -> str:
    model_path = Path(model)
    if model_path.exists() or model_path.is_absolute() or model.startswith("."):
        return str(model_path.resolve())
    return model


def remove_unsupported_quant_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_unsupported_quant_keys(item)
            for key, item in value.items()
            if key not in {"scale_dtype", "zp_dtype"}
        }
    if isinstance(value, list):
        return [remove_unsupported_quant_keys(item) for item in value]
    return value


def prepare_vllm_model_path(model: str, temp_dirs: list[tempfile.TemporaryDirectory]) -> str:
    model_path = Path(model)
    config_path = model_path / "config.json"
    if not model_path.is_dir() or not config_path.exists():
        return model

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    quantization_config = config.get("quantization_config")
    if quantization_config is None:
        return str(model_path)

    sanitized_config = remove_unsupported_quant_keys(config)
    if sanitized_config == config:
        return str(model_path)

    temp_dir = tempfile.TemporaryDirectory(prefix=f"{model_path.name}-vllm-")
    temp_dirs.append(temp_dir)
    compat_path = Path(temp_dir.name)

    for item in model_path.iterdir():
        target = compat_path / item.name
        if item.name == "config.json":
            continue
        target.symlink_to(item.resolve(), target_is_directory=item.is_dir())

    with open(compat_path / "config.json", "w", encoding="utf-8") as handle:
        json.dump(sanitized_config, handle, indent=2)
        handle.write("\n")

    return str(compat_path)


def get_visible_gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        if first:
            return int(first)
    return 0


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return value / BYTES_PER_GIB


def collect_vllm_capacity_info(engine: AsyncLLMEngine) -> dict[str, Any]:
    raw_engine = getattr(engine, "engine", None)
    if raw_engine is None:
        return {}

    cache_config = getattr(raw_engine, "cache_config", None)
    model_config = getattr(raw_engine, "model_config", None)
    model_executor = getattr(raw_engine, "model_executor", None)

    num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    num_cpu_blocks = getattr(cache_config, "num_cpu_blocks", None)
    block_size = getattr(cache_config, "block_size", None)
    max_model_len = getattr(model_config, "max_model_len", None)
    max_concurrency = None
    if num_gpu_blocks is not None and block_size is not None and max_model_len:
        max_concurrency = num_gpu_blocks * block_size / max_model_len

    worker_wrapper = getattr(model_executor, "driver_worker", None)
    worker = getattr(worker_wrapper, "worker", worker_wrapper)
    model_runner = getattr(worker, "model_runner", None)

    cache_block_size_bytes = None
    if worker is not None and hasattr(worker, "get_cache_block_size_bytes"):
        cache_block_size_bytes = worker.get_cache_block_size_bytes()

    allocated_kv_cache_memory_bytes = None
    if num_gpu_blocks is not None and cache_block_size_bytes is not None:
        allocated_kv_cache_memory_bytes = num_gpu_blocks * cache_block_size_bytes

    requested_memory_bytes = getattr(worker, "requested_memory", None)
    available_kv_cache_memory_bytes = getattr(
        worker, "available_kv_cache_memory", None
    )
    model_weight_memory_bytes = getattr(model_runner, "model_memory_usage", None)
    peak_activation_memory_bytes = getattr(worker, "peak_activation_memory", None)
    non_torch_memory_bytes = getattr(worker, "non_torch_memory", None)

    profiled_non_kv_memory_bytes = None
    parts = [
        model_weight_memory_bytes,
        peak_activation_memory_bytes,
        non_torch_memory_bytes,
    ]
    if all(part is not None for part in parts):
        profiled_non_kv_memory_bytes = sum(parts)

    return {
        "vllm_gpu_memory_utilization": getattr(
            cache_config, "gpu_memory_utilization", None
        ),
        "vllm_requested_memory_bytes": requested_memory_bytes,
        "vllm_requested_memory_gib": bytes_to_gib(requested_memory_bytes),
        "vllm_model_weight_memory_bytes": model_weight_memory_bytes,
        "vllm_model_weight_memory_gib": bytes_to_gib(model_weight_memory_bytes),
        "vllm_peak_activation_memory_bytes": peak_activation_memory_bytes,
        "vllm_peak_activation_memory_gib": bytes_to_gib(peak_activation_memory_bytes),
        "vllm_non_torch_memory_bytes": non_torch_memory_bytes,
        "vllm_non_torch_memory_gib": bytes_to_gib(non_torch_memory_bytes),
        "vllm_profiled_non_kv_memory_bytes": profiled_non_kv_memory_bytes,
        "vllm_profiled_non_kv_memory_gib": bytes_to_gib(
            profiled_non_kv_memory_bytes
        ),
        "vllm_available_kv_cache_memory_bytes": available_kv_cache_memory_bytes,
        "vllm_available_kv_cache_memory_gib": bytes_to_gib(
            available_kv_cache_memory_bytes
        ),
        "vllm_cache_block_size_bytes": cache_block_size_bytes,
        "vllm_allocated_kv_cache_memory_bytes": allocated_kv_cache_memory_bytes,
        "vllm_allocated_kv_cache_memory_gib": bytes_to_gib(
            allocated_kv_cache_memory_bytes
        ),
        "vllm_gpu_blocks": num_gpu_blocks,
        "vllm_cpu_blocks": num_cpu_blocks,
        "vllm_block_size_tokens": block_size,
        "vllm_max_concurrency": max_concurrency,
    }


class MemoryTracker:
    def __init__(self, gpu_index: int, interval_s: float = 0.1):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._running = False
        self._thread = None
        self.samples_mib: list[float] = []
        self._lock = threading.Lock()

    def current_mib(self) -> float | None:
        if pynvml is None:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)

    def start(self) -> None:
        if pynvml is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = False
        self._thread.join(timeout=2.0)

    def peak_mib(self) -> float | None:
        with self._lock:
            return max(self.samples_mib) if self.samples_mib else None

    def reset(self) -> None:
        with self._lock:
            self.samples_mib.clear()

    def _poll(self) -> None:
        while self._running:
            value = self.current_mib()
            if value is not None:
                with self._lock:
                    self.samples_mib.append(value)
            time.sleep(self.interval_s)


def count_completion_tokens(completion: Any) -> int:
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    text = getattr(completion, "text", "")
    return 1 if text else 0


async def benchmark_prompt(
    engine: AsyncLLMEngine,
    prompt: str,
    request_id: str,
    max_tokens: int,
    include_text: bool,
) -> dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        top_p=1.0,
        seed=42,
        output_kind=RequestOutputKind.DELTA,
    )

    start = time.perf_counter()
    first_token_at = None
    output_tokens = 0
    generated_text_parts: list[str] = []

    async for output in engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
    ):
        for completion in output.outputs:
            new_tokens = count_completion_tokens(completion)
            if new_tokens > 0 and first_token_at is None:
                first_token_at = time.perf_counter()
            output_tokens += new_tokens
            text = getattr(completion, "text", "")
            if text:
                generated_text_parts.append(text)

    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end

    total_latency_ms = (end - start) * 1000.0
    ttft_ms = (first_token_at - start) * 1000.0
    ms_per_token = total_latency_ms / output_tokens if output_tokens else None
    decode_ms_per_token = None
    if output_tokens > 1:
        decode_ms_per_token = ((end - first_token_at) * 1000.0) / (output_tokens - 1)

    generated_text = "".join(generated_text_parts)
    result = {
        "prompt_preview": prompt[:240],
        "generated_text_preview": generated_text[:240],
        "output_tokens": output_tokens,
        "total_latency_ms": total_latency_ms,
        "ttft_ms": ttft_ms,
        "ms_per_token": ms_per_token,
        "decode_ms_per_token": decode_ms_per_token,
    }
    if include_text:
        result["prompt"] = prompt
        result["generated_text"] = generated_text
    return result


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompt_file)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmup_prompts < 0:
        raise ValueError("--warmup-prompts cannot be negative")

    gpu_index = get_visible_gpu_index()
    temp_dirs: list[tempfile.TemporaryDirectory] = []
    model_for_vllm = prepare_vllm_model_path(args.model, temp_dirs)

    if pynvml is not None:
        pynvml.nvmlInit()

    tracker = MemoryTracker(gpu_index=gpu_index)
    baseline_memory = tracker.current_mib()
    tracker.start()

    engine_kwargs = {
        "model": model_for_vllm,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs

    engine_args = AsyncEngineArgs(
        **engine_kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    try:
        post_load_memory = tracker.current_mib()
        capacity_info = collect_vllm_capacity_info(engine)
        per_prompt_results = []
        for idx in range(args.warmup_prompts):
            prompt = prompts[idx % len(prompts)]
            await benchmark_prompt(
                engine=engine,
                prompt=prompt,
                request_id=f"{args.label}-warmup-{idx}",
                max_tokens=args.max_tokens,
                include_text=False,
            )

        load_peak_memory = tracker.peak_mib()
        tracker.reset()

        for repeat_idx in range(args.repeats):
            for prompt_idx, prompt in enumerate(prompts):
                result = await benchmark_prompt(
                    engine=engine,
                    prompt=prompt,
                    request_id=f"{args.label}-r{repeat_idx}-p{prompt_idx}",
                    max_tokens=args.max_tokens,
                    include_text=args.include_text,
                )
                result["repeat"] = repeat_idx
                result["prompt_index"] = prompt_idx
                per_prompt_results.append(result)
        measured_peak_memory = tracker.peak_mib()
    finally:
        engine.shutdown_background_loop()
        tracker.stop()
        final_memory = tracker.current_mib()
        if pynvml is not None:
            pynvml.nvmlShutdown()
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    peak_candidates = [
        value for value in [load_peak_memory, measured_peak_memory] if value is not None
    ]
    peak_memory = max(peak_candidates) if peak_candidates else None
    avg_ms_per_token = mean(
        result["ms_per_token"]
        for result in per_prompt_results
        if result["ms_per_token"] is not None
    )
    avg_ttft_ms = mean(result["ttft_ms"] for result in per_prompt_results)
    avg_decode_ms_per_token = mean(
        result["decode_ms_per_token"]
        for result in per_prompt_results
        if result["decode_ms_per_token"] is not None
    )

    return {
        "backend": "vllm",
        "label": args.label,
        "model": format_model_ref(args.model),
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "max_num_seqs": args.max_num_seqs,
        "base_prompt_count": len(prompts),
        "prompt_count": len(per_prompt_results),
        "warmup_prompts": args.warmup_prompts,
        "repeats": args.repeats,
        "include_text": args.include_text,
        "avg_ms_per_token": avg_ms_per_token,
        "avg_ttft_ms": avg_ttft_ms,
        "avg_decode_ms_per_token": avg_decode_ms_per_token,
        "baseline_gpu_memory_mib": baseline_memory,
        "post_load_gpu_memory_mib": post_load_memory,
        "load_peak_gpu_memory_mib": load_peak_memory,
        "measured_peak_gpu_memory_mib": measured_peak_memory,
        "peak_gpu_memory_mib": peak_memory,
        "final_gpu_memory_mib": final_memory,
        **capacity_info,
        "per_prompt": per_prompt_results,
    }


def append_summary_csv(result_dir: Path, summary: dict[str, Any]) -> None:
    csv_path = result_dir / "summary.csv"
    fieldnames = [
        "label",
        "backend",
        "model",
        "kv_cache_dtype",
        "tensor_parallel_size",
        "dtype",
        "max_model_len",
        "max_tokens",
        "max_num_seqs",
        "base_prompt_count",
        "prompt_count",
        "warmup_prompts",
        "repeats",
        "include_text",
        "avg_ms_per_token",
        "avg_ttft_ms",
        "avg_decode_ms_per_token",
        "baseline_gpu_memory_mib",
        "post_load_gpu_memory_mib",
        "load_peak_gpu_memory_mib",
        "measured_peak_gpu_memory_mib",
        "peak_gpu_memory_mib",
        "final_gpu_memory_mib",
        "vllm_gpu_memory_utilization",
        "vllm_requested_memory_gib",
        "vllm_profiled_non_kv_memory_gib",
        "vllm_available_kv_cache_memory_gib",
        "vllm_allocated_kv_cache_memory_gib",
        "vllm_cache_block_size_bytes",
        "vllm_gpu_blocks",
        "vllm_cpu_blocks",
        "vllm_block_size_tokens",
        "vllm_max_concurrency",
    ]
    write_header = not csv_path.exists()
    mode = "a"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, None)
        write_header = existing_header != fieldnames
        if write_header:
            mode = "w"

    with open(csv_path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: summary.get(name) for name in fieldnames})


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(run_benchmark(args))
    json_path = result_dir / f"{args.label}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_summary_csv(result_dir, summary)

    console_summary = {key: value for key, value in summary.items() if key != "per_prompt"}
    console_summary["per_prompt_count"] = len(summary["per_prompt"])
    print(json.dumps(console_summary, indent=2))
    print(f"Saved benchmark summary to {json_path}")


if __name__ == "__main__":
    main()
