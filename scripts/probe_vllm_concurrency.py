import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

python_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ.setdefault("VLLM_USE_V1", "0")

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind

from benchmark_vllm import (
    MemoryTracker,
    count_completion_tokens,
    format_model_ref,
    get_visible_gpu_index,
    prepare_vllm_model_path,
)

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether vLLM can sustain long-context concurrent requests."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--concurrency-levels",
        default="8,10,11,12",
        help="Comma-separated batch sizes to probe.",
    )
    parser.add_argument("--target-input-tokens", type=int, default=7600)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--swap-space", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
    )
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def parse_levels(raw: str) -> list[int]:
    levels = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not levels or any(level < 1 for level in levels):
        raise ValueError("--concurrency-levels must contain positive integers")
    return levels


def make_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    prefix = (
        "You are stress testing vLLM KV cache concurrency for Llama 3. "
        "Keep the following context resident and answer briefly at the end. "
    )
    unit = (
        "KV cache capacity depends on layers, KV heads, head size, block size, "
        "dtype size, prompt length, generated tokens, and active sequence count. "
    )

    prompt = prefix
    count = len(tokenizer.encode(prompt, add_special_tokens=False))
    unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))
    if unit_tokens <= 0:
        raise ValueError("Tokenizer produced no tokens for the prompt unit")

    repeats = max((target_tokens - count) // unit_tokens, 0)
    prompt += unit * repeats

    while len(tokenizer.encode(prompt + unit, add_special_tokens=False)) <= target_tokens:
        prompt += unit

    prompt += "Question: state whether the context was processed successfully."
    token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, token_count


async def generate_once(
    engine: AsyncLLMEngine,
    prompt: str,
    request_id: str,
    max_tokens: int,
) -> dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        top_p=1.0,
        seed=42,
        output_kind=RequestOutputKind.DELTA,
    )
    start = time.perf_counter()
    output_tokens = 0
    async for output in engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
    ):
        for completion in output.outputs:
            output_tokens += count_completion_tokens(completion)
    end = time.perf_counter()
    return {
        "request_id": request_id,
        "ok": True,
        "output_tokens": output_tokens,
        "latency_ms": (end - start) * 1000.0,
    }


async def run_level(
    engine: AsyncLLMEngine,
    label: str,
    level: int,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    tasks = [
        generate_once(engine, prompt, f"{label}-c{level}-r{idx}", max_tokens)
        for idx in range(level)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    end = time.perf_counter()

    request_results = []
    failures = 0
    for idx, result in enumerate(raw_results):
        if isinstance(result, Exception):
            failures += 1
            request_results.append(
                {
                    "request_id": f"{label}-c{level}-r{idx}",
                    "ok": False,
                    "error_type": type(result).__name__,
                    "error": str(result),
                }
            )
        else:
            request_results.append(result)

    return {
        "concurrency": level,
        "ok": failures == 0,
        "failures": failures,
        "wall_time_ms": (end - start) * 1000.0,
        "requests": request_results,
    }


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    levels = parse_levels(args.concurrency_levels)
    temp_dirs: list[tempfile.TemporaryDirectory] = []
    model_for_vllm = prepare_vllm_model_path(args.model, temp_dirs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt, prompt_tokens = make_prompt(tokenizer, args.target_input_tokens)
    total_tokens_per_request = prompt_tokens + args.max_tokens
    if total_tokens_per_request > args.max_model_len:
        raise ValueError(
            "target prompt plus max tokens exceeds max model length: "
            f"{prompt_tokens} + {args.max_tokens} > {args.max_model_len}"
        )

    if pynvml is not None:
        pynvml.nvmlInit()
    tracker = MemoryTracker(gpu_index=get_visible_gpu_index())
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
        "swap_space": args.swap_space,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    try:
        post_load_memory = tracker.current_mib()
        level_results = []
        for level in levels:
            tracker.reset()
            result = await run_level(
                engine=engine,
                label=args.label,
                level=level,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
            result["peak_gpu_memory_mib"] = tracker.peak_mib()
            level_results.append(result)
    finally:
        engine.shutdown_background_loop()
        tracker.stop()
        final_memory = tracker.current_mib()
        if pynvml is not None:
            pynvml.nvmlShutdown()
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    return {
        "backend": "vllm",
        "label": args.label,
        "model": format_model_ref(args.model),
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "swap_space": args.swap_space,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens_per_request": total_tokens_per_request,
        "concurrency_levels": levels,
        "baseline_gpu_memory_mib": baseline_memory,
        "post_load_gpu_memory_mib": post_load_memory,
        "final_gpu_memory_mib": final_memory,
        "levels": level_results,
    }


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(run_probe(args))
    json_path = result_dir / f"{args.label}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved concurrency probe summary to {json_path}")


if __name__ == "__main__":
    main()
