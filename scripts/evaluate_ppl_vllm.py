import argparse
import asyncio
import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

python_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ.setdefault("VLLM_USE_V1", "0")

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: datasets. Install it in the vLLM env with "
        "`python -m pip install datasets`."
    ) from exc

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind

from benchmark_vllm import (
    MemoryTracker,
    collect_vllm_capacity_info,
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
        description="Evaluate perplexity on Wikitext-2 using vLLM prompt logprobs."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer-model",
        help="Tokenizer used to build the evaluation token stream. Defaults to --model.",
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Token stride between chunks. Use seq-len for non-overlapping chunks.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional maximum number of chunks to evaluate.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int)
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


def load_wikitext_tokens(args: argparse.Namespace, tokenizer: Any) -> list[int]:
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    text = "\n\n".join(dataset["text"])
    tokenized = tokenizer(text, truncation=False)
    token_ids = tokenized["input_ids"]
    if len(token_ids) < 2:
        raise ValueError("Dataset produced fewer than two tokens")
    return token_ids


def make_chunks(
    token_ids: list[int],
    seq_len: int,
    stride: int,
    max_samples: int | None,
) -> list[list[int]]:
    if seq_len < 2:
        raise ValueError("--seq-len must be at least 2")
    if stride < 1:
        raise ValueError("--stride must be at least 1")

    chunks = []
    for start in range(0, len(token_ids) - 1, stride):
        chunk = token_ids[start : start + seq_len]
        if len(chunk) < seq_len:
            break
        chunks.append(chunk)
        if max_samples is not None and len(chunks) >= max_samples:
            break
    return chunks


def extract_prompt_logprob(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    logprob_obj = entry.get(token_id)
    if logprob_obj is None:
        return None
    value = getattr(logprob_obj, "logprob", logprob_obj)
    return float(value)


async def evaluate_chunk(
    engine: AsyncLLMEngine,
    chunk: list[int],
    request_id: str,
) -> dict[str, Any]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        top_p=1.0,
        prompt_logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenize=False,
    )

    start = time.perf_counter()
    final_output = None
    async for output in engine.generate(
        request_id=request_id,
        prompt=TokensPrompt(prompt_token_ids=chunk),
        sampling_params=sampling_params,
    ):
        final_output = output
    end = time.perf_counter()

    if final_output is None:
        raise RuntimeError(f"No output produced for {request_id}")

    prompt_logprobs = final_output.prompt_logprobs
    nll = 0.0
    counted_tokens = 0
    missing_logprobs = 0
    for idx, token_id in enumerate(chunk):
        if idx == 0:
            continue
        logprob = extract_prompt_logprob(prompt_logprobs[idx], token_id)
        if logprob is None:
            missing_logprobs += 1
            continue
        nll -= logprob
        counted_tokens += 1

    if counted_tokens == 0:
        raise RuntimeError(f"No prompt logprobs found for {request_id}")

    return {
        "request_id": request_id,
        "tokens": counted_tokens,
        "nll": nll,
        "ppl": math.exp(nll / counted_tokens),
        "missing_logprobs": missing_logprobs,
        "latency_ms": (end - start) * 1000.0,
    }


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.seq_len > args.max_model_len:
        raise ValueError("--seq-len cannot exceed --max-model-len")

    temp_dirs: list[tempfile.TemporaryDirectory] = []
    model_for_vllm = prepare_vllm_model_path(args.model, temp_dirs)
    tokenizer_ref = args.tokenizer_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=True)
    token_ids = load_wikitext_tokens(args, tokenizer)
    chunks = make_chunks(token_ids, args.seq_len, args.stride, args.max_samples)
    if not chunks:
        raise ValueError("No evaluation chunks were created")

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
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    try:
        post_load_memory = tracker.current_mib()
        capacity_info = collect_vllm_capacity_info(engine)
        tracker.reset()
        per_chunk = []
        for idx, chunk in enumerate(chunks):
            per_chunk.append(
                await evaluate_chunk(
                    engine=engine,
                    chunk=chunk,
                    request_id=f"{args.label}-{idx}",
                )
            )
        measured_peak_memory = tracker.peak_mib()
    finally:
        engine.shutdown_background_loop()
        tracker.stop()
        final_memory = tracker.current_mib()
        if pynvml is not None:
            pynvml.nvmlShutdown()
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    total_nll = sum(item["nll"] for item in per_chunk)
    total_tokens = sum(item["tokens"] for item in per_chunk)
    total_missing = sum(item["missing_logprobs"] for item in per_chunk)
    ppl = math.exp(total_nll / total_tokens)

    return {
        "backend": "vllm",
        "task": "perplexity",
        "label": args.label,
        "model": format_model_ref(args.model),
        "tokenizer_model": format_model_ref(tokenizer_ref),
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_samples": args.max_samples,
        "evaluated_chunks": len(per_chunk),
        "evaluated_tokens": total_tokens,
        "total_nll": total_nll,
        "mean_nll": total_nll / total_tokens,
        "perplexity": ppl,
        "avg_chunk_ppl": mean(item["ppl"] for item in per_chunk),
        "missing_logprobs": total_missing,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "baseline_gpu_memory_mib": baseline_memory,
        "post_load_gpu_memory_mib": post_load_memory,
        "measured_peak_gpu_memory_mib": measured_peak_memory,
        "final_gpu_memory_mib": final_memory,
        **capacity_info,
        "per_chunk": per_chunk,
    }


def append_summary_csv(result_dir: Path, summary: dict[str, Any]) -> None:
    csv_path = result_dir / "summary.csv"
    fieldnames = [
        "label",
        "backend",
        "task",
        "model",
        "tokenizer_model",
        "dataset_name",
        "dataset_config",
        "split",
        "seq_len",
        "stride",
        "max_samples",
        "evaluated_chunks",
        "evaluated_tokens",
        "perplexity",
        "mean_nll",
        "missing_logprobs",
        "kv_cache_dtype",
        "dtype",
        "gpu_memory_utilization",
        "post_load_gpu_memory_mib",
        "measured_peak_gpu_memory_mib",
        "vllm_model_weight_memory_gib",
        "vllm_available_kv_cache_memory_gib",
        "vllm_cache_block_size_bytes",
        "vllm_gpu_blocks",
        "vllm_max_concurrency",
    ]
    row = {name: summary.get(name) for name in fieldnames}
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == fieldnames:
                rows = [
                    existing_row
                    for existing_row in reader
                    if existing_row.get("label") != summary["label"]
                ]

    rows.append(row)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(run_eval(args))
    json_path = result_dir / f"{args.label}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    append_summary_csv(result_dir, summary)

    console_summary = {key: value for key, value in summary.items() if key != "per_chunk"}
    console_summary["per_chunk_count"] = len(summary["per_chunk"])
    print(json.dumps(console_summary, indent=2))
    print(f"Saved perplexity summary to {json_path}")


if __name__ == "__main__":
    main()
