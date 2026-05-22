import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Wikitext-2 perplexity with a Transformers logits forward pass."
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
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def dtype_arg(value: str) -> str | torch.dtype:
    aliases = {
        "auto": "auto",
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported dtype: {value}")
    return aliases[value]


def format_model_ref(model: str) -> str:
    path = Path(model)
    if path.exists():
        return str(path.resolve())
    return model


def load_wikitext_tokens(args: argparse.Namespace, tokenizer: Any) -> torch.Tensor:
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    text = "\n\n".join(dataset["text"])
    tokenized = tokenizer(text, return_tensors="pt", truncation=False)
    return tokenized.input_ids


def make_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "label",
        "backend",
        "task",
        "model",
        "tokenizer_model",
        "dataset_name",
        "dataset_config",
        "split",
        "seq_len",
        "max_samples",
        "evaluated_chunks",
        "evaluated_tokens",
        "perplexity",
        "mean_nll",
        "dtype",
        "device_map",
    ]
    return {key: summary.get(key) for key in keys}


def upsert_summary_csv(summary_path: Path, row: dict[str, Any]) -> None:
    rows = []
    if summary_path.exists():
        with summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(existing) for existing in reader]

    rows = [existing for existing in rows if existing.get("label") != row.get("label")]
    rows.append(row)

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if args.seq_len < 2:
        raise ValueError("--seq-len must be at least 2")

    tokenizer_ref = args.tokenizer_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=True)
    data = load_wikitext_tokens(args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_arg(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    device = next(model.parameters()).device
    n_samples = data.numel() // args.seq_len
    if args.max_samples is not None:
        n_samples = min(n_samples, args.max_samples)
    if n_samples < 1:
        raise ValueError("No full chunks were created")

    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    per_chunk = []

    with tqdm(range(n_samples), desc="Perplexity") as progress:
        for i in progress:
            start = i * args.seq_len
            end = (i + 1) * args.seq_len
            batch = data[:, start:end].to(device)

            with torch.no_grad():
                logits = model(batch).logits

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nll = loss.float() * args.seq_len
            nlls.append(nll.detach().cpu())

            curr_nll = torch.stack(nlls).sum()
            curr_ppl = torch.exp(curr_nll / ((i + 1) * args.seq_len)).item()
            per_chunk.append(
                {
                    "chunk_index": i,
                    "tokens": args.seq_len,
                    "nll": float(nll.detach().cpu()),
                    "ppl": math.exp(float(loss.detach().cpu())),
                }
            )
            progress.set_description(f"Perplexity {curr_ppl:.3f}")

    total_nll = torch.stack(nlls).sum()
    total_tokens = n_samples * args.seq_len
    ppl = torch.exp(total_nll / total_tokens).item()

    return {
        "backend": "transformers",
        "task": "perplexity",
        "label": args.label,
        "model": format_model_ref(args.model),
        "tokenizer_model": format_model_ref(tokenizer_ref),
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seq_len": args.seq_len,
        "max_samples": args.max_samples,
        "evaluated_chunks": n_samples,
        "evaluated_tokens": total_tokens,
        "total_nll": float(total_nll),
        "mean_nll": float(total_nll / total_tokens),
        "perplexity": ppl,
        "avg_chunk_ppl": mean(item["ppl"] for item in per_chunk),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "per_chunk_count": len(per_chunk),
    }


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = run_eval(args)
    output_path = result_dir / f"{args.label}.json"
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    upsert_summary_csv(result_dir / "summary_transformers.csv", make_summary_row(summary))
    print(json.dumps(summary, indent=2))
    print(f"Saved Transformers perplexity summary to {output_path}")


if __name__ == "__main__":
    main()
