import argparse
from pathlib import Path

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize Llama 3 8B with AWQ W4A16, optionally with FP8 KV."
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model id to quantize.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the compressed checkpoint will be saved.",
    )
    parser.add_argument(
        "--with-fp8-kv",
        action="store_true",
        help="Also calibrate and save FP8 KV cache scales.",
    )
    parser.add_argument(
        "--dataset-id",
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset id.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train_sft",
        help="Calibration dataset split.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=256,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum calibration sequence length.",
    )
    parser.add_argument(
        "--skip-sanity-generation",
        action="store_true",
        help="Skip the post-quantization sample generation before saving.",
    )
    return parser.parse_args()


def build_dataset(tokenizer, dataset_id: str, dataset_split: str, nsamples: int):
    dataset = load_dataset(dataset_id, split=f"{dataset_split}[:{nsamples}]")
    dataset = dataset.shuffle(seed=42)

    def preprocess(example):
        messages = example["messages"]
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
        else:
            text = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in messages
            )
        return {
            "text": text,
        }

    return dataset.map(preprocess)


def build_recipe(with_fp8_kv: bool):
    quant_modifier_kwargs = {
        "ignore": ["lm_head"],
        "scheme": "W4A16_ASYM",
        "targets": ["Linear"],
    }
    if with_fp8_kv:
        quant_modifier_kwargs["kv_cache_scheme"] = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="tensor",
            dynamic=False,
            symmetric=True,
        )

    return [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(**quant_modifier_kwargs),
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print("Preparing calibration dataset")
    dataset = build_dataset(
        tokenizer=tokenizer,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        nsamples=args.num_calibration_samples,
    )

    recipe = build_recipe(args.with_fp8_kv)
    print(f"Running oneshot quantization, with_fp8_kv={args.with_fp8_kv}")
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    if not args.skip_sanity_generation:
        print("Running a short sanity generation")
        dispatch_model(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample = {key: value.to(model.device) for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=64)
        print(tokenizer.decode(output[0]))

    print(f"Saving compressed model to {output_dir}")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print("Done")


if __name__ == "__main__":
    main()
