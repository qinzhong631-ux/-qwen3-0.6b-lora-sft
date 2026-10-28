"""Qwen3-0.6B Base �� LoRA �ල΢����ڡ�"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data import collate_batch, encode_conversation, load_jsonl, split_conversations


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class EncodedDataset(Dataset):
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--revision", default="main", help="ģ�Ͳֿ� revision������ʹ�þ��� commit SHA")
    p.add_argument("--data", type=Path, default=Path("examples/toy_conversations.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/demo"))
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=5)
    p.add_argument("--quantize-4bit", action="store_true", help="��ѡ���谲װ .[quant]���� CUDA")
    return p


def run(args: argparse.Namespace) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if args.rank < 1 or args.alpha < 1 or not 0 <= args.dropout < 1:
        raise ValueError("rank/alpha ����Ϊ������dropout ������ [0, 1) ��")
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.max_steps < 1 or args.eval_steps < 1:
        raise ValueError("batch-size��gradient-accumulation��max-steps��eval-steps ����Ϊ����")
    if args.quantize_4bit and not torch.cuda.is_available():
        raise RuntimeError("4bit ����Ŀǰ�������� CUDA �豸������")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = load_jsonl(args.data)
    train_raw, valid_raw = split_conversations(records, args.validation_fraction, args.seed)
    skipped = {"train": 0, "validation": 0}

    def encode_all(items: list, split_name: str) -> list[dict[str, list[int]]]:
        encoded = []
        for messages in items:
            try:
                encoded.append(encode_conversation(messages, tokenizer, args.max_length))
            except ValueError as exc:
                if "���� max_length" not in str(exc):
                    raise
                skipped[split_name] += 1
        if not encoded:
            raise ValueError(f"{split_name} ��û�п�ѵ�������������� --max-length ��������")
        return encoded

    train_data = EncodedDataset(encode_all(train_raw, "train"))
    valid_data = EncodedDataset(encode_all(valid_raw, "validation"))
    model_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True, "revision": args.revision}
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    found = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    absent = set(TARGET_MODULES) - found
    if absent:
        raise RuntimeError(f"ģ��ȱ�� LoRA Ŀ��ģ��: {sorted(absent)}")
    if args.quantize_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.alpha,
        lora_dropout=args.dropout, target_modules=list(TARGET_MODULES), bias="none",
    ))
    # �ݶȼ�����������ʱ�豣�� embedding ������ݶ�ͨ·��
    model.enable_input_require_grads()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    settings = TrainingArguments(
        output_dir=str(output / "trainer"), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, logging_steps=1,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="no", report_to="none", seed=args.seed, data_seed=args.seed,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=settings, train_dataset=train_data, eval_dataset=valid_data,
        data_collator=lambda items: collate_batch(items, tokenizer.pad_token_id),
    )
    started = time.perf_counter()
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    duration = time.perf_counter() - started
    adapter_dir = output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    import transformers
    import peft
    metadata = {
        "model": args.model, "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "seed": args.seed,
        "hyperparameters": {key: value for key, value in vars(args).items() if key not in {"model", "data", "output", "revision"} and not isinstance(value, Path)},
        "dataset": {"train": len(train_data), "validation": len(valid_data), "skipped_overlength": skipped},
        "parameters": {"total": total, "trainable": trainable, "trainable_fraction": trainable / total},
        "environment": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
                        "platform": platform.platform(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "peft": peft.__version__,
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                        "cuda_runtime": torch.version.cuda},
        "runtime_seconds": duration,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "train_metrics": train_result.metrics, "validation_metrics": evaluation,
        "loss_history": trainer.state.log_history,
    }
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "train_loss": train_result.metrics.get("train_loss"),
                      "validation_loss": evaluation.get("eval_loss"), "trainable_parameters": trainable}, ensure_ascii=False))
    return output


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
