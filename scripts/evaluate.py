"""����ͬ���÷ֱ����л����� LoRA adapter �ı�׼��׼��"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


TASK_SHOTS = {"mmlu": 5, "gsm8k": 5, "ifeval": 0}


def build_command(model: str, task: str, output: Path, adapter: str | None = None,
                  batch_size: int = 1, revision: str | None = None) -> list[str]:
    """���������ã����� IFEval �� few-shot ��׼������ͬ shot ����"""
    if task not in TASK_SHOTS or batch_size < 1:
        raise ValueError("����� batch_size ��Ч")
    model_args = f"pretrained={model}"
    if revision:
        model_args += f",revision={revision}"
    if adapter:
        model_args += f",peft={adapter}"
    return ["lm-eval", "run", "--model", "hf", "--model_args", model_args,
            "--tasks", task, "--num_fewshot", str(TASK_SHOTS[task]),
            "--batch_size", str(batch_size), "--apply_chat_template",
            "--output_path", str(output), "--log_samples"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--tasks", nargs="+", choices=list(TASK_SHOTS), default=list(TASK_SHOTS))
    p.add_argument("--dry-run", action="store_true", help="ֻ��ӡ�����ִ�л�׼")
    args = p.parse_args()
    if not args.adapter.exists() and not args.dry_run:
        p.error(f"adapter �����ڣ�{args.adapter}")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for task in args.tasks:
        for condition, adapter in (("base", None), ("lora", str(args.adapter.resolve()))):
            target = args.output / task / condition
            command = build_command(args.model, task, target, adapter, args.batch_size, args.revision)
            runs.append({"task": task, "condition": condition, "shots": TASK_SHOTS[task],
                         "command": command, "output": str(target)})
            if args.dry_run:
                print(" ".join(command))
                continue
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    manifest = {"model": args.model, "revision": args.revision,
                "adapter": str(args.adapter), "dry_run": args.dry_run,
                "created_utc": datetime.now(timezone.utc).isoformat(), "runs": runs}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
