"""以相同配置分别运行基座和 LoRA adapter 的标准基准。"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


TASK_SHOTS = {"mmlu": 5, "gsm8k": 5, "ifeval": 0}


def build_command(model: str, task: str, output: Path, adapter: str | None = None,
                  batch_size: int = 1, revision: str | None = None,
                  limit: int | None = None) -> list[str]:
    """单任务单配置，避免 IFEval 与 few-shot 基准混用相同 shot 数。"""
    if task not in TASK_SHOTS or batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("任务、batch_size 或 limit 无效")
    model_args = f"pretrained={model}"
    if revision:
        model_args += f",revision={revision}"
    if adapter:
        model_args += f",peft={adapter}"
    command = ["lm-eval", "run", "--model", "hf", "--model_args", model_args,
            "--tasks", task, "--num_fewshot", str(TASK_SHOTS[task]),
            "--batch_size", str(batch_size), "--apply_chat_template",
            "--output_path", str(output), "--log_samples"]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--tasks", nargs="+", choices=list(TASK_SHOTS), default=list(TASK_SHOTS))
    p.add_argument("--limit", type=int, default=None, help="仅用于流程检查；结果不代表完整基准分数")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行基准")
    args = p.parse_args()
    if not args.adapter.exists() and not args.dry_run:
        p.error(f"adapter 不存在：{args.adapter}")
    if args.limit is not None and args.limit < 1:
        p.error("limit 必须为正整数")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for task in args.tasks:
        for condition, adapter in (("base", None), ("lora", str(args.adapter.resolve()))):
            target = args.output / (f"limit-{args.limit}" if args.limit else "full") / task / condition
            command = build_command(args.model, task, target, adapter, args.batch_size, args.revision, args.limit)
            runs.append({"task": task, "condition": condition, "shots": TASK_SHOTS[task],
                         "command": command, "output": str(target)})
            if args.dry_run:
                print(" ".join(command))
                continue
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    manifest = {"model": args.model, "revision": args.revision,
                "adapter": str(args.adapter), "dry_run": args.dry_run,
                "limit": args.limit, "comparable_full_benchmark": args.limit is None and not args.dry_run,
                "created_utc": datetime.now(timezone.utc).isoformat(), "runs": runs}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
