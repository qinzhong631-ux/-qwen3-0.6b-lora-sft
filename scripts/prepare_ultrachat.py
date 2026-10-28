"""从公开 UltraChat 200k 的 train_sft 按固定窗口抽样，生成训练 JSONL。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from qwen_lora.data import encode_conversation, validate_conversation


DATASET = "HuggingFaceH4/ultrachat_200k"
DATASET_API = f"https://huggingface.co/api/datasets/{DATASET}"
ROWS_API = "https://datasets-server.huggingface.co/rows"
MODEL = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
CONTACT_PATTERN = re.compile(
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)"
)


def select_dialogue(messages: Any, tokenizer: Any, max_length: int) -> list[dict[str, str]] | None:
    """优先选择前两轮；若超长则退回首轮，避免失去对回答必需的历史。"""
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    candidates = [messages[:4], messages[:2]] if len(messages) >= 4 else [messages[:2]]
    for candidate in candidates:
        try:
            cleaned = validate_conversation(candidate)
            if CONTACT_PATTERN.search("\n".join(message["content"] for message in cleaned)):
                continue
            encode_conversation(cleaned, tokenizer, max_length)
            return cleaned
        except ValueError:
            continue
    return None


def fetch_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    """对暂时性网络错误重试；永久性 HTTP 错误向上报告。"""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code < 500:
                break
            if attempt == 2:
                break
    raise RuntimeError(f"无法读取公开数据源 {url}: {last_error}") from last_error


def prepare(source_rows: int, max_length: int, seed: int, output: Path,
            model_revision: str = MODEL_REVISION) -> dict:
    if source_rows < 100 or source_rows % 100:
        raise ValueError("source_rows 必须是至少 100 的整百数")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"输出已存在，请换一个路径以保留既有数据：{output}")
    session = requests.Session()
    source = fetch_json(session, DATASET_API)
    license_name = (source.get("cardData") or {}).get("license")
    if license_name != "mit":
        raise RuntimeError(f"数据集许可与预期不符：{license_name!r}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=model_revision, use_fast=True)
    first_page = fetch_json(session, ROWS_API, {
        "dataset": DATASET, "config": "default", "split": "train_sft", "offset": 0, "length": 1,
    })
    total = first_page["num_rows_total"]
    pages = list(range(0, total, 100))
    count = min(source_rows // 100, len(pages))
    offsets = sorted(random.Random(seed).sample(pages, count))
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted: list[list[dict[str, str]]] = []
    seen: set[str] = set()
    rejected = 0
    duplicates = 0
    downloaded = 0
    for index, offset in enumerate(offsets, 1):
        page = fetch_json(session, ROWS_API, {
            "dataset": DATASET, "config": "default", "split": "train_sft",
            "offset": offset, "length": min(100, total - offset),
        })
        rows = page["rows"]
        downloaded += len(rows)
        for entry in rows:
            messages = select_dialogue(entry["row"].get("messages"), tokenizer, max_length)
            if messages is None:
                rejected += 1
                continue
            fingerprint = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            accepted.append(messages)
        print(f"窗口 {index}/{count}: 已读取 {downloaded} 行，接受 {len(accepted)} 条", flush=True)
    if len(accepted) < 2:
        raise RuntimeError("有效对话少于两条，无法训练/验证划分")
    raw = "".join(json.dumps({"messages": item}, ensure_ascii=False) + "\n" for item in accepted).encode("utf-8")
    output.write_bytes(raw)
    manifest = {
        "dataset": DATASET, "config": "default", "split": "train_sft",
        "dataset_revision_at_download": source.get("sha"), "license": license_name,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows_requested": source_rows, "source_rows_downloaded": downloaded,
        "accepted": len(accepted), "rejected": rejected, "duplicates": duplicates,
        "multiturn": sum(len(item) > 2 for item in accepted), "offsets": offsets,
        "seed": seed, "max_length": max_length, "model": MODEL,
        "model_revision": model_revision, "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "selection": "first four messages if valid, contact-pattern free, and within max_length; else first two",
        "contact_filter": "Reject candidate spans with email-like or phone-like patterns; heuristic, not a privacy guarantee.",
        "limitation": "Dataset Viewer rows API serves the dataset at request time; it cannot pin the source revision.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-rows", type=int, default=3000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("data/ultrachat_sft.jsonl"))
    p.add_argument("--model-revision", default=MODEL_REVISION)
    args = p.parse_args()
    result = prepare(args.source_rows, args.max_length, args.seed, args.output, args.model_revision)
    print(json.dumps({key: result[key] for key in ("accepted", "rejected", "duplicates", "multiturn", "jsonl_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
