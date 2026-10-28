"""对话数据校验、目标回复标注和确定性划分。"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch


def validate_conversation(messages: Any) -> list[dict[str, str]]:
    """校验单条多轮对话；仅允许最后一条 assistant 作为监督目标。"""
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages 至少需要 user 和 assistant 两条消息")
    cleaned = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}] 的 role 无效")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}] 的 content 必须是非空字符串")
        cleaned.append({"role": message["role"], "content": content})
    if cleaned[-1]["role"] != "assistant" or cleaned[-2]["role"] != "user":
        raise ValueError("最后两条消息必须依次为 user、assistant")
    if any(item["role"] == "system" for item in cleaned[1:]):
        raise ValueError("system 消息只能位于对话开头")
    return cleaned


def encode_conversation(messages: Any, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    """用模型原生模板编码，并把历史及提示部分的标签设为 -100。"""
    messages = validate_conversation(messages)
    if max_length < 2:
        raise ValueError("max_length 必须至少为 2")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(prefix, dict):
        prefix = prefix["input_ids"]
    if isinstance(full, dict):
        full = full["input_ids"]
    if len(full) > max_length:
        raise ValueError(f"对话长度 {len(full)} 超过 max_length={max_length}；不截断目标回复")
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat Template 的生成前缀与完整对话不一致，无法安全构造标签")
    if not full[len(prefix):]:
        raise ValueError("目标回复没有可监督 token")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


def collate_batch(examples: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """批内补齐；padding 的注意力掩码为 0，标签为 -100。"""
    if not examples:
        raise ValueError("不能对空批次补齐")
    width = max(len(item["input_ids"]) for item in examples)
    result = {}
    for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
        result[key] = torch.tensor([item[key] + [padding] * (width - len(item[key]))
                                    for item in examples], dtype=torch.long)
    return result


def load_jsonl(path: str | Path) -> list[list[dict[str, str]]]:
    """逐行读取 JSONL，并在训练前报告文件和行号。"""
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(validate_conversation(value["messages"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path} 没有有效对话")
    return records


def split_conversations(
    conversations: list[list[dict[str, str]]], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[list[dict[str, str]]], list[list[dict[str, str]]]]:
    """按输入对话分组划分，避免同一输入的不同答案跨训练和验证集。"""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须介于 0 和 1 之间")
    groups: dict[str, list[list[dict[str, str]]]] = {}
    seen: set[str] = set()
    for raw in conversations:
        messages = validate_conversation(raw)
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if serialized in seen:
            continue
        seen.add(serialized)
        key = hashlib.sha256(json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        groups.setdefault(key, []).append(messages)
    if len(groups) < 2:
        raise ValueError("至少需要两个不同的输入对话，才能划分训练和验证集")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_valid = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
    valid_keys = set(keys[:n_valid])
    train = [item for key in keys if key not in valid_keys for item in groups[key]]
    valid = [item for key in keys if key in valid_keys for item in groups[key]]
    return train, valid
