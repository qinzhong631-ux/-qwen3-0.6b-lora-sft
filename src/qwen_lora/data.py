"""�Ի�����У�顢Ŀ��ظ���ע��ȷ���Ի��֡�"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch


def validate_conversation(messages: Any) -> list[dict[str, str]]:
    """У�鵥�����ֶԻ������������һ�� assistant ��Ϊ�ලĿ�ꡣ"""
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages ������Ҫ user �� assistant ������Ϣ")
    cleaned = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}] �� role ��Ч")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}] �� content �����Ƿǿ��ַ���")
        cleaned.append({"role": message["role"], "content": content})
    if cleaned[-1]["role"] != "assistant" or cleaned[-2]["role"] != "user":
        raise ValueError("���������Ϣ��������Ϊ user��assistant")
    if any(item["role"] == "system" for item in cleaned[1:]):
        raise ValueError("system ��Ϣֻ��λ�ڶԻ���ͷ")
    return cleaned


def encode_conversation(messages: Any, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    """��ģ��ԭ��ģ����룬������ʷ����ʾ���ֵı�ǩ��Ϊ -100��"""
    messages = validate_conversation(messages)
    if max_length < 2:
        raise ValueError("max_length ��������Ϊ 2")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(prefix, dict):
        prefix = prefix["input_ids"]
    if isinstance(full, dict):
        full = full["input_ids"]
    if len(full) > max_length:
        raise ValueError(f"�Ի����� {len(full)} ���� max_length={max_length}�����ض�Ŀ��ظ�")
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat Template ������ǰ׺�������Ի���һ�£��޷���ȫ�����ǩ")
    if not full[len(prefix):]:
        raise ValueError("Ŀ��ظ�û�пɼල token")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


def collate_batch(examples: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """���ڲ��룻padding ��ע��������Ϊ 0����ǩΪ -100��"""
    if not examples:
        raise ValueError("���ܶԿ����β���")
    width = max(len(item["input_ids"]) for item in examples)
    result = {}
    for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
        result[key] = torch.tensor([item[key] + [padding] * (width - len(item[key]))
                                    for item in examples], dtype=torch.long)
    return result


def load_jsonl(path: str | Path) -> list[list[dict[str, str]]]:
    """���ж�ȡ JSONL������ѵ��ǰ�����ļ����кš�"""
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
        raise ValueError(f"{path} û����Ч�Ի�")
    return records


def split_conversations(
    conversations: list[list[dict[str, str]]], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[list[dict[str, str]]], list[list[dict[str, str]]]]:
    """������Ի����黮�֣�����ͬһ����Ĳ�ͬ�𰸿�ѵ������֤����"""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction ������� 0 �� 1 ֮��")
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
        raise ValueError("������Ҫ������ͬ������Ի������ܻ���ѵ������֤��")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_valid = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
    valid_keys = set(keys[:n_valid])
    train = [item for key in keys if key not in valid_keys for item in groups[key]]
    valid = [item for key in keys if key in valid_keys for item in groups[key]]
    return train, valid
