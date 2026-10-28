import pytest

from qwen_lora.data import collate_batch, encode_conversation, split_conversations, validate_conversation


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # 测试模型模板前缀与监督边界，无需下载真实权重。
        tokens = [1]
        for message in messages:
            tokens += [2 if message["role"] == "user" else 3, len(message["content"]), 4]
        if add_generation_prompt:
            tokens += [3]
        elif messages and messages[-1]["role"] == "assistant":
            tokens[-3:-1] = [3, len(messages[-1]["content"])]
        return tokens


def test_validate_rejects_missing_target():
    with pytest.raises(ValueError):
        validate_conversation([{"role": "user", "content": "hi"}])


def test_encode_masks_history_and_keeps_target():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            if add_generation_prompt:
                return [10, 11, 12, 13]
            return [10, 11, 12, 13, 20, 21]

    messages = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"}, {"role": "assistant", "content": "four"}]
    result = encode_conversation(messages, Tokenizer(), 12)
    assert result["labels"] == [-100, -100, -100, -100, 20, 21]
    assert result["attention_mask"] == [1] * 6


def test_encode_rejects_template_mismatch_and_overlength():
    class BadTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            return [1, 2] if add_generation_prompt else [3, 4, 5]

    item = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    with pytest.raises(ValueError, match="不一致"):
        encode_conversation(item, BadTokenizer(), 10)
    with pytest.raises(ValueError, match="超过"):
        encode_conversation(item, BadTokenizer(), 2)


def test_collate_masks_padding():
    result = collate_batch([
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [3], "attention_mask": [1], "labels": [3]},
    ], pad_token_id=0)
    assert result["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert result["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert result["labels"].tolist() == [[-100, 2], [3, -100]]


def test_split_keeps_same_prompt_together_and_deduplicates():
    first = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}]
    variant = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "y"}]
    other = [{"role": "user", "content": "b"}, {"role": "assistant", "content": "z"}]
    train, valid = split_conversations([first, first, variant, other], 0.5, 42)
    assert len(train) + len(valid) == 3
    assert not ({item[0]["content"] for item in train} & {item[0]["content"] for item in valid})
