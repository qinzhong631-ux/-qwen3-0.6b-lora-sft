from scripts.prepare_ultrachat import select_dialogue


class TinyTokenizer:
    """让对话的 token 数由内容字符数决定。"""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [1]
        for item in messages:
            ids.append(2 if item["role"] == "user" else 3)
            ids.extend([5] * len(item["content"]))
            ids.append(4)
        if add_generation_prompt:
            ids.append(3)
        return ids


def test_select_retains_multiturn_when_it_fits():
    messages = [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
    ]
    assert select_dialogue(messages, TinyTokenizer(), 20) == messages


def test_select_falls_back_to_first_complete_turn():
    messages = [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "long response"},
    ]
    assert select_dialogue(messages, TinyTokenizer(), 12) == messages[:2]


def test_select_rejects_oversize_and_invalid_roles():
    assert select_dialogue([{"role": "user", "content": "long prompt"},
                            {"role": "assistant", "content": "answer"}], TinyTokenizer(), 8) is None
    assert select_dialogue([{"role": "assistant", "content": "bad"}], TinyTokenizer(), 30) is None


def test_select_filters_contact_patterns_without_losing_safe_first_turn():
    messages = [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "write to person@example.org"},
        {"role": "assistant", "content": "okay"},
    ]
    assert select_dialogue(messages, TinyTokenizer(), 100) == messages[:2]
    assert select_dialogue(messages[:2] + [{"role": "user", "content": "call 123-456-7890"},
                                            {"role": "assistant", "content": "done"}], TinyTokenizer(), 100) == messages[:2]
    assert select_dialogue([{"role": "user", "content": "person@example.org"},
                            {"role": "assistant", "content": "reply"}], TinyTokenizer(), 100) is None
