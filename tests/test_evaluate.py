from scripts.evaluate import build_command


def test_build_command_uses_same_model_and_task_specific_shots(tmp_path):
    base = build_command("Qwen/Qwen3-0.6B-Base", "ifeval", tmp_path / "base")
    adapted = build_command("Qwen/Qwen3-0.6B-Base", "ifeval", tmp_path / "adapter", "adapter")
    assert base[base.index("--num_fewshot") + 1] == "0"
    assert adapted[adapted.index("--num_fewshot") + 1] == "0"
    assert "peft=adapter" in adapted[adapted.index("--model_args") + 1]
    assert "peft=" not in base[base.index("--model_args") + 1]
