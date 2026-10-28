"""加载基座或本地 LoRA adapter，生成一条回复。"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.max_new_tokens < 1 or args.temperature <= 0:
        p.error("max-new-tokens 和 temperature 必须为正数")
    torch.manual_seed(args.seed)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, revision=args.revision if not args.adapter else None)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        output = model.generate(inputs, max_new_tokens=args.max_new_tokens,
                                do_sample=True, temperature=args.temperature,
                                pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
