# Qwen3-0.6B LoRA Fine-Tuning Implementation Plan

> **For agentic workers:** Execute the tasks below in order in this session. No subagents are needed for this single training pipeline.

**Goal:** Train and verify a LoRA adapter for the official Qwen3-0.6B-Base model on a licensed public instruction dataset, then compare it with the base model using actual evaluation output.

**Architecture:** Convert a deterministic sample of UltraChat 200k `train_sft` into the existing JSONL dialogue format. Retain one target assistant turn per original dialogue and mask only the target reply. Run a short GPU smoke train before the longer run; store all data, weights and result files under ignored paths. Evaluate base and adapter with identical settings and report only measured values.

**Tech Stack:** Python 3.12, PyTorch 2.7.1+cu118, Transformers 4.57.6, PEFT 0.21.2, RTX 3060 Laptop 6 GB.

## Global Constraints

- Dataset: `HuggingFaceH4/ultrachat_200k`, `train_sft`; its card declares MIT license. Use the public Dataset Viewer `/rows` API, at most 100 rows per request.
- Base model: `Qwen/Qwen3-0.6B-Base`; pin the resolved revision in run metadata.
- Hardware: approximately 5.3 GB free GPU memory at plan time; use batch size 1, gradient accumulation, gradient checkpointing, BF16 where available, and a 512 token cap for the initial trial.
- Keep downloaded/derived data, model weights, adapter checkpoints and benchmark outputs outside Git tracking.
- Never describe a limited benchmark sample as a full standard benchmark score. Never invent measurements.

---

### Task 1: Prepare auditable training data

**Files:** Create `scripts/prepare_ultrachat.py`; create `tests/test_prepare_ultrachat.py`; update `README.md`.

**Interfaces:** `select_dialogue(messages, tokenizer, max_length) -> list[dict[str,str]] | None`; CLI writes `data/ultrachat_sft.jsonl` and `data/ultrachat_sft.manifest.json`.

- [x] Test role validation, token-length filtering, and deterministic dialogue selection with synthetic API rows.
- [x] Implement bounded API pagination with timeouts/retries, deterministic offset windows, license/source metadata, and JSONL checksum.
- [x] Run the focused tests, then prepare a fixed sample of 3,000 source rows with contact-pattern filtering: 1,703 accepted, 1,297 rejected, 660 multi-turn.
- [x] Commit script, tests and documentation; keep `data/` ignored.

### Task 2: Smoke train on official weights

**Files:** Update `src/qwen_lora/train.py` only if a real run exposes a reproducible issue; add a focused regression test for any fix.

- [ ] Download the exact base revision into the Hugging Face cache and record its commit SHA.
- [ ] Train 10 optimization steps using the prepared data, `--max-length 512 --batch-size 1 --gradient-accumulation 4 --eval-steps 5`, with a separate ignored output directory.
- [ ] Inspect actual train/validation loss, trainable parameter count, elapsed time, peak CUDA memory and adapter files. If out of memory, reduce sequence length to 384, then use optional 4-bit quantization only if necessary.
- [ ] Run one adapter generation and record that it ran; do not judge quality from a single example.

### Task 3: Main LoRA run and ablation

**Files:** Update `README.md` with only observed settings/results; preserve raw `outputs/` logs locally.

- [ ] Use the successful smoke configuration for a 200-step primary run with seed 42 and fixed source data/checksum.
- [ ] Run at least one controlled rank comparison with the same data, step budget, learning rate, seed, max length, and evaluation schedule. Record each run's trainable parameter count; changing rank cannot keep the exact trainable count fixed.
- [ ] Summarize loss trajectory, runtime and peak GPU memory from `run.json`, without claiming improved capability from training loss alone.

### Task 4: Base versus adapter evaluation and publication

**Files:** Modify `scripts/evaluate.py` only if a real harness invocation shows a CLI issue; update `README.md` and `docs/PLAN.md`.

- [ ] Install the optional evaluation dependency and verify task availability.
- [ ] Run a small labeled evaluation check first. Run the full MMLU, GSM8K and IFEval comparisons only if dataset access, time and GPU budget permit; record any partial result as partial.
- [ ] Compare base and adapter under the same model revision, template, task shots, quantization and harness version. Keep raw result JSON.
- [ ] Run relevant tests, inspect Git diff and secret/large-file exclusions, commit source/docs, then publish to the confirmed GitHub repository and verify remote file contents.

## Stop conditions and reporting

If model download, CUDA memory, evaluation dependencies or GitHub access blocks a phase, capture the exact error and complete unaffected work. Do not replace the official model with random weights or report smoke metrics as final-model results.
