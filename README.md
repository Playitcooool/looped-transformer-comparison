# Looped Transformer Comparison

A headless, single-GPU experiment training two causal language models from scratch on WikiText-2. Target deployment: one NVIDIA H100 80GB. No GUI, notebook, pretrained weights, or external logging account is required.

## Controlled comparison

| Setting | Standard decoder | Looped decoder |
|---|---:|---:|
| Unique transformer blocks | 24 | 6 |
| Stack repetitions | 1 | 4 |
| Effective block applications | 24 | 24 |
| Hidden width / attention heads | 1,088 / 17 | 1,088 / 17 |
| Parameters at vocabulary 8,192 | 350,451,328 | 94,508,032 |
| Context / target BPE vocabulary | 256 / 8,192 | 256 / 8,192 |
| Optimizer steps / tokens per step | 2,000 / 16,384 | 2,000 / 16,384 |

The constructor enforces `loop_layers * loops == depth`. The looped network passes hidden states through the same stack repeatedly; there is no detach, extra loop embedding, or loop-specific parameter. Both use pre-norm LayerNorm, causal SDPA attention, GELU MLPs with 4× expansion, learned positions added once, tied token/output embeddings, and zero dropout by default. There is no KV cache.

**This controls effective depth, width, data, batches, tokenizer, optimizer, learning-rate schedule and total training tokens, not parameter count.** The looped model has fewer unique parameters and optimizer states. Block forward/backward operation counts are broadly comparable, but optimizer cost, kernel behavior and wall time need not match. Shared parameters accumulate gradients from every use. Repeated depth still needs activation memory during backpropagation.

Both runs use the same seeded batch generator, so every optimizer step receives identical sampled training windows, including after resume. Initialization shares token/position weights and the first six block values at the same seed; the remaining standard blocks are independently initialized. Each model gets its own optimizer. GPU kernels may not be bitwise deterministic. Run several seeds before interpreting differences as reliable findings. No separately tuned hyperparameters are claimed.

## H100 quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a working NVIDIA driver first. Clone this repository and enter its root, then:

```bash
./scripts/setup.sh
./scripts/check.sh --require-cuda
./scripts/prepare.sh
./scripts/train.sh
```

`setup.sh` creates this repository's `.venv` using the committed cross-platform `uv.lock`. PyTorch's Linux dependencies supply its CUDA runtime; the host still needs a compatible NVIDIA driver. The current lock selects PyTorch 2.14.0 and CUDA 13 Linux dependencies, with Linux glibc 2.28+ wheels. CUDA 13 normally needs an R580-or-newer NVIDIA driver; see [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html). Check the reported CUDA build and GPU before training. If your cluster has an older driver, ask its administrator for a compatible runtime or resolve a matching older PyTorch version in this project before running both arms; do not mix runtimes between arms. Python is pinned to 3.12. The default configuration explicitly requests BF16; it fails clearly on unsupported hardware rather than silently changing precision.

Training runs the standard model, then the looped model, on one visible GPU. The default output directory is `runs/h100-350m`; start a fresh run because previous 42M checkpoints are incompatible. The earlier small configuration is preserved as `configs/h100-small.json` (use a separate output directory). The default processes **32,768,000 training tokens per model**. This remains a bounded small-data comparison, not a claim that a 350M model is fully pretrained. WikiText-2 windows will be reused extensively; watch validation loss for overfitting. WikiText-103 is available below for a larger corpus, with the same prepared data and token budget in both arms. Data windows are sampled with replacement; this is a fixed-token-budget experiment, not an epoch-based traversal. Training and validation print JSON to stdout with immediate flushing.

```bash
# A noninteractive SSH shell, keep running after disconnect:
nohup ./scripts/train.sh > training.log 2>&1 &
# Or submit from the repository root, adapting partition/account to your cluster:
sbatch scripts/train.sbatch
# Select a GPU outside a scheduler, if needed:
CUDA_VISIBLE_DEVICES=0 ./scripts/train.sh --output runs/h100-350m-seed42
# Continue interrupted paired training (also starts the second model if needed):
./scripts/train.sh --resume
```

Do not launch both the quick-start training and nohup/Slurm examples for the same directory. A fresh run rejects occupied model output directories. Resume requires exactly matching settings and data checksums, and restores model, optimizer, step, data RNG and PyTorch CPU/CUDA RNG states. It continues from the last validation checkpoint (every 100 steps by default); work since that checkpoint is lost. Keep `last.pt` and `best.pt` together. Checkpoint writes use an atomic rename. Resume on the same device/runtime for reproducibility. Metrics may contain repeated steps if a process died after logging but before checkpointing.

**Resources:** the default standard model is approximately 350M parameters; the looped model is approximately 94.5M. Width 1,088 with 17 heads retains 64 dimensions per head. Microbatch size 4 and accumulation 16 preserve 16,384 tokens per optimizer step while limiting activation memory. FP32 parameters, gradients and Adam moments alone take about 5.6GB for the standard model; activations, attention workspace, logits and temporary copies require additional memory. Actual peak memory on H100 has not been measured. CPU RAM request in the Slurm example is 32GB. Allow roughly 25–35GB of free disk for the Linux CUDA dependencies, caches, data and paired optimizer checkpoints (an estimate). Runtime is not measured on H100: the first logged steps and `train_tokens_per_second` provide the basis for an estimate. The 4-hour Slurm limit is a configurable allocation request, not a benchmark. Scale batch size and inverse accumulation together to retain 16,384 tokens/step. This project intentionally does not attempt multi-GPU training.

## Data and evaluation

The downloader uses [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext) `wikitext-2-raw-v1` at a pinned revision. Respect the source dataset's CC BY-SA attribution/license terms. Byte-level BPE is trained **only on the training split**, including a full byte alphabet; its actual vocabulary size is saved and used by both models. Nonblank source rows are independently encoded and terminated by `<eos>`. Packed attention can span row boundaries. Train, validation and test remain separate token streams.

The manifest records the source revision, tokenizer checksum and all split checksums/token counts. Training validates them before use. Local text is also supported:

```bash
./scripts/prepare.sh --local-dir /path/to/text-splits --output data/custom --vocab-size 8192
# Folder must contain train.txt, validation.txt and test.txt, split by the caller.
# WikiText-103 is optional (larger memory/download requirements):
./scripts/prepare.sh --dataset-config wikitext-103-raw-v1 --output data/wikitext103
```

Validation evaluates all next-token targets in fixed, non-overlapping context windows, including the short final window. Each target is counted once; context resets every 256 targets. Validation selects `best.pt`. The held-out test split is evaluated at completion using that checkpoint. Reported perplexity is **this tokenizer's BPE-token perplexity**, not directly comparable with published word-level WikiText perplexities or different context protocols. Repeated test inspection should not guide hyperparameter tuning.

## Outputs and commands

Under `runs/h100-350m/standard/` and `runs/h100-350m/looped/`:

- `metadata.json`: resolved settings, parameter counts, data identity and runtime information.
- `metrics.jsonl`: training loss and validation measurements.
- `last.pt`: resumable state; `best.pt`: lowest validation-loss state.
- `result.json`: held-out test loss/perplexity, tokens, training-only throughput and peak CUDA allocation.

The parent contains `comparison.json` and `comparison.md`. Training-only time includes batch construction and optimizer updates, excludes validation/checkpoint/test time, and synchronizes CUDA. Peak CUDA allocation includes evaluation and may differ after resume; it is allocated tensor memory rather than total process VRAM. The comparison rejects mismatched settings, data and token budgets.

```bash
./scripts/evaluate.sh --checkpoint runs/h100-350m/standard/best.pt
./scripts/evaluate.sh --checkpoint runs/h100-350m/looped/best.pt
./scripts/infer.sh --checkpoint runs/h100-350m/looped/best.pt --prompt 'The history of science'
# Greedy generation:
./scripts/infer.sh --checkpoint runs/h100-350m/standard/best.pt --prompt 'The Earth' --temperature 0
# Train just one architecture:
uv run looped-transformer-comparison train --architecture standard --output runs/single
# Pause at a checkpoint without changing the full LR schedule:
uv run looped-transformer-comparison train --architecture looped --output runs/paused --stop-after 100
uv run looped-transformer-comparison train --architecture looped --output runs/paused --resume
# Debug synchronous CUDA errors:
./scripts/debug.sh --architecture looped --output runs/debug --stop-after 1
```

Evaluation command uses FP32 explicitly for portability; the final training report uses configured BF16 on H100, so tiny numerical differences are expected. Generation verifies the tokenizer checksum and rolls the context window when full. These small models are experimental and should not be expected to produce fluent text after a smoke run.

## Local verification and custom experiments

```bash
./scripts/setup.sh
uv run pytest -q
# Small real-WikiText check (full WikiText validation can still take time on CPU):
./scripts/prepare.sh --output data/smoke --vocab-size 512
./scripts/train.sh --config configs/smoke.json --data data/smoke --output runs/smoke
```

`configs/smoke.json` is for CPU verification only. For research runs, copy `configs/h100.json`, modify `training.seed`, and give each paired run a separate output directory. Try seeds 42, 43 and 44. To change loop factor, retain `depth == loop_layers * loops`, for example 24 = 12×2, 6×4 or 3×8. Keeping 24×1 for the looped model gives the architecture-equivalence control. Each configuration has the same width in both arms; parameter matching is a different experiment.

Implementation uses [PyTorch causal scaled-dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). Setup and data preparation need internet; after dependencies and data are cached, training/evaluation are fully local and headless. Data, environments, secrets, logs and model weights are ignored by Git.
