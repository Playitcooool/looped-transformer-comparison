# Independent verification

Verified 2026-09-05 by a separate testing agent on macOS, Python 3.12.12, PyTorch 2.14.0, CPU only. No H100 or CUDA execution was available.

## Automated suite

Command: `env -u VIRTUAL_ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false uv run --no-sync pytest -q`

Result: **18 passed in 7.34 seconds**.

Coverage includes:

- Invalid depth/product, dimensions and dropout are rejected; forward hooks confirm actual block applications equal effective depth.
- Shared embedding and initial blocks match across architectures for a shared seed. One-loop architecture matches the standard model exactly.
- Future-token changes cannot affect earlier logits in either architecture.
- Shared block gradients match the sum of equivalent independently unrolled block gradients, verifying gradient flow through every recurrence without detachment.
- Altering held-out text does not alter tokenizer training. Prepared data validates token counts/checksums and rejects corruption and occupied destinations.
- Seeded batches match and targets are correctly shifted.
- Evaluation counts each target once, handles partial tails, weights by tokens, and restores model training/evaluation mode.
- Both architectures resume to bit-identical final weights and RNG states after an off-schedule pause with nonzero dropout and gradient accumulation. Changed configs and occupied outputs are rejected.
- Paired CLI training, standalone evaluation, greedy generation, report generation, completed-run resume, and mismatched comparison rejection succeed.
- Shell scripts and Slurm script pass `bash -n`; `uv lock --check` succeeds. CPU capability checks work and required CUDA check fails as intended without a GPU.
- Mocked tests verify `auto`, bare `cuda`, and explicit CUDA indices resolve correctly. These are control-flow tests, not GPU execution.

## Live WikiText verification

Successfully downloaded `Salesforce/wikitext`, config `wikitext-2-raw-v1`, pinned revision `f776294184f13b8ff2337b3841cf9269a6216d1e`, and prepared a 512-token BPE vocabulary using only its training split.

Prepared token counts: train **5,313,311**, validation **555,985**, test **628,817**. Tokenizer SHA-256: `a74a835d699f3c3ddb6ffc362f7f0b9885a70e522aad57746371afbbe857b9da`.

A deliberately bounded verification subset used the first 4,096 training tokens and first 1,025 tokens from each held-out split, retaining the train-only tokenizer. The subset manifest explicitly labels this truncation and has recomputed split checksums. Both architectures completed `configs/smoke.json` through `scripts/train.sh`: 4 optimizer steps, 512 training tokens, and 1,024 test targets per architecture.

| CPU verification model | Unique layers | Effective depth | Parameters | Test loss |
|---|---:|---:|---:|---:|
| Standard | 4 | 4 | 68,288 | 6.189627 |
| Looped | 2 | 4 | 42,880 | 6.193700 |

These tiny runs verify operation only; they do not support architecture-quality, speed, convergence, or H100 performance conclusions. Generated data and checkpoints remain under ignored `data/verify-wiki`, `data/verify-wiki-subset`, and `runs/verify-wiki-subset` directories and are not published to Git.

## Default H100 configuration inspection

Actual CPU instantiation of `configs/h100.json` gives **42,155,008** parameters for the 12-layer standard model and **13,783,552** parameters for the 3-layer looped model repeated 4 times. Both have effective depth 12. This is depth/token-budget matching; parameter count is intentionally different.

The Linux lock selects CUDA 13 dependencies for PyTorch 2.14 and a manylinux 2.28 wheel: plan for an R580+ NVIDIA driver, glibc 2.28+, and roughly 10–15 GB for the environment/cache. Actual Linux dependency installation, H100 BF16 kernels, CUDA RNG resume, GPU memory use, throughput, multi-hour training, and Slurm scheduling remain unverified and must be checked on the deployment host. No GPU timing or memory benchmark is claimed.
