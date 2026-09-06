# Project guidance

This is a headless SSH/H100 experiment. Preserve the matched effective-depth invariant and identical data/token budgets. Never claim H100 training or benchmark results without running there. Use uv and the project-local environment; keep data, checkpoints and secrets out of Git.

## SSH and Slurm assumptions

- Treat every SSH host as a Slurm cluster. Assume a new SSH session is on a login node with no allocated GPU, even when `nvidia-smi` or CUDA software is present.
- Never run training, calibration, evaluation, inference, GPU debugging, dataset tokenization, or other sustained compute directly from the login shell. Submit a batch job with `sbatch`, or use `srun` inside an active `salloc` allocation.
- Make Slurm the default documented execution path. Plain `./scripts/train.sh` commands are valid only inside a Slurm allocation where `SLURM_JOB_ID` is set.
- Do not use `nohup` as a substitute for Slurm, and do not document manual `CUDA_VISIBLE_DEVICES` selection as an SSH execution path. Treat existing instructions that suggest direct login-node training, `nohup` training, or manual GPU selection as defects and correct them before following or extending those instructions.
- Every GPU workflow must have a checked-in `.sbatch` entry point with explicit GPU count, CPU count, host memory, wall time, job name, and log destination. Keep partition, account, and cluster-specific module names configurable; never invent them.
- After Slurm grants the allocation and before creating run state, require the selected device to be an NVIDIA H100, expose at least 75 GiB, and support BF16. Let Slurm own `CUDA_VISIBLE_DEVICES`; never replace it in project scripts.
- Use `SLURM_SUBMIT_DIR` or an explicit project path, noninteractive commands, flushed logs, resumable checkpoints, and nonzero exit codes on failure. Do not assume notebooks, display servers, or persistent SSH sessions.
- Keep environment setup and large dataset preparation separate from the timed GPU training allocation. Run dependency installation, nontrivial setup, and dataset preparation in a CPU batch job or with `srun` inside `salloc`, unless the cluster policy explicitly permits that work on the login node. Provide checked-in Slurm jobs for expensive preparation; allow only lightweight Git and environment inspection on the login node by default.
- Show `sbatch`, `squeue`, `sacct`, and log-following commands in run instructions. A submitted job is not proof of successful execution: verify allocation, exit status, logs, checkpoints, and final result files before reporting completion.

## Development workflow

For changes requiring tests, use a separate testing agent to design and execute tests when one is available. Finish each implementation step, obtain independent verification, and commit verified changes before starting the next step. Keep CLI wrappers, Slurm entry points, and README examples consistent.
