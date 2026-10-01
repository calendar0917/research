# E2E-DictEnv-RoleCorr-Increment-v2 — GPU environment blocker record (2026-10-01)

Operational note (no scientific result).  The round implementation and
pre-registration are complete and locally tested; the formal GPU screen could
not start because the remote compute host `res` (`a100-2`) cannot initialise
CUDA at all.  This note records the exact evidence so a future session can
resume immediately after the infrastructure is repaired.

## 1. Symptom

On `res` (2026-10-01, uptime 22 days):

```text
$ nvidia-smi
Unable to determine the device handle for GPU0000:4B:00.0: Unknown Error
(exit 255)

$ nvidia-smi -i 1          # per-device query DOES work
GPU 1  NVIDIA A100-SXM4-40GB ... 31C  P0  34W / 400W  14MiB / 40960MiB  0%
```

CUDA driver initialisation fails independently of torch:

```text
$ python3 - <<'PY'
import ctypes
cuda = ctypes.CDLL("libcuda.so.1")
print("cuInit rc", cuda.cuInit(0))
n = ctypes.c_int(); print("cuDeviceGetCount rc", cuda.cuDeviceGetCount(ctypes.byref(n)), "n", n.value)
PY
cuInit rc 999          # CUDA_ERROR_UNKNOWN
cuDeviceGetCount rc 3  # cudaErrorInitializationError, n = 0
```

torch therefore reports `cuda_available=False, device_count=0` and raises the
warning `Can't initialize NVML`.  `CUDA_VISIBLE_DEVICES` is applied *after*
driver init, so every restriction tried still fails:

```text
CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES=0000:b1:00.0
CUDA_VISIBLE_DEVICES=b1:00.0
CUDA_VISIBLE_DEVICES=GPU-7c4f57e5-21e9-d1e6-c7c6-779b2624700d
PYTORCH_NVML_BASED_CUDA_CHECK=0 CUDA_VISIBLE_DEVICES=1
=> all: cuda_available False, count 0
```

## 2. Device identity (recorded)

| | GPU0 (broken) | GPU1 (healthy, unreachable through CUDA) |
|---|---|---|
| PCI | `0000:4b:00.0` | `0000:b1:00.0` |
| model | NVIDIA A100-SXM4-40GB | NVIDIA A100-SXM4-40GB |
| UUID | `GPU-5b51f95e-48cd-f1bd-287e-8bcb14f1ca5f` | `GPU-7c4f57e5-21e9-d1e6-c7c6-779b2624700d` |
| driver | 550.163.01 (`/proc/driver/nvidia/version`) | same kernel module |
| NVML query | `Unknown Error` | works, 14 MiB used by Xorg, 0 % util |
| PCI presence | `lspci` present; `/sys/.../enable` = 1 | present |

Environment: Python 3.12, `torch==2.5.1+cu124`, CUDA 12.4 wheel, `uv` frozen
sync OK, 104 CPU cores, 503 GB RAM, `/home` 469 GB free.

No process holds `/dev/nvidia0` (checked with `lsof`); no leftover repository
job is running; the user `hxy` has **no sudo** (`sudo: a password is
required`) and no docker group.  GPU0's driver state therefore cannot be
recovered from this account.

## 3. Required fix (root / administrator)

Any of the following (in increasing scope), all host-wide and disruptive to
other users, so they need the machine owner:

1. reset the failed device: `sudo nvidia-smi --gpu-reset -i 0` (GPU0 has no
   processes);
2. reload the driver stack:
   `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia`;
3. reboot the host.

After the fix, the acceptance check is:

```bash
ssh res 'CUDA_VISIBLE_DEVICES=1 uv run --directory /home/hxy/cy/research python -c "
import torch; print(torch.cuda.is_available(), torch.cuda.device_count())
p = torch.cuda.get_device_properties(0); print(p.name, p.uuid)"'
# expected: True 1 / NVIDIA A100-SXM4-40GB / GPU-7c4f57e5-...
```

## 4. Ready state and resume commands

Code, pre-registration and focused tests are committed locally (revision in
`STATE.yaml` / `research context`).  The formal round must run on the remote
physical **GPU1 as logical `cuda:0`** and must not touch GPU0.  The helper
scripts live in the runner skill; invoke them from there (they expect the
active repository as CWD unless noted):

```bash
SKILL=~/.pi/agent/skills/remote-research-runner

# local: deploy the committed revision (requires a clean worktree)
bash $SKILL/scripts/deploy.sh

# GPU1-only acceptance check (authoritative while GPU0 is broken)
ssh res 'CUDA_VISIBLE_DEVICES=1 uv run --directory /home/hxy/cy/research python -c "
import torch; print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_properties(0).name, torch.cuda.get_device_properties(0).uuid)"'
# expected: True 1 / NVIDIA A100-SXM4-40GB / GPU-7c4f57e5-...

# short stages (cache/verify/pca16/correctness/smoke) may use the foreground runner:
bash $SKILL/scripts/run_remote.sh 1 uv run research run zinc_e2e_dictenv_rolecorr_increment_v2 \
  --study zinc-context-gap --mode screen --purpose "Increment-v2 route1 primary screen" --set runtime.device=cuda

# the 320-epoch formal run is long: durable launch + wait + pull
bash $SKILL/scripts/launch_remote.sh 1 incv2-route1 \
  uv run research run zinc_e2e_dictenv_rolecorr_increment_v2 \
  --study zinc-context-gap --mode screen --purpose "Increment-v2 route1 primary screen" --set runtime.device=cuda
bash $SKILL/scripts/wait_remote.sh incv2-route1
bash $SKILL/scripts/pull_results.sh --light tracks/ksvd/results/e2e_dictenv_rolecorr_increment_v2
```

`$SKILL/scripts/preflight.sh` includes a two-GPU visibility check and will
fail while GPU0 is wedged; treat the GPU1-only acceptance check above as the
preflight until the host is repaired.

Required on-machine artifacts (RoleCorr-v1 read-only upstream): the frozen SDB
dictionary `results/sdb_v0/dictionary.pt` and shared env caches already exist
remotely; the correspondence caches, `standardizers.json`,
`dictionary_struct.pt` and `dictionary_corr.pt` are rebuilt/verified by the
`cache` / `verify` stages (the two small dictionaries and the scaler must be
synced with `scripts/sync_path.sh` unless the remote already has them; the
runner refuses to proceed if their SHA-256 does not match the frozen identity).

## 5. What was validated locally while blocked

* `uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_rolecorr_increment_v2.py`
  → 13 passed (geometry, base/appended-block bit-identities with the frozen
  CSSD / RoleCorr paths, binding extension layout, shared init, purity,
  block-shuffle, joint scaler/PCA, device resolver, test blocker).
* Full stage-runner integration dry run on local CPU with a temporary results
  directory and a 2-epoch toy budget (`/tmp/inc_smoke_results`, not the round
  results dir): cache → verify → pca16 → correctness → smoke → train →
  interventions → analysis all completed; correctness `all_passed: true`;
  smoke passed including the appended-binding gradient probe.

No formal run, no official-valid model selection and no official-test access
has happened for this round.
