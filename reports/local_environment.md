# Local Environment Report

**Date / Timestamp:** 2026-10-23 22:13 UTC+1  
**Workspace:** `/home/abolarinwa-sa/Documents/Smol-VLAxYOLO`  
**Git Remote (`origin`):** `https://github.com/SamuelAbolarinwa/smol-vla_mod.git` (branch: `main`)

---

## 1. Operating System and Architecture

- **OS:** Ubuntu 24.04.4 LTS (Noble Numbat)
- **Kernel:** Linux `7.0.0-38-generic` #38~24.04.4-Ubuntu SMP PREEMPT_DYNAMIC
- **Architecture:** `x86_64` (64-bit)

---

## 2. Hardware Resources

### CPU
- **Model:** Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz
- **Sockets / Cores / Threads:** 1 socket, 8 physical cores, 16 logical threads (2 threads per core)

### Memory & Swap
- **Total Physical RAM:** 15 GiB
- **Used RAM:** ~4.9 GiB
- **Available RAM:** ~10 GiB
- **Total Swap:** 4.0 GiB (380 KiB used, 4.0 GiB free)

### Storage (Disk Space)
- **Root Filesystem (`/`):** `/dev/nvme0n1p5`
- **Total Size:** 208 GiB
- **Used:** 190 GiB (96% utilization)
- **Available Free Space:** **9.4 GiB**

> **Note on Storage:** Free disk space is critically constrained (~9.4 GiB). Local downloading of massive datasets or multi-gigabyte foundation model checkpoints must be avoided locally, strictly adhering to the project compute plan (using Kaggle for heavy GPU and storage workloads).

---

## 3. GPU and Acceleration (`nvidia-smi`)

- **GPU Device:** NVIDIA Quadro T2000 with Max-Q Design
- **VRAM Total:** 4096 MiB (4.0 GiB)
- **Driver Version:** `580.178.04`
- **CUDA Version (Driver Reported):** `13.0`
- **Power Usage / Cap:** 1W / 30W
- **Current Temperature:** 47°C
- **Active Processes:** `gnome-shell` (~1 MiB), `snapd-desktop-integration` (~1 MiB)

---

## 4. Python Runtimes and Environment Managers

### Python Executables Available
- `python3` / `python3.12`: Python 3.12.3 (`/usr/bin/python3`, `/usr/bin/python3.12`)
- `python3.11`: Python 3.11.15 (`/usr/bin/python3.11`)
- `python3.10`: Python 3.10.20 (`/usr/bin/python3.10`)

### Environment Managers
- **`venv` (`python3 -m venv`):** Available (standard library).
- **`uv`:** Not installed / not in PATH.
- **`conda` / `mamba` / `pixi`:** Not installed / not in PATH.
- **Existing virtual environments:** None in workspace (`.venv` absent); none in `~/.virtualenvs` or `~/.conda/envs`.

---

## 5. Development Tools

- **Git:** `git version 2.43.0` (active repo branch: `main`, origin configured)
- **FFmpeg:** `ffmpeg version 6.1.1-3ubuntu5` (built with gcc 13, libx264, libvpx, libwebp, etc.)

---

## 6. Constraints & Compute Plan Alignment

1. **Development & Lightweight Execution:** The local machine is capable of code editing, AST/static analysis, unit tests, and lightweight configuration tests.
2. **Kaggle GPU Delegation:** Local GPU VRAM (4 GB Quadro T2000) and available local disk (~9.4 GB) are insufficient for full SmolVLA training or large batch evaluations. Full model training and evaluation runs must be dispatched to Kaggle.
3. **No Architecture Downscaling:** The research architecture (SmolVLA + YOLO) will remain uncompromised (no artificial quantisation or layer stripping for the sake of local hardware).
