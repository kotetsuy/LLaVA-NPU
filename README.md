# LLaVA on ROCm — USB camera × YOLO11m × Nemotron Nano Omni × Chrome MJPEG

A demo that runs on the NucBox EVO X2 (Ryzen AI MAX+ 395 / Radeon 8060S, ROCm 7.2.1): live USB-camera video is streamed to Chrome over MJPEG (`multipart/x-mixed-replace`), with real-time YOLO11m bbox overlay (30 fps) and Japanese captions from Nemotron Nano Omni (0.5 fps).

> **Note on transport**: this used to be a WebRTC (aiortc) demo, but on fully-offline networks (Wi-Fi off / no Internet) Chrome refuses to emit a single ICE host candidate, leaving the connection stuck. We migrated to MJPEG over HTTP, which needs no ICE at all and works on the LAN unchanged.

For design details see [`HANDOFF.md`](./HANDOFF.md) and [`TECHNICAL.md`](./TECHNICAL.md).
日本語版は [`READMEJ.md`](./READMEJ.md) / [`HANDOFFJ.md`](./HANDOFFJ.md) / [`TECHNICALJ.md`](./TECHNICALJ.md).

---

## Requirements

| Item | Expected value |
|------|------|
| Machine | NucBox EVO X2 (AMD Ryzen AI MAX+ 395, gfx1151, 48 GB unified) |
| OS | Ubuntu 24.04.4 LTS (HWE kernel) |
| ROCm | 7.2.1 (symlinked at `/opt/rocm`) |
| Python | 3.12 |
| Package manager | `uv` (typically at `~/.local/bin/uv`) |
| USB camera | Any UVC-compliant device |
| Chrome | Any recent build (same machine or another LAN host) |

The following must already be installed:

- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `tmux` (`sudo apt install tmux`)
- ROCm 7.2.1 (`sudo apt install rocm` or AMD's official path)
- llama.cpp ROCm/HIP build (`~/llama.cpp/build/bin/llama-server` and `llama-mtmd-cli` already built)

---

## Setup steps

### 1. Clone the repository

```bash
git clone <this-repo-url> ~/LLaVA
cd ~/LLaVA
```

### 2. Python venv and base dependencies

```bash
uv venv
uv sync
```

This installs `numpy / opencv-python / pyyaml / pyudev` and gets you the state where Step 1 (USB → SHM) and Step 2 (hot-plug-aware CAL) work.

### 3. ROCm PyTorch (required from Step 3 onwards)

PyPI's `torch` is a CUDA build and won't work. Fetch the AMD ROCm wheels directly:

```bash
mkdir -p ~/wheels && cd ~/wheels
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
wget "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

cd ~/LLaVA
uv pip install ~/wheels/torch-*.whl ~/wheels/torchvision-*.whl \
               ~/wheels/torchaudio-*.whl ~/wheels/triton-*.whl
```

### 4. YOLO + ONNX (fallback plan)

```bash
uv pip install -e .[yolo,onnx]
```

This pulls in `ultralytics` (which auto-downloads YOLO11m on the first `predict`) and `onnx / onnxruntime` (CPU build).

### 5. Server dependencies (FastAPI + uvicorn + requests)

```bash
uv pip install -e .[webrtc]
```

The extra is still named `webrtc` for historical reasons; the current server streams MJPEG, so `aiortc` is installed but effectively unused.

### 6. ROCm environment variables

Add these to `~/.bashrc` so new shells pick them up automatically:

```bash
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
```

(`start_all.sh` re-exports these inside each tmux pane, so you're covered even if you forget to set them in your shell.)

> **Do not set `HSA_OVERRIDE_GFX_VERSION`.** The ROCm wheels and the llama.cpp
> build (`-DAMDGPU_TARGETS=gfx1151`) are native gfx1151 builds, so the override
> buys nothing — and a stale value copied from an old runbook is fatal: with
> `HSA_OVERRIDE_GFX_VERSION=11.0.0` the runtime reports `gfx1100` and every
> kernel launch fails (`HIP error: invalid device function`). `start_all.sh`
> explicitly `unset`s it in case a shell profile exports it.

### 7. Fetch the Nemotron Nano Omni GGUF

```bash
mkdir -p ~/nemotron-3
cd ~/nemotron-3

# unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF (Q4_K_XL)
huggingface-cli download \
  unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
  --local-dir Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF

huggingface-cli download \
  unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  mmproj-F16.gguf \
  --local-dir Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF
```

Around 24.5 GB total. Confirm that `vlm.model` / `vlm.mmproj` in `config.yaml` match these paths.

### 8. (Optional) confirm the GPU is visible

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True AMD Radeon Graphics
```

---

## Start / stop

### One-shot launch (recommended)

```bash
cd ~/LLaVA
./start_all.sh
```

This single command:

1. Creates the tmux session `llava`
2. window 0: `uv run capture-run` (USB → SHM, Steps 1+2)
3. window 1: `uv run serve` (FastAPI + MJPEG `/stream.mjpg` + YOLO bbox + VLM caption WS, Steps 6+7)
4. window 2: `llama-server --reasoning off` (resident Nemotron, Step 7b)
5. Polls `http://localhost:8080/` for up to 30 seconds
6. Opens Chrome automatically

Options:

```bash
./start_all.sh --no-browser     # over SSH or when you don't want auto-open
./start_all.sh --help
```

Attach to the session:

```bash
tmux attach -t llava            # watch logs
# Ctrl-b 0 / 1 / 2 to switch windows
# Ctrl-b d to detach but keep the session running
```

### Stop

```bash
./stop_all.sh
```

Sends `Ctrl-C` to each window, waits 5 seconds, then `tmux kill-session`. If anything survived, it falls through SIGINT → SIGKILL cleanup.

---

## What you should see in the browser

After `./start_all.sh`, open `http://localhost:8080/` in Chrome (or `http://<NucBox-IP>:8080/` from another LAN host):

- An `<img src="/stream.mjpg">` showing the live camera (1280×720, MJPEG)
- A semi-transparent `<canvas>` overlay with color-coded YOLO bboxes (30 fps; labels like `person 92%`)
- A semi-transparent caption box below with the Nemotron Japanese caption (~50 chars, refreshed every 2 s)
- A status row showing the stream state, bbox WS state, and the latest VLM `inference_ms` and `t/s`

Captions stay at `(no caption yet)` for the first ~10 s while llama-server loads the GGUF, then updates kick in.

---

## Step-by-step (for debugging)

If you want to bring components up one at a time without `start_all.sh`:

```bash
# Steps 1+2: capture
uv run capture-run                          # in another terminal
uv run shm-reader-demo --ticks 10           # confirm SHM read path
uv run shm-reader-demo --save /tmp/snap.jpg # save a single frame
uv run list-cameras                         # list /dev/v4l devices

# Step 3: YOLO standalone
uv run benchmark-yolo --source synthetic    # synthetic 1280x720 noise
uv run benchmark-yolo --source shm          # against the running capture-run
uv run export-yolo-onnx --verify            # ONNX fallback

# Step 4: VLM standalone (mtmd-cli subprocess)
uv run benchmark-vlm --image /tmp/snap.jpg

# Step 5: YOLO + VLM concurrent
uv run benchmark-concurrent --frames 600
uv run benchmark-concurrent --no-vlm        # baseline

# Steps 6+7: server only
uv run serve                                # ≈ T2
~/llama.cpp/build/bin/llama-server -m ... --mmproj ... --reasoning off  # ≈ T3
```

---

## Troubleshooting

### `uv run capture-run` doesn't see a camera

```bash
ls /dev/v4l/by-id              # is the USB camera symlinked?
v4l2-ctl --list-devices        # (sudo apt install v4l-utils)
```

`camera.preferred` in `config.yaml` is an ordered priority list — add one entry per camera (`by_id` glob or `vid_pid`) and the highest-ranked connected one is shown. If none match, `fallback: any` still picks something (`fallback: none` refuses unlisted cameras). Cameras can be hot-swapped while running: unplug the active one and it reconnects to the next listed camera; plug in a higher-priority camera and it switches over automatically (set `preempt: false` to keep the current feed).

### Browser shows a black image or status is stuck at `stream error`

Hit `http://localhost:8080/stream.mjpg` directly and confirm a 200 with MJPEG bytes:

- 200 + black frame → `capture-run` is probably stuck in SEARCHING. `tmux attach -t llava` and check window 0 (capture) for `-> CAPTURING dev=...` and `30 fps`. If absent, verify `config.yaml`'s `camera.preferred[].by_id` matches the symlink under `/dev/v4l/by-id/`.
- 404 / 500 → inspect the `serve` window (`tmux attach -t llava` → Ctrl-b 1).
- Can't reach it from another LAN host → open the firewall: `sudo ufw allow 8080`.

### Caption stays empty (`(no caption yet)`)

Either llama-server is still loading (~10 s the first time) or `--reasoning off` was forgotten. Check `serve`'s log via `tmux attach -t llava` → Ctrl-b 1:

```
vlm-runner: caption (1300ms) 'これは...'                  ← OK
vlm-runner: empty caption after strip; raw='<think>...'   ← --reasoning off missing
```

### `start_all.sh` fails with "session already exists"

```bash
./stop_all.sh                  # stop first
./start_all.sh                 # then restart
```

Or `tmux kill-session -t llava` to force-kill.

### Model load is unusually slow (30+ s the first time)

That's the time to pull the 21 GB GGUF off NVMe and into the page cache. Subsequent loads drop to ~10 s.

### Switching YOLO from fp16 back to fp32

Change `yolo.half: true` → `false` in `config.yaml`. fp32 has marginally better accuracy, but Step 5's concurrent benchmark showed fp16 leaves the VLM more headroom, so we default to fp16.

---

## Self-checks during development

```bash
# Byte-compile every Python file
python3 -m compileall -q src scripts && echo OK

# Module import smoke (also resolves dependency closure)
uv run python -c "from src.server.app import app; print('imports OK')"
```

---

## NPU backend (run YOLO11m on the XDNA2 NPU)

YOLO11m object detection can be moved off the GPU (ROCm, `yolo11m.pt`) onto the
**Strix Halo NPU (XDNA2, VitisAI EP, `yolo11m_a16w8.onnx`)**. The VLM (Nemotron),
camera, and MJPEG delivery are untouched. It is implemented as a sidecar (a separate
process talking local HTTP, exactly like the VLM's llama-server). Switch backends with
`yolo.backend` in `config.yaml`; rolling back to the GPU path is a one-liner.
See [`TECHNICAL.md`](./TECHNICAL.md) for architecture, design rationale, and on-device verification.

### Results summary (what now works)

| Item | Result |
|---|---|
| YOLO11m detection on the NPU | ✅ On bus.jpg, **4× person + 1× bus** (conf 0.89/0.89/0.89/0.75, bus 0.87) — matches yolotest's A16W8 |
| NPU offload proven | ✅ `xrt-smi` shows **HW Context=Active, Columns[0-7], rising Submissions** |
| Coordinate system (inverse letterbox) | ✅ bboxes map back to the input-frame coordinate system (≈1280×720); all boxes stay in-frame |
| Throughput | ✅ ~28–29 inf/s after warmup (on par with yolotest; the 30 fps camera is naturally decimated by "process latest frame") |
| Backend switch | ✅ `config.yaml` `yolo.backend: npu|gpu` — if npu misbehaves, just go back to gpu |
| Client compatibility | ✅ bbox JSON schema and `/ws/bbox` are unchanged (the browser canvas needs no edits) |

### Usage

#### Switch backend (`config.yaml`)
```yaml
yolo:
  backend: npu   # npu = XDNA2 NPU (VitisAI, separate process) / gpu = Ultralytics (ROCm, inside serve)
```

#### Start / stop
```bash
./start_all.sh          # with backend:npu, the npu-yolo window starts automatically too
./stop_all.sh
tmux attach -t llava    # Ctrl-b 0/1/2/3 = capture/serve/vlm/npu-yolo
```
Start order doesn't matter (serve retries over HTTP until the sidecar is up).

#### Verify (on real hardware)
```bash
# With the sidecar running, confirm NPU offload from another terminal
/opt/xilinx/xrt/bin/xrt-smi examine -d 0000:c6:00.1 -r all
#  → HW Context=Active, Columns[0-7], rising Submissions = running on the NPU
```
Open `http://localhost:8080/` in the browser and eyeball that people/objects sit inside
their bboxes. (The pass criterion is not "`Test Finished`" but "`xrt-smi` shows Active"
plus "boxes in the right place in the browser".)

### Operational notes (learned during implementation)

- **First compile ~20 s**: on the first launch in this repo, VitisAI compiles the
  quantized model, so the very first inference takes about 20 s. The sidecar only starts
  serving `/latest` **after warmup completes**, so serve simply waits (bbox is empty, then
  starts appearing). Subsequent runs are fast. `vaip_cache/` is added to `.gitignore` just in case.
- **Keep the venvs separate**: the sidecar runs under the RAI venv
  (`source setup_ryzenai_env.sh`); serve runs under the uv venv. `start_all.sh` sources the
  RAI env only in the sidecar window and does **not** apply the ROCm `ENV_PREFIX`
  (`ROCM_PATH` / `HIP_VISIBLE_DEVICES` — the NPU doesn't need them).
- **Sidecar launch command**: it runs under the RAI venv's python with `PYTHONPATH=<repo>`
  (so it can import `src.capture.shm_writer` / `src.npu_yolo.postprocess`). It is **not** `uv run`.
  `start_all.sh` assembles this form automatically.
- **Rollback**: if the NPU misbehaves, set `yolo.backend: gpu` in `config.yaml`. The sidecar
  is then not started and the legacy ultralytics/GPU path runs inside the serve process.

### What you need at runtime (the `~/yolotest` folder is not required)

- `~/ryzenai/ryzenai_venv` (onnxruntime-vitisai 1.23.3 / voe 1.7.1) + the XRT/NPU stack
- `models/yolo11m_a16w8.onnx` (already copied in)
- LLaVA's uv venv (the npu path uses `requests` = the webrtc extra)

`~/yolotest` is needed **only when re-quantizing** (below); normal operation never touches it.
Both the model and the pre/post-processing code are vendored into the repo, so LLaVA-NPU is
self-contained.

#### When you need to re-quantize (normally unnecessary)
Only if you want to rebuild the A16W8 model:
```bash
cd ~/ryzenai/ryzenai_venv && source setup_ryzenai_env.sh
python ~/yolotest/quantize_yolo11m_a16w8.py --input ~/yolo/yolo11m.onnx \
    --output ~/LLaVA-NPU/models/yolo11m_a16w8.onnx --calib-dir ~/yolotest/calib2
```

---

## NPU recovery / operations notes (after an environment upgrade)

Right after upgrading Ubuntu to **26.04** and ROCm to **7.14**, the NPU (`amdxdna`) can stop
working. Three independent problems were isolated and fixed until, at ordinary user
privilege, `xrt-smi examine` recognizes the **NPU Strix Halo (Firmware 1.1.2.65)**. See the
"NPU recovery" section of [`TECHNICAL.md`](./TECHNICAL.md) for root causes and the fix commands.

### Final verification (after reboot, as ordinary user `araki`) — ✅ all pass

```bash
ulimit -l                                              # → unlimited                     ✅
ls -l /dev/accel/accel0                                # crw-rw-rw-+ root render 261,0    ✅
source /opt/xilinx/xrt/setup.sh && xrt-smi examine     # [0000:c6:00.1] NPU Strix Halo    ✅
```

If all three pass without `sudo`, the YOLO11m NPU pipeline (`start_all.sh` /
`scripts/npu_yolo_sidecar.py`) can be run at ordinary user privilege.

---

## License

See [`LICENSE`](./LICENSE).
