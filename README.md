# LLaVA-NPU — Camera, YOLO26n and Nemotron Nano Omni

[日本語](READMEJ.md) · [Design handoff](HANDOFF.md) · [Technical notes](TECHNICAL.md)

The demo streams a USB camera or a local video playlist over HTTP MJPEG, overlays object detections, and displays Japanese captions. By default, the XDNA2 NPU runs YOLO26n through a Ryzen AI sidecar, and the Radeon GPU runs Nemotron through a ROCm build of llama.cpp. The optional GPU detection backend uses YOLO11m through Ultralytics.

## 1. Set up the application and GPU captioning

Use a Linux host with a supported AMD NPU, ROCm for the GPU, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a UVC camera or local video files. AMD's Ryzen AI 1.9 Linux instructions specify Ubuntu 24.04 LTS and kernel 6.10 or newer.

```bash
git clone --branch main --single-branch https://github.com/kotetsuy/LLaVA-NPU.git
cd LLaVA-NPU
sudo apt install tmux curl v4l-utils build-essential
uv sync --locked --inexact --extra webrtc
```

For Radeon 8060S (gfx1151), use the [ROCm 10 llama.cpp build instructions](GFX1151_ROCM10.md). For HX 370 (gfx1150), see the [branch-specific setup notes](HX370_ROCM10.md). Use the GPU build target for your hardware and use the NPU model paths and ports in this README for main.

Download the `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` and matching `mmproj-F16.gguf` from the [Nemotron GGUF repository](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF). The default directory for both is `~/nemotron-3/`. If you use another directory, set `vlm.model` and `vlm.mmproj` in `config.yaml`. Set `vlm.binary` to the path of `llama-mtmd-cli` in your build directory. The launcher looks for an executable `llama-server` in the same directory.

## 2. Prepare the YOLO26n ONNX and JSON pair

The ONNX and its matching JSON are local assets ignored by Git. A fresh checkout requires you to prepare both files. A public download location for the exact configured pair has not been verified. Obtain the pair through the maintainer or another authorized source, or generate a compatible pair as described below.

```yaml
yolo:
  backend: npu
  imgsz: 640
  npu:
    onnx: models/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx
    sidecar_url: http://127.0.0.1:8082
    port: 8082
```

Merge these settings into the existing `yolo` section. The default filenames are

```text
models/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx
models/yolo26n-640-split-v3-a16w8-perchannel-int16.json
```

If you received the pair, run this from the repository root after replacing `/path/to/authorized-pair` with its directory.

```bash
mkdir -p models
install -m 0644 "/path/to/authorized-pair/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx" models/
install -m 0644 "/path/to/authorized-pair/yolo26n-640-split-v3-a16w8-perchannel-int16.json" models/
python3 -m json.tool "models/yolo26n-640-split-v3-a16w8-perchannel-int16.json"
sha256sum "models/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx"
```

If the JSON includes `model_sha256`, compare it with `sha256sum`; otherwise use the checksum supplied by the source, if available. Keep both files adjacent and give them the same basename if you change `yolo.npu.onnx`. The launcher checks for the ONNX and exits if it is missing. The sidecar reads the same-stem JSON and rejects an INT16 model when the required input quantization metadata is missing or inconsistent.

### Generating a compatible pair

1. Obtain the upstream [YOLO26n detection weights](https://docs.ultralytics.com/models/yolo26/) and representative calibration images you are authorized to use. Record their versions or hashes and the tool versions.
2. Export a static batch-1 YOLO26n ONNX at 640×640. Extract the decoded XYXY boxes and post-sigmoid COCO class scores in that order, before the end-to-end Top-k. The sidecar expects shapes `[1,4,8400]` and `[1,80,8400]` and performs Top-k itself.
3. Quantize the split graph with AMD Quark using A16W8 and per-channel weights. The related project uses MinMax calibration and CLE. Follow its [export, split, quantization and INT16 conversion source](https://github.com/kotetsuy/yolodemo/blob/28145ab6c20744af77f5f6a0d085d7b6cefa3b1c/scripts/prepare_models.py) as a reference for the stages. It targets 448×448 and requires adaptation for 640×640, including graph node names and output shapes. Its separate CPU classification head is outside this sidecar's model interface.
4. Read the scalar scale and zero point from the input `QuantizeLinear` in the quantized graph before removing that node and exposing the external INT16 input `[1,3,640,640]`. Write a same-stem JSON object with `input_type` set to `"INT16"`, numeric `scale` from the conversion and integer `zero_point` from the conversion. Optionally record the final ONNX SHA-256 as `model_sha256`. Keep the values from that exact conversion.
5. Compare the split outputs before and after INT16 input conversion on representative images, using equivalent FP32 and externally quantized inputs. Check both output shapes and compare decoded detections against the CPU reference on the target NPU. Then place the pair in `models/` and set `yolo.npu.onnx` to its path.

Main provides no end-to-end 640×640 quantization recipe or calibration set. The exact original generation inputs and a bit-identical rebuild have not been verified. `scripts/export_yolo_onnx.py` is a regular ONNX exporter that defaults to the GPU model in `yolo.model`; it does not perform this split, quantization, or metadata generation.

## 3. Install and check Ryzen AI 1.9

1. Follow [AMD's Linux installation instructions](https://ryzenai.docs.amd.com/en/latest/linux.html) for Ryzen AI Software 1.9.0. Install the matching XRT 2.26.89 packages including the development headers, the AMD XDNA driver, and the Ryzen AI package in a dedicated Python 3.12 environment. Use `~/ryzenai_1_9/venv` as the installer target or export `RAI19_VENV` with the absolute venv path. Make custom paths available to the shell that runs the tmux sidecar too.
2. For Ubuntu 26.04, see the [maintainer's 1.9 upgrade instructions](https://github.com/kotetsuy/ryzenai_1_9). They cover an upgrade from a working 1.8 installation, including archive names, driver packages and the installer command. AMD's official Linux procedure targets Ubuntu 24.04 LTS.
3. Follow AMD's device permission setup and use [the maintainer's memlock setup example](https://qiita.com/kotetsu_yama/items/bb2ef5692b66ba87d90f) if the limit is low, then sign in again if you changed limits or group membership. Run the following as the normal launch user in a fresh shell. Confirm access to the NPU device and adequate soft and hard memlock limits. The launcher rejects the stock 8192 KiB hard limit.

For the extracted Ryzen AI 1.9.0 tarball, run the installer as shown below. The installer creates the venv.

```bash
# Run from the directory containing the extracted AMD installer.
# -a yes accepts the AMD installer license.
./install_ryzen_ai.sh -a yes -p "$HOME/ryzenai_1_9/venv"
```

```bash
(
  source scripts/rai_env.sh || exit 1
  test "$RAI_VERSION" = 1.9 || exit 1
  python -c 'import numpy, onnxruntime as ort; p = ort.get_available_providers(); print(ort.__version__, p); assert "VitisAIExecutionProvider" in p' || exit 1
  test -r /opt/xilinx/xrt/include/xrt/xrt_bo.h || exit 1
  PYTHONPATH="$PWD" python -c 'from src.npu_yolo.npu_output_sync import runtime_libraries, verify_libraries; print(verify_libraries(runtime_libraries()))'
)
ls -l /dev/accel/
ulimit -Sl
ulimit -Hl
/opt/xilinx/xrt/bin/xrt-smi examine
```

For the AMD tarball layout, run its supplied CNN quicktest and look for a positive `Actually running on NPU` count. Then check the LLaVA pipeline separately.

```bash
(
  source scripts/rai_env.sh || exit 1
  test "$RAI_VERSION" = 1.9 || exit 1
  cd "${RYZEN_AI_INSTALLATION_PATH}/quicktest" || exit 1
  python quicktest.py
)
```

`scripts/rai_env.sh` searches 1.9, 1.8 and 1.7.1 in order. The output synchronization shim accepts verified library hashes for 1.9 / XRT 2.26.89 and 1.8 / XRT 2.25.37. An unrecognized library combination stops the sidecar, including 1.7.1. The sidecar uses the Ryzen AI venv and the main application uses the project uv venv. Keep the vendor ONNX Runtime in the Ryzen AI venv.

## 4. Start and verify the pipeline

Connect a camera. With `camera.fallback: any`, an eligible unlisted camera can be selected. For a camera-free run, create a local M3U8 playlist with paths to playable video files and set `camera.playlist` in `config.yaml` (default `videos/playlist.m3u8`). The playlist is used when no eligible camera is available.

```bash
./start_all.sh --no-browser
tmux list-windows -t llava
tmux attach -t llava
```

Windows 0–3 are `capture`, `serve`, `vlm`, and `npu-yolo`. Use Ctrl-b then the window number to switch, or Ctrl-b d to detach. After model compilation and warmup, run the checks below from another terminal. The URLs use the current default ports.

```bash
curl -fsS http://127.0.0.1:9931/health
curl -fsS http://127.0.0.1:8082/health
curl -fsS http://127.0.0.1:8082/latest | python3 -m json.tool
/opt/xilinx/xrt/bin/xrt-smi examine
```

- Open `http://localhost:8080/` and confirm the video plays. Point the camera at a person or another recognizable object, or use a clip with one. Check that the boxes track the objects and that Japanese captions appear after the VLM loads.
- Check that sidecar `/health` returns `{"ready":true}`. Repeat `/latest` and look for `connected: true`, advancing `frame_seq` and a populated `boxes` array with the known object in view. A 503 from `/latest` means no frame has been published yet.
- Copy the NPU BDF from `xrt-smi examine`, then run the command below twice during live detection. Look for an active hardware context with increasing `Submissions` to confirm NPU execution.

```bash
read -r -p 'NPU BDF: ' npu_bdf
/opt/xilinx/xrt/bin/xrt-smi examine -d "$npu_bdf" -r all
```

YOLO26n throughput and startup time for this combined pipeline are unverified here. The sidecar logs measured `inf/s` while frames are being processed.

Stop with `./stop_all.sh`.

## Troubleshooting

- **No video**. Run `uv run --no-sync list-cameras` and `v4l2-ctl --list-devices`. Check camera priority and logs in the `capture` window, or check the playlist entries and paths.
- **No detections**. Check `npu-yolo` for model, metadata, VitisAI and synchronization errors. Initial compilation can delay readiness. The sidecar writes a local `*_ctx.onnx` cache and a `*_ctx.json` stamp next to the model, keyed to the model and runtime. Give the launch user write access to that directory.
- **No captions**. Check `vlm` for GGUF and ROCm errors, then check `serve` and `vlm.server.base_url`. The launcher passes `--reasoning off` to the VLM.
- **Session already exists**. Attach with `tmux attach -t llava` or stop the session using `./stop_all.sh` before restarting.

### Optional GPU detection with YOLO11m

Stop the running pipeline and set `yolo.backend: gpu`. In the project venv, install the ROCm PyTorch and torchvision builds for your GPU and ROCm release **before** installing the YOLO extra. Then verify HIP and restart. Ultralytics can fetch the configured `yolo11m.pt` on first use if it is not already local.

```bash
source scripts/rocm_env.sh
uv pip install -e '.[yolo]'
uv run --no-sync python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
./start_all.sh --no-browser
```

## License

See [LICENSE](LICENSE). The NPU output synchronization code is adapted from [yolodemo](https://github.com/kotetsuy/yolodemo) under AGPL-3.0-only. See the source SPDX headers and [AGPL-3.0 text](licenses/AGPL-3.0.txt).
