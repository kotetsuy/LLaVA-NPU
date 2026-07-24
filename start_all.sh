#!/usr/bin/env bash
# Start the full LLaVA pipeline in a tmux session named "llava".
# 3 windows (visible via `tmux attach -t llava`):
#   capture  -- USB camera -> SHM (Step 1+2)
#   serve    -- FastAPI + aiortc + YOLO bbox WS + VLM caption WS
#   vlm      -- llama-server (Nemotron-3 Nano Omni multimodal)
#
# Order doesn't matter — yolo-runner and vlm-runner inside `serve` retry
# until SHM and llama-server are reachable. We add a short stagger
# anyway so logs read top-to-bottom on first attach.

set -euo pipefail

SESSION=llava
# Resolve the repo root from this script's own location so the pipeline always
# runs against the checkout that contains this start_all.sh (this repo has the
# NPU sidecar + config.yaml; the older ~/LLaVA does not).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
VLM_MODEL="$HOME/nemotron-3/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf"
VLM_MMPROJ="$HOME/nemotron-3/mmproj-F16.gguf"
URL="http://localhost:8080/"
SERVER_TIMEOUT=30  # seconds to wait for the server before opening the browser

# NPU YOLO sidecar (only started when config.yaml yolo.backend == npu). It runs
# under the Ryzen AI venv, which is the only env with onnxruntime-vitisai + XRT.
RAI_ENV="$HOME/ryzenai/ryzenai_venv/setup_ryzenai_env.sh"

# CLI options
OPEN_BROWSER=1
for arg in "$@"; do
    case "$arg" in
        --no-browser) OPEN_BROWSER=0 ;;
        -h|--help)
            echo "Usage: $0 [--no-browser]"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ROCm env (mirrors CLAUDE.md whisperx setup; safe to set per-pane).
ENV_PREFIX='export HSA_OVERRIDE_GFX_VERSION=11.5.1 ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0; '

if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux is not installed. Install with: sudo apt install tmux" >&2
    exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: project dir not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-server not found at $LLAMA_BIN" >&2
    exit 1
fi
if [[ ! -f "$VLM_MODEL" ]]; then
    echo "ERROR: VLM model not found at $VLM_MODEL" >&2
    exit 1
fi
if [[ ! -f "$VLM_MMPROJ" ]]; then
    echo "ERROR: mmproj not found at $VLM_MMPROJ" >&2
    exit 1
fi

# Read the YOLO backend + NPU settings from config.yaml (pyyaml is a core dep,
# so `uv run` always has it). Defaults keep us on the GPU path if parsing fails.
CONFIG_LINE="$(cd "$PROJECT_DIR" && uv run python -c '
import yaml
y = yaml.safe_load(open("config.yaml")).get("yolo", {})
npu = y.get("npu", {})
print(y.get("backend", "gpu"), npu.get("onnx", "models/yolo11m_a16w8.onnx"), npu.get("port", 8082))
' 2>/dev/null)"
[[ -z "$CONFIG_LINE" ]] && CONFIG_LINE="gpu - -"
read -r YOLO_BACKEND NPU_ONNX NPU_PORT <<< "$CONFIG_LINE"

if [[ "$YOLO_BACKEND" == "npu" ]]; then
    if [[ ! -f "$RAI_ENV" ]]; then
        echo "ERROR: yolo.backend=npu but Ryzen AI env not found at $RAI_ENV" >&2
        echo "       Set yolo.backend: gpu in config.yaml to use the GPU path instead." >&2
        exit 1
    fi
    if [[ ! -f "$PROJECT_DIR/$NPU_ONNX" ]]; then
        echo "ERROR: NPU model not found at $PROJECT_DIR/$NPU_ONNX" >&2
        echo "       Copy it: cp ~/yolotest/yolo11m_a16w8.onnx $PROJECT_DIR/models/" >&2
        exit 1
    fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' is already running."
    echo "       Run ./stop_all.sh first, or attach: tmux attach -t $SESSION"
    exit 1
fi

# capture
tmux new-session -d -s "$SESSION" -n capture -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:capture" "${ENV_PREFIX}uv run capture-run" C-m

# Give capture ~1s head-start so SHM is ready before yolo/vlm start probing.
sleep 1

# serve (FastAPI + aiortc + YoloRunner + VlmRunner). The webrtc extra carries
# fastapi/aiortc/uvicorn/requests — `uv run` without it syncs the env down to the
# base deps and serve dies with ModuleNotFoundError: fastapi.
tmux new-window -t "$SESSION:" -n serve -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:serve" "${ENV_PREFIX}uv run --extra webrtc serve" C-m

# llama-server (multimodal Nemotron). --reasoning off is required:
# without it the *-Reasoning model spends n_predict on thinking tokens.
tmux new-window -t "$SESSION:" -n vlm -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:vlm" "${ENV_PREFIX}${LLAMA_BIN} \
  -m '$VLM_MODEL' \
  --mmproj '$VLM_MMPROJ' \
  -c 8192 -ngl 99 --port 8081 --host 127.0.0.1 --reasoning off" C-m

# npu-yolo (only for backend: npu). Runs under the Ryzen AI venv — do NOT apply
# ENV_PREFIX (HSA_OVERRIDE etc. are for the GPU); setup_ryzenai_env.sh sets XRT.
# PYTHONPATH lets the sidecar import src.capture.shm_writer (pure-python SHM).
WINDOWS_MSG="3 windows: capture, serve, vlm"
SWITCH_MSG="Ctrl-b 0 (capture), Ctrl-b 1 (serve), Ctrl-b 2 (vlm)"
if [[ "$YOLO_BACKEND" == "npu" ]]; then
    tmux new-window -t "$SESSION:" -n npu-yolo -c "$PROJECT_DIR"
    tmux send-keys -t "$SESSION:npu-yolo" \
      "source '$RAI_ENV' && PYTHONPATH='$PROJECT_DIR' \
python '$PROJECT_DIR/scripts/npu_yolo_sidecar.py' --model '$NPU_ONNX' --port $NPU_PORT" C-m
    WINDOWS_MSG="4 windows: capture, serve, vlm, npu-yolo"
    SWITCH_MSG="Ctrl-b 0 (capture), 1 (serve), 2 (vlm), 3 (npu-yolo)"
fi

cat <<EOF
Started tmux session '$SESSION' with $WINDOWS_MSG.

  Attach :  tmux attach -t $SESSION
  Switch :  $SWITCH_MSG
  Detach :  Ctrl-b d
  Stop   :  ./stop_all.sh

EOF

# Open Chrome once the server is reachable. Caption may still take ~10s
# more after this point — the page just renders "(no caption yet)" until
# llama-server finishes loading the GGUF.
if [[ "$OPEN_BROWSER" -eq 1 ]]; then
    echo -n "waiting for $URL "
    READY=0
    for ((i = 0; i < SERVER_TIMEOUT; i++)); do
        if curl -sf -o /dev/null --max-time 1 "$URL"; then
            READY=1
            break
        fi
        echo -n "."
        sleep 1
    done
    echo
    if [[ "$READY" -eq 0 ]]; then
        echo "WARNING: server didn't respond within ${SERVER_TIMEOUT}s; check 'tmux attach -t $SESSION'."
        echo "Open manually: $URL"
        exit 0
    fi

    if command -v google-chrome >/dev/null 2>&1; then
        google-chrome "$URL" >/dev/null 2>&1 &
        echo "opened in google-chrome."
    elif command -v chromium >/dev/null 2>&1; then
        chromium "$URL" >/dev/null 2>&1 &
        echo "opened in chromium."
    elif command -v chromium-browser >/dev/null 2>&1; then
        chromium-browser "$URL" >/dev/null 2>&1 &
        echo "opened in chromium-browser."
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL" >/dev/null 2>&1 &
        echo "opened via xdg-open (default browser)."
    else
        echo "no browser command found; open $URL manually."
    fi
else
    echo "Browser auto-open skipped (--no-browser). Open: $URL"
fi
