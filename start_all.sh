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
URL="http://localhost:8080/"
SERVER_TIMEOUT=30  # seconds to wait for the server before opening the browser

# NPU YOLO sidecar (only started when config.yaml yolo.backend == npu). It runs
# under the Ryzen AI venv, which is the only env with the VitisAI EP + XRT.
# The RAI install path differs per machine (1.8 lives in ~/ryzenai_1_8 and ships
# no setup script; 1.7.1 had its own), so we source our own rai_env.sh, which
# picks whichever of the two is actually installed and working.
RAI_ENV="$PROJECT_DIR/scripts/rai_env.sh"

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
# HSA_OVERRIDE_GFX_VERSION must NOT be set: torch/llama.cpp here are native
# gfx1150 builds, so the override is at best a no-op and at worst fatal — a
# stale value copied from an old runbook (e.g. 11.0.0) makes the runtime report
# gfx1100 and every kernel launch fails. We unset it explicitly in case a shell
# profile exports it.
ENV_PREFIX='unset HSA_OVERRIDE_GFX_VERSION; export ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0; '

if ! command -v tmux >/dev/null; then
    echo "ERROR: tmux is not installed. Install with: sudo apt install tmux" >&2
    exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: project dir not found: $PROJECT_DIR" >&2
    exit 1
fi

# Sync once before any process starts; concurrent uv run commands can remove
# optional server dependencies from the shared environment.
cd "$PROJECT_DIR"
uv sync --locked --inexact --extra webrtc

# Use the same config as capture/serve, including this host's model locations.
# shlex.quote makes the generated shell assignments safe for paths with spaces.
CONFIG_VARS="$(uv run --no-sync python - <<'PY'
from pathlib import Path
from urllib.parse import urlsplit
import shlex
import yaml

cfg = yaml.safe_load(Path('config.yaml').read_text())
v = cfg['vlm']
y = cfg['yolo']
n = y.get('npu', {})
endpoint = urlsplit(v['server']['base_url'])
values = {
    'LLAMA_BIN': str(Path(v['binary']).expanduser().with_name('llama-server').resolve()),
    'VLM_MODEL': str(Path(v['model']).expanduser().resolve()),
    'VLM_MMPROJ': str(Path(v['mmproj']).expanduser().resolve()),
    'VLM_CTX': int(v.get('ctx_size', 8192)),
    'VLM_NGL': int(v.get('ngl', 99)),
    'VLM_PORT': endpoint.port or 8081,
    'VLM_HOST': endpoint.hostname or '127.0.0.1',
    'YOLO_BACKEND': y.get('backend', 'gpu'),
    'NPU_ONNX': str(Path(n.get('onnx', 'models/yolo11m_a16w8.onnx')).expanduser().resolve()),
    'NPU_PORT': int(n.get('port', 8082)),
    'URL': f"http://localhost:{cfg['server']['port']}/",
}
for key, value in values.items():
    print(f'{key}={shlex.quote(str(value))}')
PY
)"
eval "$CONFIG_VARS"

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

if [[ "$YOLO_BACKEND" == "npu" ]]; then
    if [[ ! -f "$RAI_ENV" ]]; then
        echo "ERROR: yolo.backend=npu but Ryzen AI env script not found at $RAI_ENV" >&2
        echo "       Set yolo.backend: gpu in config.yaml to use the GPU path instead." >&2
        exit 1
    fi
    # rai_env.sh picks Ryzen AI 1.8 or falls back to 1.7.1, and prints its own
    # diagnosis to stderr if neither is usable. Run it in a subshell here so we
    # fail before tmux starts, instead of leaving a dead npu-yolo window behind.
    if ! RAI_VERSION="$(bash -c 'source "$1" >/dev/null && printf "%s" "$RAI_VERSION"' _ "$RAI_ENV")"; then
        echo "ERROR: yolo.backend=npu but no usable Ryzen AI environment (see above)." >&2
        exit 1
    fi
    echo "NPU sidecar will use Ryzen AI $RAI_VERSION."
    if [[ ! -f "$NPU_ONNX" ]]; then
        echo "ERROR: NPU model not found at $NPU_ONNX" >&2
        echo "       Copy it: cp ~/yolotest/yolo11m_a16w8.onnx $PROJECT_DIR/models/" >&2
        exit 1
    fi
    # XRT allocates pinned buffers; with the stock 8 MB memlock limit every NPU
    # run dies with EAGAIN. ~/ryzenai_1_8/fix_memlock.sh raises it, but only for
    # terminals opened afterwards — so fail loudly here instead of in the sidecar.
    if [[ "$(ulimit -H -l)" == "8192" ]]; then
        echo "ERROR: memlock hard limit is still 8192 KB — XRT will fail with EAGAIN." >&2
        echo "       Run 'bash ~/ryzenai_1_8/fix_memlock.sh', then open a NEW terminal." >&2
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
tmux send-keys -t "$SESSION:capture" "${ENV_PREFIX}uv run --no-sync capture-run" C-m

# Give capture ~1s head-start so SHM is ready before yolo/vlm start probing.
sleep 1

# serve (FastAPI + aiortc + YoloRunner + VlmRunner). The webrtc extra carries
# fastapi/aiortc/uvicorn/requests — `uv run` without it syncs the env down to the
# base deps and serve dies with ModuleNotFoundError: fastapi.
tmux new-window -t "$SESSION:" -n serve -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:serve" "${ENV_PREFIX}uv run --no-sync serve" C-m

# llama-server (multimodal Nemotron). --reasoning off is required:
# without it the *-Reasoning model spends n_predict on thinking tokens.
tmux new-window -t "$SESSION:" -n vlm -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:vlm" "${ENV_PREFIX}${LLAMA_BIN} \
  -m '$VLM_MODEL' \
  --mmproj '$VLM_MMPROJ' \
  -c '$VLM_CTX' -ngl '$VLM_NGL' -np 1 --port '$VLM_PORT' --host '$VLM_HOST' --reasoning off" C-m

# npu-yolo (only for backend: npu). Runs under the Ryzen AI venv — do NOT apply
# ENV_PREFIX (ROCM_PATH etc. are for the GPU); scripts/rai_env.sh sets XRT.
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
