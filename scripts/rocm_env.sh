#!/usr/bin/env bash
# Source before building/running native gfx1151 binaries on ROCm 10.
# ROCM_PATH may select another installation; never spoof the GPU architecture.
unset HSA_OVERRIDE_GFX_VERSION
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="$ROCM_PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
