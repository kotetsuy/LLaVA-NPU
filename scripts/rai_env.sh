#!/usr/bin/env bash
# Source this to get a working Ryzen AI environment (VitisAI EP + XRT).
#
#     source scripts/rai_env.sh
#
# Two installations are supported, tried in this order:
#
#   1. Ryzen AI 1.8  — venv at ~/ryzenai_1_8/venv        (override: RAI18_VENV)
#   2. Ryzen AI 1.7.1 — ~/ryzenai/ryzenai_venv/setup_ryzenai_env.sh
#                                                        (override: RAI171_SETUP)
#
# 1.8 wins when both are present. A machine that upgraded to 1.8 still has the
# 1.7.1 directory (and its setup script) lying around, but the venv behind it is
# dead — so "the file exists" is not evidence that 1.7.1 works. The 1.7.1 branch
# is kept only for machines that repaired 1.7.1 instead of upgrading.
# On success, RAI_VERSION is exported ("1.8" / "1.7.1").
#
# Prerequisite: the memlock hard limit must not be 8192 KB, or XRT fails with
# EAGAIN. ~/ryzenai_1_8/fix_memlock.sh raises it; it only takes effect in a
# terminal opened afterwards.

RAI18_VENV="${RAI18_VENV:-$HOME/ryzenai_1_8/venv}"
RAI171_SETUP="${RAI171_SETUP:-$HOME/ryzenai/ryzenai_venv/setup_ryzenai_env.sh}"

if [[ -f "$RAI18_VENV/bin/activate" ]]; then
    # Ryzen AI 1.8 ships no setup script of its own, so reproduce the
    # LD_LIBRARY_PATH construction from ~/ryzenai_1_8/run_quicktest.sh.
    set +u
    source /opt/xilinx/xrt/setup.sh >/dev/null
    source "$RAI18_VENV/bin/activate"

    export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib/:${LD_LIBRARY_PATH:-}
    # libonnxruntime_vitisai_ep.so NEEDs libpeano-lib.so.21.0git, which ships only
    # under site-packages/lnx64.o/tools/peano/lib and is not on the documented
    # LD_LIBRARY_PATH. Without this the VitisAI EP silently falls back to CPU.
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${RYZEN_AI_INSTALLATION_PATH}/lib/python3.12/site-packages/lnx64.o/tools/peano/lib
    # The venv's activate puts voe/lib ahead of /opt/xilinx/xrt/lib, and voe/lib
    # ships a stale libxrt_coreutil.so.2.19.184. Loading that makes XRT 2.25.37's
    # libxrt_core.so.2 fail with "undefined symbol: xrt_core::smi::get_option_options"
    # -> "Failed to create runner" -> abort. Keep the installed XRT libs first.
    export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
    set -u
    export RAI_VERSION=1.8

elif [[ -f "$RAI171_SETUP" ]]; then
    # The 1.7.1 venv was built with `python3.12 -m venv --copies` from the
    # distro's python. An Ubuntu 26.04 upgrade removes the python3.12 package,
    # so the copied interpreter loses its stdlib and cannot boot at all
    # ("No module named 'encodings'"). Catch that here instead of letting the
    # sidecar die 20 lines into an unrelated-looking traceback.
    _rai171_python="$(dirname "$RAI171_SETUP")/venv/bin/python"
    if ! "$_rai171_python" -c '' 2>/dev/null; then
        echo "ERROR: the Ryzen AI 1.7.1 interpreter at $_rai171_python cannot start." >&2
        echo "       It was copied from a /usr/bin/python3.12 that no longer exists." >&2
        echo "       Either install Ryzen AI 1.8 (https://github.com/kotetsuy/ryzenai_1_8)," >&2
        echo "       or rebuild the venv against a standalone 3.12 (TECHNICAL.md 9.3)." >&2
        unset _rai171_python
        return 1 2>/dev/null || exit 1
    fi
    unset _rai171_python
    set +u
    source "$RAI171_SETUP"
    set -u
    export RAI_VERSION=1.7.1

else
    echo "ERROR: no Ryzen AI installation found." >&2
    echo "       Looked for 1.8   at $RAI18_VENV/bin/activate" >&2
    echo "       Looked for 1.7.1 at $RAI171_SETUP" >&2
    echo "       Set yolo.backend: gpu in config.yaml to use the GPU path instead." >&2
    return 1 2>/dev/null || exit 1
fi
