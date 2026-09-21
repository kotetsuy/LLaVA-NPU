# ROCm 10 / gfx1151 での起動

[参考記事](https://qiita.com/kotetsu_yama/items/fa1a533293f880234298)の
GPU画像説明＋NPU物体検出の構成を、Ryzen AI MAX+ 395 / Radeon 8060S
（gfx1151）向けに適用する。記事のHX 370はgfx1150なので、ビルドターゲットを
そのままコピーしない。HX 370での測定値はこの機種の測定値ではない。

## 前提とモデル配置

- ROCm 10、gfx1151対応のllama.cpp、Ryzen AI 1.8 / XRTを導入済み。
- Python 3.12は `.python-version` で指定。本体とNPU用venvを分離する。
- NPU用venvは `~/ryzenai_1_8/venv`（変更時は `RAI18_VENV`）。
- `config.yaml` の `~` は実行ユーザーのホームに展開される。

```text
~/llama.cpp/build/bin/llama-server
~/llama.cpp/build/bin/llama-mtmd-cli
~/nemotron-3/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
~/nemotron-3/mmproj-F16.gguf
~/yolotest/yolo11m_a16w8.onnx
```

別の配置を使う場合は `config.yaml` の `vlm.binary/model/mmproj` と
`yolo.npu.onnx` を変更する。カメラの優先順位も同ファイルで設定する。
`fallback: any` なら優先カメラがなくても接続中のカメラを選ぶ。
既定の `yolo.backend: npu` では本体venvにPyTorchは不要。
GPU版YOLOへ変更する場合だけ、別途gfx1151対応のROCm版PyTorchと
Ultralyticsを導入する。旧READMEのROCm 7.2.1 wheelはこの手順では使用しない。

## llama.cppを再ビルドする場合

既存バイナリがgfx1151を認識するなら再ビルドは不要。
記事で使用したコミットは `d4365d955488d21688c0a4d8d06bcfd68225ad01`。
以下は対応ソースを `~/llama.cpp` に用意した後の例。

```bash
source scripts/rocm_env.sh
export HIPCXX="$ROCM_PATH/lib/llvm/bin/clang++"
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build "$HOME/llama.cpp/build" --config Release -j 4 \
  --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
"$HOME/llama.cpp/build/bin/llama-cli" --list-devices
```

`scripts/rocm_env.sh` は古い `HSA_OVERRIDE_GFX_VERSION` を解除し、ROCmの
実行・ライブラリ検索パスを設定する。起動スクリプトも同じ環境を使用する。
ROCm製品版は `/opt/rocm/core-10.0/.info/version` で確認する。
HIPのバージョン番号が7.xでも、ROCm製品版とは別なので異常とは限らない。

## 起動・確認・停止

```bash
uv sync --locked --inexact --extra webrtc
./start_all.sh --no-browser
# ブラウザで http://localhost:8080/ を開く
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8082/latest
tmux attach -t llava
# Ctrl-b d で抜ける
./stop_all.sh
```

NPU側は `scripts/rai_env.sh` がXRT・Peanoのライブラリを設定する。
memlockとNPUドライバーの確認方法は参考記事を参照する。
NPUのBDFは `xrt-smi examine` で確認し、記事の `0000:c5:00.1` を固定しない。
初回コンパイル後、映像、検出枠、日本語字幕、NPUの実行カウンターを確認する。

## この変更時の確認（2026-09-21）

- ROCm製品版: 10.0.0。
- 既存llama.cppのCMakeキャッシュ: `AMDGPU_TARGETS=gfx1151`。
- ホスト側の `llama-cli --list-devices`: Radeon 8060S、49152 MiB。
- Ryzen AI環境: ONNX Runtime 1.27.0、VitisAIExecutionProviderを列挙。

プロバイダーの列挙はNPU推論成功の証明ではない。
この変更ではカメラからの統合推論・ブラウザ表示の実機検証は未実施。
以前のHX 370の動作記録は [HX370_ROCM10.md](HX370_ROCM10.md) に保持する。
