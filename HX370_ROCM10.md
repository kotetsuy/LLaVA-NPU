# HX370 / gfx1150 / ROCm 10 動作記録

確認日: 2026-09-17。ブランチ: `hx370-gfx1150`。

[参考記事](https://qiita.com/kotetsu_yama/items/542829bcb0056b3dc48a)の
カメラ → SHM → NPU物体検出 / GPU画像説明 → MJPEG・WebSocket配信を、
このホストの導入済み環境で動作確認した。

## 現在の状態

2026-09-17、ユーザーがブラウザで動作確認を完了し、`./stop_all.sh` で停止した。
その後、ホスト側で以下を確認した。現在パイプラインは停止中。

- tmux: サーバーは起動しておらず、`llava` セッションの残存なし。
- LLaVA-NPUの関連プロセス（capture / serve / llama-server / npu-yolo）: 残存なし。
- TCPポート8080 / 8081 / 8082: 待受なし。
- NPU: `xrt-smi` は `No hardware contexts running on device`。
- カメラ `/dev/video0`・`/dev/video1`: 使用プロセスなし。
- `/dev/kfd`・`/dev/accel/accel0`: 使用プロセスなし。
- 共有メモリ `/dev/shm/webcam_latest`: 削除済み。
- `/dev/dri/renderD128` はデスクトップ環境とChrome等が使用しているが、
  LLaVA-NPUの推論プロセスは残っていない。

停止処理は正常に完了している。再開する場合は、下記の起動手順を実行する。

## 実機環境

| 項目 | 確認値 |
|---|---|
| ホスト | SER9 / AMD Ryzen AI 9 HX 370 |
| OS | Ubuntu 26.04.1 LTS / kernel 7.0.0-31-generic |
| GPU | Radeon 890M / gfx1150 |
| ROCm | 10.0.0 (`/opt/rocm/core-10.0`) |
| HIP | 7.15.26333（ROCm製品バージョンとは別） |
| NPU | RyzenAI-npu4 / `0000:c5:00.1` / XDNA2 |
| Ryzen AI / XRT | 1.8 / 2.25.37 |
| ONNX Runtime | 1.27.0 / VitisAIExecutionProvider |
| Python | uv管理の3.12.14 |
| memlock | unlimited |
| カメラ | Jieli Technology USB PHY 2.0 / `1124:2925` / `/dev/video0` |

GPU/NPUドライバ、モデル、llama.cppは既存資産を使用した。
追加のシステムパッケージは `tmux`（ユーザーがsudoでインストール）。

## このチェックアウトの設定

- `config.yaml` のモデルパスを `/home/test` の実際の配置に合わせた。
- NPUモデル: `/home/test/yolotest/yolo11m_a16w8.onnx`。
- GGUF本体とmmproj: `/home/test/nemotron-3/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF/`。
- llama.cpp: `/home/test/llama.cpp/build/bin/`。gfx1150向けROCmビルドを使用。
- `.python-version` でPython 3.12を指定。
- `start_all.sh` はモデル・コンテキスト・GPUレイヤー数・ポートを
  `config.yaml` から読み、llama-serverを1スロット・8192トークンで起動する。
- 起動前に `uv sync --locked --inexact --extra webrtc` を一度実行し、
  各プロセスは `uv run --no-sync` で起動する。
  `--inexact` は別途導入した追加パッケージを保持する。
- `HSA_OVERRIDE_GFX_VERSION` は解除し、`HIP_VISIBLE_DEVICES=0` を使用。
- 本体は `.venv`、NPUサイドカーは `/home/test/ryzenai_1_8/venv` に分離。
  このNPU構成では本体にPyTorchは不要。GPU版YOLO経路は今回の検証対象外。

## 起動・確認・停止

```bash
cd /home/test/LLaVA-NPU
./start_all.sh --no-browser
```

ブラウザで <http://localhost:8080/> を開く。
`./start_all.sh` ならブラウザも自動起動する。
既に `llava` セッションが動いている場合は、重複起動せず既存画面を開く。

```bash
tmux attach -t llava
# Ctrl-b 0: capture / 1: serve / 2: vlm / 3: npu-yolo
# Ctrl-b d: 接続を抜け、実行は継続

curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8082/latest
/opt/xilinx/xrt/bin/xrt-smi examine -d 0000:c5:00.1 -r aie-partitions

# 停止
./stop_all.sh
```

NPU初回コンパイルは約27秒だった。
`/home/test/yolotest/yolo11m_a16w8_ctx.onnx` と対応するJSONスタンプが生成される。
以後はソースモデル・Ryzen AI・ONNX Runtimeのスタンプが一致するとキャッシュを使用する。

## 実測と検証範囲

- カメラ: MJPG、1280×720、約30fps。
- NPU物体検出: 約24〜25 inf/s。実映像から `person` 等の検出を取得。
- `xrt-smi`: HW ContextがActive、Columns 0〜7、Submissionsが2634から3816へ増加。
- VLM: 日本語キャプション生成に約2.8〜4.5秒。
  完了後に2秒待つため、更新周期は約5〜7秒であり、記事の0.5fpsとは異なる。
- 実行中llama-serverが `/dev/kfd` と `/dev/dri/renderD128` を開き、
  `/opt/rocm/core-10.0/lib/` のHIP/HSAライブラリをロードしていることを確認。
- WebトップページはHTTP 200。MJPEGからJPEGを復号し720×1280×3を確認。
- `/ws/bbox` と `/ws/caption` をそれぞれ2回受信し、フレーム番号の前進を確認。
  bboxの座標範囲と、日本語の非空キャプションも確認した。
- 既存テストは13件成功。シェル構文・差分の空白チェックも成功。

ユーザーによるブラウザでの動作確認も完了。
bbox位置やキャプション内容の正確性について、個別の評価結果は記録していない。

テストを再実行する場合:

```bash
uv sync --locked --inexact --extra webrtc --extra dev
uv run --no-sync pytest -q
```
