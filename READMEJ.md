# LLaVA-NPU ― カメラ・YOLO26n・Nemotron Nano Omni

[English](README.md) · [設計引き継ぎ](HANDOFFJ.md) · [技術資料](TECHNICALJ.md)

USBカメラまたはローカル動画のプレイリストをHTTP MJPEGで配信し、検出枠と日本語キャプションを重ねるデモです。既定ではRyzen AIのサイドカーがXDNA2 NPUでYOLO26nを実行し、ROCm版llama.cppがRadeon GPUでNemotronを実行します。物体検出をGPUに切り替えた場合はUltralyticsのYOLO11mを使います。

## 1. アプリとGPUキャプションの準備

対応するAMD NPUとROCm対応GPUを備えるLinuxマシン、Python 3.12、[uv](https://docs.astral.sh/uv/getting-started/installation/)、UVCカメラまたはローカル動画を用意します。AMDのRyzen AI 1.9向けLinux手順ではUbuntu 24.04 LTSとカーネル6.10以上が前提です。

```bash
git clone --branch main --single-branch https://github.com/kotetsuy/LLaVA-NPU.git
cd LLaVA-NPU
sudo apt install tmux curl v4l-utils build-essential
uv sync --locked --inexact --extra webrtc
```

Radeon 8060S（gfx1151）は[ROCm 10でのllama.cppビルド手順](GFX1151_ROCM10.md)、HX 370（gfx1150）は[ブランチ固有のセットアップ情報](HX370_ROCM10.md)を参照してください。GPUのビルドターゲットは機種に合わせ、mainのNPUモデルパスとポートはこのREADMEに従ってください。

[Nemotron GGUFリポジトリ](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF)から `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf` と対応する `mmproj-F16.gguf` を取得します。既定の配置先は両方とも `~/nemotron-3/` です。別の場所に置く場合は `config.yaml` の `vlm.model` と `vlm.mmproj` を変更します。`vlm.binary` はビルド先の `llama-mtmd-cli` を指すように設定します。起動スクリプトは同じディレクトリにある実行可能な `llama-server` を使います。

## 2. YOLO26nのONNXとJSONを準備

ONNXと対応するJSONはローカルに配置するファイルで、Gitの管理対象外です。新規クローンでは両方の準備が必要です。既定の組み合わせを入手できる公開配布先は確認できていません。メンテナーなど正規の提供元から一組で受け取るか、以下を参考に互換モデルを生成してください。

```yaml
yolo:
  backend: npu
  imgsz: 640
  npu:
    onnx: models/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx
    sidecar_url: http://127.0.0.1:8082
    port: 8082
```

上記は既存の `yolo` セクションに反映します。既定の配置先は次の2ファイルです。

```text
models/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx
models/yolo26n-640-split-v3-a16w8-perchannel-int16.json
```

一組で受け取った場合は、`/path/to/authorized-pair` を実際の入手先ディレクトリに置き換え、リポジトリのルートで実行します。

```bash
mkdir -p models
install -m 0644 "/path/to/authorized-pair/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx" models/
install -m 0644 "/path/to/authorized-pair/yolo26n-640-split-v3-a16w8-perchannel-int16.json" models/
python3 -m json.tool "models/yolo26n-640-split-v3-a16w8-perchannel-int16.json"
sha256sum "models/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx"
```

JSONに `model_sha256` があれば `sha256sum` の結果と照合します。記載がなければ、提供元が公開したチェックサムがある場合に照合してください。`yolo.npu.onnx` を変更する場合も、同じディレクトリにベース名が同じJSONを置きます。起動スクリプトはONNXが見つからないと終了します。サイドカーは同名のJSONを読み、INT16入力に必要な量子化情報が欠けていたり、入力型と一致しなかったりすると起動に失敗します。

### 互換モデルを生成する場合

1. [公式のYOLO26n物体検出用重み](https://docs.ultralytics.com/models/yolo26/)と、利用許可のある代表的な校正画像を用意します。重みと画像の版またはハッシュ、ツールの版を記録します。
2. バッチ1・640×640固定でYOLO26nをONNXにエクスポートし、end-to-endのTop-kより手前で、デコード済みのXYXY座標とSigmoid適用後のCOCOクラススコアをこの順で切り出します。サイドカーが要求する出力形状は `[1,4,8400]` と `[1,80,8400]` で、Top-kはサイドカーが処理します。
3. 分割したONNXをAMD QuarkでA16W8・重みper-channelに量子化します。関連プロジェクトはMinMax校正とCLEを使用しています。[yolodemoのエクスポート・分割・量子化・INT16変換コード](https://github.com/kotetsuy/yolodemo/blob/28145ab6c20744af77f5f6a0d085d7b6cefa3b1c/scripts/prepare_models.py)で各工程を確認できます。このコードは448×448用なので、グラフのノード名や出力形状を含め640×640向けに調整します。CPU分類ヘッドを別ファイルにする構成は、このサイドカーのモデル仕様には含まれません。
4. 量子化後のグラフで入力の `QuantizeLinear` からスカラーのscaleとzero pointを取得します。そのノードを外して外部INT16入力 `[1,3,640,640]` を公開し、ONNXと同名のJSONに `input_type` として `"INT16"`、`scale` として取得した数値、`zero_point` として取得した整数を保存します。最終ONNXのSHA-256は任意で `model_sha256` に記録できます。量子化パラメーターは変換したモデルに対応する値を使います。
5. 代表画像について、変換前のFP32入力と変換後の外部量子化入力で、両方の出力形状と結果が一致することを確認します。対象NPUで実行した検出結果もCPU参照と比較します。検証後、ペアを `models/` に置き、`yolo.npu.onnx` をそのパスに設定します。

mainには640×640用の一括量子化レシピや校正画像一式がありません。既定ファイルを生成した入力一式と、バイト単位で一致する再生成手順は未確認です。`scripts/export_yolo_onnx.py` は通常のONNXエクスポーターで、既定では `yolo.model` のGPU用モデルを読みます。出力分割・量子化・JSON生成には対応していません。

## 3. Ryzen AI 1.9の導入と確認

1. [AMDのLinuxインストール手順](https://ryzenai.docs.amd.com/en/latest/linux.html)に従い、Ryzen AI Software 1.9.0、開発ヘッダーを含むXRT 2.26.89の関連パッケージとAMD XDNAドライバーを導入します。Ryzen AIはPython 3.12の専用環境に導入します。インストーラーの既定の配置先として `~/ryzenai_1_9/venv` を指定するか、別の場所ならvenvの絶対パスを `RAI19_VENV` にエクスポートします。別パスはtmuxのサイドカーを実行するシェルからも参照できるようにします。
2. Ubuntu 26.04の場合は[メンテナーによる1.9への更新手順](https://github.com/kotetsuy/ryzenai_1_9)を参照してください。動作済みの1.8からの更新を対象とし、アーカイブ名、ドライバーパッケージ、インストーラーの実行例を掲載しています。AMD公式のLinux手順はUbuntu 24.04 LTS向けです。
3. AMDの手順に沿ってデバイスアクセスを設定し、memlockが低い場合は[メンテナーの設定例](https://qiita.com/kotetsu_yama/items/bb2ef5692b66ba87d90f)を参照します。上限やグループを変更したら再ログインし、起動に使う一般ユーザーの新しいシェルで以下を確認します。NPUへのアクセスとmemlockのソフト・ハード上限を確認してください。ハード上限が既定の8192 KiBだと起動スクリプトが停止します。

Ryzen AI 1.9.0のtarballを展開した場合のインストーラー実行例です。venvはインストーラーが作成します。

```bash
# AMDのインストーラーを展開したディレクトリで実行。
# -a yes はAMDのライセンスへの同意を示す。
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

AMDのtarball版では付属のCNN quicktestを実行し、`Actually running on NPU` の件数が1以上であることを確認してから、アプリを起動します。

```bash
(
  source scripts/rai_env.sh || exit 1
  test "$RAI_VERSION" = 1.9 || exit 1
  cd "${RYZEN_AI_INSTALLATION_PATH}/quicktest" || exit 1
  python quicktest.py
)
```

`scripts/rai_env.sh` は1.9、1.8、1.7.1の順に探します。出力同期処理は1.9／XRT 2.26.89と1.8／XRT 2.25.37の検証済みライブラリハッシュに対応します。1.7.1を含め、照合に失敗する組み合わせではサイドカーが停止します。サイドカーにはRyzen AIのvenv、本体にはプロジェクトのuv環境を使います。Ryzen AIのvenvにはベンダー版ONNX Runtimeを保持してください。

## 4. 起動と動作確認

カメラを接続します。`camera.fallback: any` なら優先リストにないカメラも選択できます。カメラなしで試す場合は、再生可能な動画ファイルのパスを記載したローカルM3U8を作り、`config.yaml` の `camera.playlist`（既定は `videos/playlist.m3u8`）を設定します。利用可能なカメラがない場合にプレイリストが使われます。

```bash
./start_all.sh --no-browser
tmux list-windows -t llava
tmux attach -t llava
```

ウィンドウ0〜3は `capture`、`serve`、`vlm`、`npu-yolo` です。Ctrl-bのあとに数字で切り替え、Ctrl-b dでデタッチできます。モデルのコンパイルとウォームアップ後、別ターミナルで以下を実行します。URLはmainの既定ポートです。

```bash
curl -fsS http://127.0.0.1:9931/health
curl -fsS http://127.0.0.1:8082/health
curl -fsS http://127.0.0.1:8082/latest | python3 -m json.tool
/opt/xilinx/xrt/bin/xrt-smi examine
```

- `http://localhost:8080/` を開いて映像を確認します。人物など検出しやすい対象を映すか、その対象が写った動画を使い、検出枠の位置と、VLMロード後に日本語キャプションが出ることを確認します。
- サイドカーの `/health` が `{"ready":true}` を返すことを確認します。`/latest` を繰り返し取得し、`connected: true`、増加する `frame_seq`、対象が映っているときに `boxes` に検出結果が入ることを確認します。`/latest` が503の場合は、まだフレーム結果が公開されていません。
- `xrt-smi examine` に表示されたNPUのBDFを使い、検出中に下記を2回実行します。ハードウェアコンテキストがActiveで、`Submissions` が増えていればNPU実行を確認できます。

```bash
read -r -p 'NPU BDF: ' npu_bdf
/opt/xilinx/xrt/bin/xrt-smi examine -d "$npu_bdf" -r all
```

この統合構成でのYOLO26nのスループットと起動時間は未確認です。推論中の実測値はサイドカーのログに `inf/s` で出力されます。

停止は `./stop_all.sh` を実行します。

## トラブルシューティング

- **映像が出ない**場合は `uv run --no-sync list-cameras` と `v4l2-ctl --list-devices` でカメラを探します。`capture` ウィンドウのログと優先設定、動画を使う場合はプレイリストと動画パスを確認します。
- **検出枠が出ない**場合は、`npu-yolo` のログでモデル、JSON、VitisAI、出力同期のエラーを確認します。初回はコンパイル完了を待ちます。サイドカーはモデルとランタイムに対応する `*_ctx.onnx` と `*_ctx.json` をモデルの隣に生成するため、起動ユーザーにはそのディレクトリへの書き込み権限が必要です。
- **字幕が出ない**場合は、`vlm` でGGUFとROCmのエラーを確認し、`serve` のログと `vlm.server.base_url` を照合します。起動スクリプトはVLMに `--reasoning off` を渡します。
- **セッションが既にある**場合は、`tmux attach -t llava` で既存のセッションに接続するか、`./stop_all.sh` で停止してから起動し直します。

### GPUでYOLO11mを使う場合

稼働中のパイプラインを止め、`yolo.backend: gpu` に変更します。プロジェクトのuv環境に、GPUとROCmの版に合うPyTorchとtorchvisionを先に入れてから、YOLOの追加依存を入れてHIPを確認します。設定された `yolo11m.pt` がローカルにない場合は、Ultralyticsが初回使用時に取得できます。

```bash
source scripts/rocm_env.sh
uv pip install -e '.[yolo]'
uv run --no-sync python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
./start_all.sh --no-browser
```

## ライセンス

[LICENSE](LICENSE)を参照してください。NPU出力同期コードは [yolodemo](https://github.com/kotetsuy/yolodemo) 由来でAGPL-3.0-onlyです。ソースのSPDX表記と[AGPL-3.0本文](licenses/AGPL-3.0.txt)を参照してください。
