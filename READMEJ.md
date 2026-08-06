# LLaVA on ROCm — USB カメラ × YOLO11m × Nemotron Nano Omni × Chrome MJPEG

NucBox EVO X2 (Ryzen AI MAX+ 395 / Radeon 8060S, ROCm 7.2.1) 上で USB カメラ映像を Chrome ブラウザに MJPEG (`multipart/x-mixed-replace`) で配信し、同じ映像に対して YOLO11m の物体検出 (30fps) と Nemotron Nano Omni による日本語キャプション (0.5fps) をリアルタイムオーバーレイするデモ。

> **転送経路について**: もともと WebRTC (aiortc) を使っていましたが、完全オフライン環境 (Wi-Fi OFF / インターネット非接続) では Chrome が ICE host candidate を 1 件も emit しなくなり接続不能になる事象があったため、ICE を必要としない MJPEG over HTTP に移行しました。LAN 越し配信も plain HTTP なのでそのまま動作します。

設計の詳細は [`HANDOFFJ.md`](./HANDOFFJ.md) と [`TECHNICALJ.md`](./TECHNICALJ.md) を参照。

---

## 必要なもの

| 項目 | 想定値 |
|------|------|
| マシン | NucBox EVO X2 (AMD Ryzen AI MAX+ 395, gfx1151, 48GB unified) |
| OS | Ubuntu 24.04.4 LTS (HWE kernel) |
| ROCm | 7.2.1 (`/opt/rocm` symlink) |
| Python | 3.12 |
| パッケージ管理 | `uv` (ローカル: `~/.local/bin/uv`) |
| USB カメラ | UVC 対応のもの 1 台 |
| Chrome | 任意の最近のバージョン (同一マシンまたは LAN 内の別端末) |

事前にインストール済みであることを期待:

- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `tmux` (`sudo apt install tmux`)
- ROCm 7.2.1 (`sudo apt install rocm` 等、AMD 公式手順)
- llama.cpp ROCm/HIP ビルド (`~/llama.cpp/build/bin/llama-server` と `llama-mtmd-cli` がビルド済み)

---

## セットアップ手順

### 1. リポジトリ取得

```bash
git clone <this-repo-url> ~/LLaVA
cd ~/LLaVA
```

### 2. Python 仮想環境と基本依存

```bash
uv venv
uv sync
```

これで `numpy / opencv-python / pyyaml / pyudev` がインストールされ、Step 1 (USB カメラ → SHM) と Step 2 (ホットプラグ対応 CAL) が動く状態になります。

### 3. ROCm 版 PyTorch (Step 3 以降に必要)

PyPI の torch は CUDA 版なので使えません。AMD の ROCm wheel を直接 `wget` してインストール:

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

### 4. YOLO + ONNX (退避プラン用)

```bash
uv pip install -e .[yolo,onnx]
```

`ultralytics` (YOLO11m を初回 predict 時に自動ダウンロード) と `onnx / onnxruntime` (CPU 版) が入ります。

### 5. サーバ依存 (FastAPI + uvicorn + requests)

```bash
uv pip install -e .[webrtc]
```

extra 名は歴史的経緯で `webrtc` のままですが、現行サーバは MJPEG 配信なので aiortc は実質未使用です (依存解決のために一緒にインストールされるだけ)。

### 6. ROCm 環境変数

`~/.bashrc` などに追加して、新しいシェルで自動的に効くようにしておくと楽:

```bash
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
```

(`start_all.sh` は内部で再 export するので、シェル設定を忘れていても tmux セッションでは効きます。)

> **`HSA_OVERRIDE_GFX_VERSION` は設定しないこと。** ROCm wheel も llama.cpp
> (`-DAMDGPU_TARGETS=gfx1151`) も gfx1151 ネイティブビルドなので、override を付けても
> 得るものはありません。逆に古い手順書からコピーした値が残っていると致命的で、
> `HSA_OVERRIDE_GFX_VERSION=11.0.0` だとランタイムが `gfx1100` として認識し、
> カーネル起動がすべて失敗します (`HIP error: invalid device function`)。
> シェルのプロファイルで export されている場合に備え、`start_all.sh` は明示的に
> `unset` しています。

### 7. Nemotron Nano Omni GGUF の準備

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

合計 ~24.5 GB。`config.yaml` の `vlm.model` / `vlm.mmproj` のパスがこれと一致することを確認してください。

### 8. (任意) GPU が見えていることを確認

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True AMD Radeon Graphics
```

---

## 起動と停止

### 一括起動 (推奨)

```bash
cd ~/LLaVA
./start_all.sh
```

これだけで:

1. tmux セッション `llava` を作成
2. window 0: `uv run capture-run` (USB カメラ → SHM、Step 1+2)
3. window 1: `uv run serve` (FastAPI + MJPEG `/stream.mjpg` + YOLO bbox + VLM caption WS、Step 6+7)
4. window 2: `llama-server --reasoning off` (Nemotron 常駐、Step 7b)
5. `http://localhost:8080/` がレスポンスを返すまで最大 30 秒待機
6. Chrome を自動で開く

オプション:

```bash
./start_all.sh --no-browser     # SSH 越しなどで自動 open 不要なとき
./start_all.sh --help
```

セッションへの接続:

```bash
tmux attach -t llava            # ログを直接見る
# Ctrl-b 0 / 1 / 2 で window 切替
# Ctrl-b d でセッションを生かしたまま離脱
```

### 停止

```bash
./stop_all.sh
```

各 window に `Ctrl-C` を送って 5 秒待ち、そのあと `tmux kill-session`。万一プロセスが残っていれば SIGINT → SIGKILL の段階で後始末します。

---

## ブラウザで確認

`./start_all.sh` 実行後、Chrome で `http://localhost:8080/` (または LAN 内別端末から `http://<NucBox の IP>:8080/`) を開くと:

- 中央の `<img src="/stream.mjpg">` に USB カメラ映像 (1280x720, MJPEG)
- 上に半透明 `<canvas>` で YOLO の bbox (色分け、30fps、`person 92%` 形式のラベル)
- 下に半透明ボックスで Nemotron の日本語キャプション (約 50 字、2 秒ごとに更新)
- ステータス行にストリーム状態 / bbox WS / caption の inference 時間と t/s

llama-server がモデルをロードする最初の ~10 秒は caption が `(no caption yet)` のまま、その後更新が始まります。

---

## ステップごとの個別実行 (デバッグ用)

`start_all.sh` を使わず一つずつ起動したいとき:

```bash
# Step 1+2: capture
uv run capture-run                          # 別ターミナル
uv run shm-reader-demo --ticks 10           # SHM 読出しのみ確認
uv run shm-reader-demo --save /tmp/snap.jpg # 1 枚保存
uv run list-cameras                         # /dev/v4l デバイス一覧

# Step 3: YOLO 単独
uv run benchmark-yolo --source synthetic    # synthetic 1280x720 ノイズ
uv run benchmark-yolo --source shm          # 上の capture を起動した上で
uv run export-yolo-onnx --verify            # ONNX 退避プラン

# Step 4: VLM 単独 (mtmd-cli subprocess)
uv run benchmark-vlm --image /tmp/snap.jpg

# Step 5: YOLO + VLM 同居
uv run benchmark-concurrent --frames 600
uv run benchmark-concurrent --no-vlm        # baseline

# Step 6+7: サーバ単体起動
uv run serve                                # T2 相当
~/llama.cpp/build/bin/llama-server -m ... --mmproj ... --reasoning off  # T3 相当
```

---

## トラブルシューティング

### `uv run capture-run` でカメラが見つからない

```bash
ls /dev/v4l/by-id              # USB カメラの symlink が出るか
v4l2-ctl --list-devices        # (要 sudo apt install v4l-utils)
```

`config.yaml` の `camera.preferred` は優先順位つきリストです。カメラ 1 台につき 1 エントリ (`by_id` glob か `vid_pid`) を並べると、接続中で最上位のカメラが選ばれます。どれも一致しなければ `fallback: any` で適当に選ばれます (`fallback: none` なら未登録カメラは使わない)。動作中のホットスワップにも対応: 使用中のカメラを抜くと次の登録カメラへ繋ぎ直し、より優先度の高いカメラを挿すと自動で乗り換えます (`preempt: false` で現在の映像を維持)。

### ブラウザに映像が出ない / ステータスが `stream error` のまま

`http://localhost:8080/stream.mjpg` を直接開いて 200 で MJPEG が降ってくるかを確認してください。

- 200 + 黒画面 → `capture-run` がまだ SEARCHING の可能性。`tmux attach -t llava` で window 0 (capture) を見て、`-> CAPTURING dev=...` と `30 fps` が出ているか確認。出ていなければ `/dev/v4l/by-id/` の symlink と `config.yaml` の `camera.preferred[].by_id` が一致しているか見る
- 404 / 500 → serve window のログを確認 (`tmux attach -t llava` → Ctrl-b 1)
- LAN 別端末から繋がらない場合は `sudo ufw allow 8080` でファイアウォール開放

### caption が空のまま (`(no caption yet)`)

llama-server がまだモデルロード中 (~10 秒) か、`--reasoning off` を付け忘れたか。`tmux attach -t llava` → Ctrl-b 1 で `serve` window のログを確認:

```
vlm-runner: caption (1300ms) 'これは...'         ← OK
vlm-runner: empty caption after strip; raw='<think>...'  ← --reasoning off 不足
```

### `start_all.sh` が「session already exists」で失敗

```bash
./stop_all.sh                  # まず停止
./start_all.sh                 # 再起動
```

または `tmux kill-session -t llava` で強制終了。

### モデルロードが極端に遅い (初回 30 秒以上)

21 GB の GGUF を NVMe から SSD にコピー → ページキャッシュに乗せる時間。2 回目以降は ~10 秒に短縮されます。

### YOLO の fp16 を fp32 に戻したい

`config.yaml` の `yolo.half: true` を `false` に。fp32 のほうが精度はわずかに上がるが、Step 5 の同居測定で fp16 のほうが VLM 側の余裕が大きくなったので fp16 を採用しています。

---

## 開発時のセルフチェック

```bash
# 全 Python ファイルの構文チェック
python3 -m compileall -q src scripts && echo OK

# モジュール import 確認 (依存解決の確認も兼ねる)
uv run python -c "from src.server.app import app; print('imports OK')"
```

---

## NPU バックエンド（YOLO11m を XDNA2 NPU で実行）

YOLO11m の物体検出を、従来の **Ultralytics YOLO(GPU/ROCm, `yolo11m.pt`)** から
**NPU 実行(VitisAI EP, `yolo11m_a16w8.onnx`)** に載せ替えられます。VLM(Nemotron)・カメラ・MJPEG 配信は無改修。
サイドカー方式（VLM の llama-server と同じ別プロセス+ローカルHTTP）で実装されており、`config.yaml` の
`yolo.backend` を `npu`/`gpu` で切替でき、GPU 経路へのロールバックは一行です。
アーキテクチャ・設計背景・実機検証の詳細は [`TECHNICAL.md`](./TECHNICAL.md) を参照。

### 成果サマリ（何が動くようになったか）

| 項目 | 結果 |
|---|---|
| NPU で YOLO11m 検出 | ✅ bus.jpg で **person×4 + bus×1**（信頼度 0.89/0.89/0.89/0.75, bus 0.87）= yolotest の A16W8 と一致 |
| NPU オフロード実証 | ✅ `xrt-smi` で **HW Context=Active・Columns[0-7]・Submissions 増加** |
| 座標系（逆letterbox） | ✅ bbox は入力フレーム座標系（1280×720 相当）に正しく戻る。全ボックスがフレーム内 |
| スループット | ✅ ウォームアップ後 ~28–29 inf/s（yolotest と同等。カメラ30fpsは最新フレーム処理で自然に間引き） |
| バックエンド切替 | ✅ `config.yaml` `yolo.backend: npu|gpu`。npu 不調なら gpu に戻すだけ |
| クライアント互換 | ✅ bbox JSON スキーマ・`/ws/bbox` は不変（ブラウザ側 canvas は無改修） |

### 使い方

#### バックエンド切替（`config.yaml`）
```yaml
yolo:
  backend: npu   # npu = XDNA2 NPU(VitisAI, 別プロセス) / gpu = Ultralytics(ROCm, serve内)
```

#### 起動・停止
```bash
./start_all.sh          # backend:npu なら npu-yolo ウィンドウも自動起動
./stop_all.sh
tmux attach -t llava    # Ctrl-b 0/1/2/3 = capture/serve/vlm/npu-yolo
```
起動順は不問（serve 側はサイドカーが立つまで HTTP retry する）。

#### 検証（実機）
```bash
# サイドカー稼働中に別ターミナルで NPU オフロードを確認
/opt/xilinx/xrt/bin/xrt-smi examine -d 0000:c6:00.1 -r all
#  → HW Context=Active・Columns[0-7]・Submissions 増加 なら NPU 実行中
```
ブラウザ `http://localhost:8080/` を開き、人/物が bbox に正しく収まるか目視確認する。
（判定基準は「`Test Finished`」ではなく「`xrt-smi` が Active」＋「ブラウザで正しい位置」）

### 運用上の注意（実装で判明した点）

- **初回コンパイル ~20秒**: このリポジトリでの初回起動時、VitisAI が量子化モデルをコンパイルするため
  最初の1推論に約20秒かかる。サイドカーは**ウォームアップ完了後に `/latest` を出す**設計なので、
  serve 側は準備できるまで自然に待つ（bbox 空→準備後に出始める）。2回目以降は速い。
  念のため `.gitignore` に `vaip_cache/` を追加済み。
- **venv 分離は厳守**: サイドカーは RAI venv(`source scripts/rai_env.sh`)、serve は uv venv。
  `start_all.sh` はサイドカーのウィンドウにだけ RAI env を source し、ROCm 用 `ENV_PREFIX`
  (`ROCM_PATH` / `HIP_VISIBLE_DEVICES`) は付けない（NPU には不要）。
- **サイドカーの起動コマンド**: RAI venv の python で、`PYTHONPATH=<repo>` を通して起動する
  （`src.capture.shm_writer` / `src.npu_yolo.postprocess` を import するため）。`uv run` ではない。
  `start_all.sh` が自動でこの形にする。
- **ロールバック**: NPU が不調なら `config.yaml` の `yolo.backend: gpu` に戻すだけ。サイドカーは
  起動されず、従来の ultralytics/GPU 経路が serve プロセス内で動く。

### 実行時に必要なもの（`~/yolotest` フォルダは不要）

- Ryzen AI のインストール。有効化は **`scripts/rai_env.sh`** を source する。以下を順に試す:
  1. **1.8** = `~/ryzenai_1_8/venv`（onnxruntime 1.27.0、`VitisAIExecutionProvider` 入り）
     + XRT 2.25.37 / NPU スタック。RAI 1.8 には `setup_ryzenai_env.sh` が同梱されていないため、
     このスクリプトが環境構築を肩代わりする。
  2. **1.7.1** = `~/ryzenai/ryzenai_venv/setup_ryzenai_env.sh`（onnxruntime-vitisai 1.23.3 /
     voe 1.7.1 + XRT 2.21）。1.8 に上げず 1.7.1 を修理して使っているマシン向けの互換経路。

  両方ある場合は 1.8 を優先する（1.8 に移行したマシンにも 1.7.1 のディレクトリは残っているが、
  その venv はもう起動しないため）。`RAI18_VENV` / `RAI171_SETUP` で上書き可能。
- `models/yolo11m_a16w8.onnx`（コピー済み）— 1.7.1 で作ったモデルは 1.8 でもそのまま読めて動く。
  **アップグレード後の再量子化は不要**。
- LLaVA の uv venv（npu 経路は `requests`=webrtc extra を使用）

`~/yolotest` は**再量子化する時のみ**必要（下記）。通常運用では参照しない。モデルも前後処理コードも
リポジトリ内に取り込み済みで、LLaVA-NPU 単体で自己完結する。

#### 再量子化が要るとき（通常不要）
A16W8 を作り直したい場合のみ:
```bash
source ~/LLaVA-NPU/scripts/rai_env.sh
python ~/yolotest/quantize_yolo11m_a16w8.py --input ~/yolo/yolo11m.onnx \
    --output ~/LLaVA-NPU/models/yolo11m_a16w8.onnx --calib-dir ~/yolotest/calib2
```

---

## NPU 復旧・運用メモ（環境アップグレード後）

Ubuntu を **26.04** へ、ROCm を **7.14** へアップグレードした直後、NPU(`amdxdna`) が動作しなくなった
ことがある。3つの独立した問題を切り分けて解消し、一般ユーザ権限で `xrt-smi examine` が
**NPU Strix Halo（Firmware 1.1.2.65）** を認識する状態まで復旧した（根本原因と対処コマンドは
[`TECHNICAL.md`](./TECHNICAL.md) の「NPU recovery」節を参照）。

### 最終検証（再起動後、一般ユーザ `araki` で）— ✅ 全て合格

```bash
ulimit -l                                              # → unlimited                     ✅
ls -l /dev/accel/accel0                                # crw-rw-rw-+ root render 261,0    ✅
source /opt/xilinx/xrt/setup.sh && xrt-smi examine     # [0000:c6:00.1] NPU Strix Halo    ✅
```

3点すべて `sudo` なしで通れば、YOLO11m の NPU パイプライン（`start_all.sh` /
`scripts/npu_yolo_sidecar.py`）を一般ユーザ権限で実行できる。

---

## ライセンス

[`LICENSE`](./LICENSE) を参照。
