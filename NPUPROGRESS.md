# NPUPROGRESS: LLaVA-NPU の YOLO11m を GPU から NPU(XDNA2)へ載せ替え

- 計画立案・現状調査: **Fable 5**(2026-07-07)
- 実装・実機検証: **Opus**(2026-07-07)
- 対象リポジトリ: `/home/araki/LLaVA-NPU`

> ## ✅ ステータス: 実装完了（ライブ end-to-end のみ実機カメラ待ち）
> YOLO11m の物体検出を、従来の **Ultralytics YOLO(GPU/ROCm, `yolo11m.pt`)** から
> **NPU 実行(VitisAI EP, `yolo11m_a16w8.onnx`)** に載せ替えた。VLM(Nemotron)・カメラ・MJPEG 配信は無改修。
> サイドカー方式（VLM の llama-server と同じ別プロセス+ローカルHTTP）で実装し、実機 NPU で検出再現・
> `xrt-smi` オフロード実証・HTTP 配線の結合テストまで完了。`config.yaml` の `yolo.backend` を
> `npu`/`gpu` で切替でき、GPU 経路へのロールバックは一行。

---

## 1. 成果サマリ（何が動くようになったか）

| 項目 | 結果 |
|---|---|
| NPU で YOLO11m 検出 | ✅ bus.jpg で **person×4 + bus×1**（信頼度 0.89/0.89/0.89/0.75, bus 0.87）= yolotest の A16W8 と一致 |
| NPU オフロード実証 | ✅ `xrt-smi` で **HW Context=Active・Columns[0-7]・Submissions 増加** |
| 座標系（逆letterbox） | ✅ bbox は入力フレーム座標系（1280×720 相当）に正しく戻る。全ボックスがフレーム内 |
| スループット | ✅ ウォームアップ後 ~28–29 inf/s（yolotest と同等。カメラ30fpsは最新フレーム処理で自然に間引き） |
| バックエンド切替 | ✅ `config.yaml` `yolo.backend: npu|gpu`。npu 不調なら gpu に戻すだけ |
| クライアント互換 | ✅ bbox JSON スキーマ・`/ws/bbox` は不変（ブラウザ側 canvas は無改修） |

---

## 2. 実装したアーキテクチャ（as-built）

VLM(llama-server)と同じ「別プロセス + ローカルHTTP」パターン。NPU 推論を独立プロセス(サイドカー)に
分離し、serve プロセスはその HTTP を polling するだけ。

```
[capture proc]          [npu-yolo sidecar proc]              [serve proc (uv venv)]
 USB cam                 RAI venv + VitisAI EP                FastAPI + MJPEG + WS
   |                       |  attach SHM(webcam_latest)          |
   |---> SHM ------------->|  read latest frame (seq が進んだ時)  |
                           |  preprocess: letterbox640/RGB/÷255  |
                           |  onnx(a16w8) 推論 @ NPU             |
                           |  decode + class-aware NMS           |
                           |  逆letterbox → 入力フレーム座標      |
                           |  bbox JSON  --HTTP GET /latest------>|  poll(60Hz) → get_latest()
                           |  (http.server /latest, /health)     |  → /ws/bbox へ push(既存のまま)
```

**なぜ別プロセスか**: NPU 実行には `onnxruntime-vitisai` + XRT の `LD_LIBRARY_PATH` が要り、これは
`~/ryzenai/ryzenai_venv` にしか無い。LLaVA 本体の uv venv（torch-ROCm / ultralytics / fastapi）と
統合すると依存衝突・環境汚染のリスクがある。よって VLM と同じくプロセス分離した。
`src/capture/shm_writer.py` の `FrameSHM` が純 Python（numpy + `multiprocessing.shared_memory` のみ、
torch 非依存）なので、RAI venv からも import して SHM を読めるのが成立の鍵。

---

## 3. 追加・変更したファイル

| ファイル | 種別 | 内容 |
|---|---|---|
| `src/npu_yolo/postprocess.py` | 新規 | 純 numpy/cv2 の前後処理。`letterbox`・`preprocess`・**逆letterbox付き `decode_detections` + class-aware NMS**・`COCO_CLASSES`。torch 非依存で RAI venv でも import 可 |
| `src/npu_yolo/__init__.py` | 新規 | 空。`src/inference/__init__.py`（vlm/yolo worker を import）を巻き込まないための独立サブパッケージ |
| `scripts/npu_yolo_sidecar.py` | 新規 | **NPUサイドカー本体**。RAI venv 実行。SHM購読→VitisAI EP推論→decode→stdlib `http.server` で `/latest`・`/health` 配信。推論は daemon スレッド、HTTP は main スレッド |
| `src/server/yolo_runner.py` | 変更 | `backend` で分岐。`npu`=サイドカー `/latest` を poll して再配信 / `gpu`=従来の ultralytics スレッド。対外I/F(`start`/`stop`/`ready`/`get_latest`)は不変で **app.py 無改修** |
| `config.yaml` | 変更 | `yolo.backend: npu` と `yolo.npu:{onnx,sidecar_url,port,poll_hz}` を追加。gpu 用設定(`model`/`device`/`half`)も残置 |
| `start_all.sh` | 変更 | config の `backend` を読み、`npu` の時だけ4つ目の tmux ウィンドウ `npu-yolo` を RAI venv で起動。model/RAI-env の存在チェック付き。既存の 0/1/2(capture/serve/vlm) は不変、npu-yolo は 3 |
| `stop_all.sh` | 変更 | `npu-yolo` ウィンドウと `npu_yolo_sidecar` プロセスの後始末を追加 |
| `tests/test_npu_postprocess.py` | 新規 | 逆letterbox・clip・class-aware NMS・閾値の単体テスト(6件) |
| `.gitignore` | 変更 | `vaip_cache/` 等の NPU コンパイルキャッシュを追加 |
| `models/yolo11m_a16w8.onnx` | 配置 | `~/yolotest` からコピー。`*.onnx` は `.gitignore` 済み=リポジトリには含めない |

---

## 4. 使い方

### バックエンド切替（`config.yaml`）
```yaml
yolo:
  backend: npu   # npu = XDNA2 NPU(VitisAI, 別プロセス) / gpu = Ultralytics(ROCm, serve内)
```

### 起動・停止
```bash
./start_all.sh          # backend:npu なら npu-yolo ウィンドウも自動起動
./stop_all.sh
tmux attach -t llava    # Ctrl-b 0/1/2/3 = capture/serve/vlm/npu-yolo
```
起動順は不問（serve 側はサイドカーが立つまで HTTP retry する）。

### 検証（実機）
```bash
# サイドカー稼働中に別ターミナルで NPU オフロードを確認
/opt/xilinx/xrt/bin/xrt-smi examine -d 0000:c6:00.1 -r all
#  → HW Context=Active・Columns[0-7]・Submissions 増加 なら NPU 実行中
```
ブラウザ `http://localhost:8080/` を開き、人/物が bbox に正しく収まるか目視確認する。
（判定基準は「`Test Finished`」ではなく「`xrt-smi` が Active」＋「ブラウザで正しい位置」）

---

## 5. 実機検証の結果（実施済み）

| 検証 | 結果 |
|---|---|
| 単体テスト(前後処理6件) | ✅ 6/6 pass |
| 実NPU 推論+decode(bus.jpg) | ✅ **person×4 + bus×1**(0.89/0.89/0.89/0.75, bus 0.87)。推論 ~34ms/frame |
| 逆letterbox の座標 | ✅ 全bboxが 810×1080 フレーム内(inbounds)。入力フレーム座標系で正しい |
| NPU オフロード(`xrt-smi`) | ✅ **HW Context=Active・Columns[0-7]・Submissions 5→64→123 と増加** |
| HTTP 配線(サイドカー⇄YoloRunner) | ✅ 結合テストで `/latest`(200/503)・`/health`・`runner.get_latest()` 一致を確認 |
| 構文チェック | ✅ `compileall` / `bash -n` とも OK |

### 未検証（物理USBカメラが必要なため）
- **カメラ→SHM→サイドカー→serve→ブラウザ**のライブ end-to-end。
  SHM 読取コードは既存 gpu 経路から流用し、`FrameSHM.attach` は RAI venv で import 可能を確認済み。
  実機で `./start_all.sh` 後、ブラウザで bbox 位置を目視確認すること（§4 検証）。

---

## 6. 運用上の注意（実装で判明した点）

- **初回コンパイル ~20秒**: このリポジトリでの初回起動時、VitisAI が量子化モデルをコンパイルするため
  最初の1推論に約20秒かかる（計画で「数十秒」と見込んだ通り）。サイドカーは**ウォームアップ完了後に
  `/latest` を出す**設計なので、serve 側は準備できるまで自然に待つ（bbox 空→準備後に出始める）。
  2回目以降は速い。※ 計画で懸念した「vaip_cache が repo 直下に散らかる」事象は今回は発生せず。
  念のため `.gitignore` に `vaip_cache/` を追加済み。
- **venv 分離は厳守**: サイドカーは RAI venv(`source setup_ryzenai_env.sh`)、serve は uv venv。
  `start_all.sh` はサイドカーのウィンドウにだけ RAI env を source し、ROCm 用 `ENV_PREFIX`
  (`HSA_OVERRIDE_GFX_VERSION` 等) は付けない（NPU には不要）。
- **サイドカーの起動コマンド**: RAI venv の python で、`PYTHONPATH=<repo>` を通して起動する
  （`src.capture.shm_writer` / `src.npu_yolo.postprocess` を import するため）。`uv run` ではない。
  `start_all.sh` が自動でこの形にする。
- **ロールバック**: NPU が不調なら `config.yaml` の `yolo.backend: gpu` に戻すだけ。サイドカーは
  起動されず、従来の ultralytics/GPU 経路が serve プロセス内で動く。

---

## 7. 実行時に必要なもの（`~/yolotest` フォルダは不要）

- `~/ryzenai/ryzenai_venv`（onnxruntime-vitisai 1.23.3 / voe 1.7.1）+ XRT/NPU スタック
- `models/yolo11m_a16w8.onnx`（コピー済み）
- LLaVA の uv venv（npu 経路は `requests`=webrtc extra を使用）

`~/yolotest` は**再量子化する時のみ**必要（下記）。通常運用では参照しない。モデルも前後処理コードも
リポジトリ内に取り込み済みで、LLaVA-NPU 単体で自己完結する。

### 再量子化が要るとき（通常不要）
A16W8 を作り直したい場合のみ:
```bash
cd ~/ryzenai/ryzenai_venv && source setup_ryzenai_env.sh
python ~/yolotest/quantize_yolo11m_a16w8.py --input ~/yolo/yolo11m.onnx \
    --output ~/LLaVA-NPU/models/yolo11m_a16w8.onnx --calib-dir ~/yolotest/calib2
```

---

## 8. 設計の背景（なぜこうしたか）

### 8.1 yolotest で確定していた前提
| 事実 | 内容 |
|---|---|
| NPU で YOLO11m は動く | A16W8 量子化 → VitisAI EP で全ノードオフロード、~28 inf/s |
| **A16W8 が必須** | XINT8(活性化8bit)は分類ヘッドを潰し**検出ゼロ**。A16W8(活性化INT16/重みINT8)で FP32 相当に回復 |
| 量子化済みモデルが存在 | `~/yolotest/yolo11m_a16w8.onnx`（再量子化不要でそのまま使える） |
| 前後処理の原型が存在 | `~/yolotest/decode_detect.py`（letterbox640 + decode + NMS） |

一次情報: `~/yolotest/READMEJ.md`（成功手順）, `~/yolotest/HANDOFF.md`（失敗例と xrt-smi 判定基準）。

### 8.2 座標系（逆letterbox）— 最大の実装ポイント
Ultralytics は 1280×720 入力に対し bbox を**入力フレーム座標系**で返していた。NPU 経路は自前で
640 letterbox するので、`decode_detect.py`（640座標のまま表示していた）をそのまま使うと bbox がズレる。
`src/npu_yolo/postprocess.py` で **逆letterbox** を厳密に実装した:
`scale = min(640/h, 640/w)`、`left,top` はパディング量として、`x_orig = (x_lb - left)/scale`,
`y_orig = (y_lb - top)/scale`、その後フレーム範囲に clip。この変換は単体テストで固定してある。
また NMS は Ultralytics 既定に合わせ **class-aware**（別クラス同士は抑制し合わない）にした。

### 8.3 bbox JSON スキーマ（クライアント互換のため不変）
サイドカーは `YoloRunner._publish` と同一スキーマを出し、serve はそれを verbatim で `/ws/bbox` へ流す:
```python
{"frame_seq", "ts_ns", "frame_w", "frame_h", "connected",
 "boxes": [{"label", "conf", "x1", "y1", "x2", "y2"}, ...]}   # 座標は入力フレーム系
```
`frame_seq` は SHM の seq をそのまま載せるので、app.py 側の seq ベース重複排除も従来どおり効く。

---

## 9. スコープ外（今回触っていない）
- VLM(Nemotron / llama-server)経路 — 無改修。
- カメラ capture / SHM writer / MJPEG 配信 / WebSocket 配線 — 無改修（`FrameSHM` はサイドカーから読むだけ）。
- 精度の作り込み — A16W8 で FP32 相当が既に出ているため追加の量子化作業なし。
