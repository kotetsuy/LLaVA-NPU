# 技術詳細

このドキュメントは [`HANDOFF.md`](./HANDOFF.md) で示した設計をどう実装したか、なぜその設計を選んだか、そして実装中に判明した落とし穴を体系化したものです。導入手順は [`READMEJ.md`](./READMEJ.md) を参照。

> **転送経路について**: 当初は WebRTC (aiortc) で映像配信する設計でしたが、オフライン環境 (Wi-Fi OFF / インターネット非接続) で Chrome が ICE host candidate を 1 件も emit しないために接続不能になる事象を踏み、MJPEG (`multipart/x-mixed-replace`) over HTTP に移行しました。詳細は §5 と §7.4 を参照。

---

## 1. アーキテクチャ全体像

![Pipeline architecture](./docs/01_pipeline_architecture.svg)

NucBox EVO X2 (Ryzen AI MAX+ 395, gfx1150, 48 GB unified) 1 台のうえで 4 つの並走コンポーネントを動かし、Chrome へ MJPEG (`/stream.mjpg`) + WebSocket で配信します。

| コンポーネント | プロセス | 入力 | 出力 |
|---|---|---|---|
| Capture | `uv run capture-run` | USB カメラ | SHM (1280×720 BGR letterbox 済) |
| YOLO11m | `serve` 内のバックグラウンドスレッド | SHM | bbox JSON → `/ws/bbox` |
| VLM | `llama-server --reasoning off` (別プロセス) | `serve` の VlmRunner からの HTTP リクエスト (画像+プロンプト) | キャプション JSON → `/ws/caption` |
| MJPEG サーバ | `uv run serve` (FastAPI + uvicorn) | SHM | `/stream.mjpg` (multipart/x-mixed-replace) + WS broadcast |

**設計の鍵となる判断**

- **SHM は "latest frame slot" 1 つだけ**。Capture は常時上書き、読み手 (MJPEG / yolo / vlm) は推論開始時にスナップショット (`np.array(copy=True)`) する。古いフレームでの推論積み残しを構造的に防ぐ。
- **キューにはフレームを乗せない**。bbox / caption JSON だけが asyncio.Queue → WebSocket に流れる。
- **VLM へのフレーム受け渡しは JPEG 圧縮済みバイト列**。SHM の生 BGR を `cv2.imencode('.jpg', ...)` してから llama-server の `/v1/chat/completions` に base64 で投げる。
- **YOLO は `serve` プロセス内のスレッドで動く**。HANDOFF の元案は別プロセスだが、torch/CUDA は GIL を C++ 中で解放するので asyncio loop を block しない。Step 5 で fp16 推論 8 ms p50 が確認できた時点でこの判断は妥当。
- **VLM だけは別プロセス (llama-server)**。理由は (a) 21 GB の GGUF を Python venv とは独立に常駐させたいこと、(b) llama.cpp のチューニングや再起動を server 本体と切り離したいこと。

---

## 2. カメラ抽象化レイヤ (CAL)

![Camera abstraction layer](./docs/02_camera_abstraction_layer.svg)

「画角の違うカメラ・差し込むポートが変わっても動く」要件を満たすため、`src/capture/` に物理層 → 正規化フレーム生成までを集約しました。

### 2.1 デバイス検出 (`device_manager.py`)

- `pyudev.Context().list_devices(subsystem='video4linux')` で udev に登録された v4l デバイスを列挙
- USB の `ID_VENDOR_ID` / `ID_MODEL_ID` を取り出して `vid_pid="046d:0892"` のような指定にも対応
- `/dev/v4l/by-id/...` の symlink からは安定 ID として `by_id` を取得
- 1 つの USB カメラが `index0` (映像) / `index1` (メタデータ) など複数 v4l ノードを作るので、`is_capture_capable()` で実際に `cv2.VideoCapture.read()` 成功するノードに絞り込み
- `config.yaml` の `camera.preferred[]` は**優先順位つきのカメラリスト** (1 エントリ = 1 台、`by_id` glob か `vid_pid` で指定)。`select_device_ranked()` が接続中で capture 可能な最優先の 1 台と rank (`preferred` の index、`fallback` 経由のマッチは `None`) を返す。`fallback: any` は未登録カメラも採用、`fallback: none` は `preferred` 一致のみ採用。よって登録済みカメラが複数挿さっていても画面に流すのは最上位の 1 台だけ

### 2.2 フォーマット交渉 (`format_negotiator.py`)

`cv2.VideoCapture` の `set(CAP_PROP_FOURCC, ...)` ladder を `MJPG → YUYV` の優先順で試行し、`get(CAP_PROP_*)` で実際に確定した値を読み戻します。MJPEG を優先するのは USB 帯域効率が YUYV より圧倒的に良いから (1280×720@30fps が YUYV だと USB 2.0 帯域で破綻しやすい)。

### 2.3 letterbox 二段構成 (`frame_normalizer.py`)

- **CAL 出力 = 1280×720 固定 BGR letterbox 済**
- スケール = `min(target_w/src_w, target_h/src_h)` でアスペクト保ったまま縮小、余白は `(114, 114, 114)` グレーで埋める
- `pad_x / pad_y / scale / original_w / original_h` を SHM ヘッダに記録 → 下流で逆変換可能
- **YOLO11m 入力 (640×640) と VLM 入力 (~448) は Ultralytics / mtmd 側で自動 letterbox**。CAL では二段目を作らない (重複変換のコストを避ける)
- bbox 座標は CAL 正規化フレーム (1280×720) 上の絶対座標で WS 送出。Chrome 側は `<canvas width=1280 height=720>` の固有解像度を CSS でビデオに合わせるだけ → 1 段の縮尺だけで描画可能

### 2.4 ホットプラグ対応 (`hotplug_watcher.py`, `capture_session.py`)

- `pyudev.MonitorObserver(filter_by='video4linux')` で `add` / `remove` イベントを Queue に積む
- メイン loop は **2 状態の状態機械**:
  - `SEARCHING`: capture 不能。30 fps で黒フレーム + `connected=False` を SHM に書き続ける (下流が「カメラ不在」を即座に認識できる)
  - `CAPTURING`: `cv2.VideoCapture` を別スレッド (`CaptureReader`) で読み、main thread が SHM へ書く
- `CaptureReader` は `cv2.VideoCapture.read()` の抜き取り時ブロックを避けるため、daemon thread で常時 read。停止時は `cap.release()` で stuck read を解除
- `remove` 時に当該 dev_path がアクティブなら SEARCHING に遷移し、rescan して残っている登録済みカメラへ繋ぎ直す
- SEARCHING 中の `add` は即時 rescan。CAPTURING 中の `add` は `preempt_settle_sec` のデバウンスタイマーを張り (udev はデバイス ready 前に発火するため)、満了時にアクティブ dev_path を除外して再評価。**より優先度の高いカメラが現れた場合のみ**、先に新カメラを open してから `CaptureReader` を差し替える (黒フレームを挟まない)。同等以下の優先度の `add` は無視 = 「複数挿さっても 1 台だけ流す」を維持。ライブ切替を止めたいときは `preempt: false`
- MJPEG ストリームは SHM 経由で間接接続なので、カメラが入れ替わっても Chrome の `<img src="/stream.mjpg">` を切らない (HTTP 接続は維持、黒フレームを挟んで実映像復帰)

### 2.5 カメラの追加方法

新しい USB カメラを使えるようにするには `config.yaml` の `camera.preferred[]` に 1 エントリ追記するだけ。コード変更は不要。

1. **識別子を調べる** — カメラを挿して `uv run list-cameras` を実行する。`CAPTURE` 列が `yes` のノードが映像を出せるノード (1 台が `index0`=映像 / `index1`=メタデータ等の複数ノードを作るため)。その行の `BY-ID` (= `by_id`) か `VID:PID` (= `vid_pid`) を控える。

   ```
   CAPTURE  DEV            VID:PID     BY-ID
   yes      /dev/video0    056e:701a   usb-Alcor_Micro__Corp._ELECOM_2MP_Webcam-video-index0
   no       /dev/video1    056e:701a   usb-Alcor_Micro__Corp._ELECOM_2MP_Webcam-video-index1
   ```

2. **識別子を選ぶ** — 同型カメラを複数挿して個体を区別したいなら `by_id` (シリアルを含む `BY-ID` なら一意。末尾を `*` にして glob 可)、機種単位で手軽に指定したいなら `vid_pid` (小文字化して完全一致)。1 エントリに書くのは **どちらか一方**。

3. **`camera.preferred[]` に追記する** — リストは**上ほど高優先度**。登録済みカメラが複数挿さっていても、最上位にマッチした 1 台だけが画面に流れる。最優先にしたいなら先頭へ、フォールバック扱い (上位に一致が無いときだけ使う) なら末尾へ置く。`name` はログ表示用の任意ラベルで選択ロジックには不使用。

   ```yaml
   camera:
     preferred:
       - name: 2k-usb-camera          # 優先度 0 (最優先)
         by_id: usb-DC474C08_..._2K_USB_Camera_...*
       - name: elecom-2mp             # 最下位 = フォールバック扱い
         by_id: usb-Alcor_Micro__Corp._ELECOM_2MP_Webcam-video-index0
   ```

4. **反映** — `config.yaml` の編集を読み込むにはキャプチャを再起動する (`./stop_all.sh && ./start_all.sh`)。`preempt: true` なら稼働中に挿し替えるだけでもより高優先度のカメラへ自動で乗り換わるが、config の編集内容自体は再起動で反映する。

> `fallback: any` (既定) なら `preferred` に未登録のカメラも最低優先度で繋がる。`preferred` に載せたカメラ**だけ**を使いたいときは `fallback: none` にする。

---

## 3. SharedMemory 設計 (`shm_writer.py`)

### 3.1 レイアウト (合計 36 B + frame data)

| offset | size | フィールド |
|--------|------|----------|
| 0 | 8 | `seq_lock` (uint64): even=stable / odd=writer mid-write |
| 8 | 8 | `timestamp_ns` (uint64) |
| 16 | 2 | `original_w` (uint16) |
| 18 | 2 | `original_h` (uint16) |
| 20 | 2 | `frame_w` (uint16) |
| 22 | 2 | `frame_h` (uint16) |
| 24 | 2 | `pad_x` (uint16) |
| 26 | 2 | `pad_y` (uint16) |
| 28 | 4 | `scale` (float32) |
| 32 | 1 | `channels` (uint8) |
| 33 | 1 | `pixel_format` (uint8): 0=BGR / 1=RGB |
| 34 | 1 | `connected` (uint8): 0=合成黒フレーム / 1=実映像 |
| 35 | 1 | (padding) |
| 36 | W·H·3 | frame data (uint8) |

`struct` フォーマット: `<QQHHHHHHfBBB1x` (Python の struct.calcsize で 36 確認済)。

### 3.2 seqlock の挙動

ライター (Capture) は単一プロセス。x86_64 の整列 8 byte 書き込みは hardware atomic なのでロックフリー seqlock が成立します。

```
ライター:
    1. seq = next odd      (= 書き込み中マーカー)
    2. ヘッダとフレームを書き込む
    3. seq = next even     (= 完了マーカー)

リーダー:
    for retry in range(16):
        s1 = read seq
        if s1 odd: sleep(100us); continue   ← ライターと衝突中
        copy header + frame
        s2 = read seq
        if s1 == s2: success
        else: continue                       ← copy 中にライターが上書き
    return None                              ← 16 retry でも捕まえられず
```

### 3.3 「画面が一瞬黒くなる」バグの修正

初版ではリーダーがタイトに 8 retry → 失敗で `None` 返却 → 下流 (当時の `ShmVideoTrack`、現在の MJPEG generator) が黒フレームに fallback、というパスで Chrome 上に時々 1 frame の黒が出ていました。原因は:

- ライターの "odd" 滞在時間 ≈ 500 µs (`np.copyto` で 2.6 MB)
- リーダーの 8 retry はタイトループで合計 ~8 µs しか経過せず、ライターが終わる前に諦めていた

修正:

1. `read()` の retry に `time.sleep(100us)` を挟み、最大 retry を 8 → 16 に
2. 下流側で「最後に成功したフレーム」をキャッシュし、None だったら直前フレームで埋める (TTL 1 秒で stale ガード) — WebRTC 時代の `ShmVideoTrack` で導入、現行の MJPEG generator にも同様のフォールバックを実装

これで Chrome 上の黒フレーム発生は実測 0 に。

### 3.4 resource_tracker パッチ

`multiprocessing.shared_memory` には [bpo-38119](https://bugs.python.org/issue38119) があり、attach した側でも `unlink` しようとして spurious warning や二重 unlink が起きます。`_suppress_resource_tracker_for_shm()` で `register` / `unregister` をモンキーパッチして無視させる定石対応を入れています。

---

## 4. 推論バックエンド

### 4.1 YOLO11m (Step 3)

| 項目 | 値 |
|------|----|
| バックエンド | Ultralytics + ROCm PyTorch 2.9.1 |
| 入力 | 1280×720 BGR (SHM 正規化フレーム) |
| `imgsz` | 640 (Ultralytics 内部で letterbox + 逆変換、bbox は元座標で返る) |
| 量子化 | fp16 (`yolo.half: true`) |
| 単独 fps | **97.8 fps** (`benchmark-yolo --source shm`、ただし dedup なし、GPU 律速) |
| パイプライン fps | **30.1 fps** (`benchmark-concurrent --no-vlm`、カメラ 30fps 律速) |

**ベースラインの取り違いに注意**: `benchmark-yolo --source shm` は SHM 重複 read で同フレーム何度も推論する → GPU 純粋スループット 97.8。一方 `benchmark-concurrent --no-vlm` は `meta.seq` で dedup → カメラレートに張り付く 30 fps。Step 5 の比較は後者を baseline に取らないと「-71% 劣化」のような誤読が起きます。

退避プラン (`scripts/export_yolo_onnx.py`): `model.export(format='onnx', imgsz=640, simplify=True)` で ONNX を吐き、`onnxruntime` で読める状態を維持。CPU EP で実測 **15.4 fps** (30 fps target には届かないので primary ではなく、graceful degradation 用)。MIGraphX EP は AMD-built ort 必須。

### 4.2 Nemotron-3 Nano Omni (Step 4 / 7b)

| 項目 | 値 |
|------|----|
| モデル | `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF`, Q4_K_XL (~21 GB) + mmproj-F16 (~1.5 GB) |
| ランタイム | llama.cpp ROCm/HIP build (`llama-server` 常駐) |
| 入力 | 1280×720 BGR → JPEG (quality 90) → base64 → `/v1/chat/completions` |
| `n_predict` | 128 (50 字程度の日本語キャプション、~60-100 tokens) |
| 単独 inference | **~1262 ms** (Step 4 mtmd-cli) / `~1300 ms` (Step 7b llama-server) |
| 並走 inference | **~1294 ms** (Step 5、YOLO 同居時)、+2.5% の劣化のみ |

**`--reasoning off` 必須**。Reasoning モデルなのでデフォルト (`auto`) では `<think>` タグに `n_predict` を全消費し、回答が空になります。同じ GGUF を `mtmd-cli` で使うと空 `<think></think>` の挙動になるので両ランタイムで違うのは要注意。

`VlmServerWorker` は llama-server の `/v1/chat/completions` レスポンスから:

- `choices[0].message.content` をキャプション本文として取得
- `<think>...</think>` を正規表現で除去 (保険)
- `timings.prompt_ms / predicted_ms / *_per_second` と `usage.prompt_tokens / completion_tokens` を `VlmTiming` に詰める

### 4.3 Step 5: YOLO + VLM 同居の Go/No-Go

```
                          YOLO alone   YOLO+VLM (fp32)   YOLO+VLM (fp16)
fps                       30.1         27.9              27.9
p50 latency (ms)          11.07        11.78             8.05
p99 latency (ms)          11.80        88.67             112.32
VLM median inf (ms)       —            1294              1151
VLM eval_tps              —            48.1              53.6
```

`benchmark-concurrent --frames 600` の出力。fp16 が VLM 側の余裕を増やす ("YOLO が早く終わるので VLM が触れる窓が広がる") のがこのデータから読み取れる重要な発見です。p99 spike (~100 ms) は VLM の eval phase でのコンテキストスイッチ起因で、bbox 描画上は約 3 frame の stutter として現れる程度。

---

## 5. 映像配信 (`src/server/`)

### 5.1 経緯: なぜ WebRTC を捨てたか

当初は aiortc + `RTCPeerConnection` で WebRTC video track を配信する設計で、同一 LAN ならホスト candidate だけで繋がる想定でした。実装後、自宅オフライン環境 (Wi-Fi OFF / インターネット非接続) で次の症状に遭遇:

- `POST /offer` は 200 を返す
- aiortc は `connection state -> connecting` に遷移する
- が、その先 `connected` にも `failed` にも進まず無限に滞留 → ブラウザの映像が黒のまま

`chrome://webrtc-internals` で確認したところ、Chrome 側で `onicecandidate` が**一度も発火しておらず**、`iceState=new` のまま。Wi-Fi OFF で非ループバック interface が無くなった結果、Chrome の WebRTC スタックが local host candidate を 1 件も emit しなくなったのが直接原因です。試した対策と結果:

1. **aioice の loopback フィルタ除外をモンキーパッチ** (サーバ側で `127.0.0.1` を host candidate に乗せる) → サーバ側 candidate は出るが、Chrome 側がそもそも候補を出さないので無効
2. **`http://localhost:8080/` ↔ `http://127.0.0.1:8080/` 切替** → 変化なし
3. **`chrome://flags/#enable-webrtc-hide-local-ips-with-mdns` を Disabled** → 変化なし
4. **`RTCPeerConnection({iceServers: [{urls: 'stun:127.0.0.1:3478'}]})`** (届かない dummy STUN を入れて Chrome の gathering を起こす) → 変化なし
5. **trickle ICE 実装** (`POST /candidate` で逐次受け取り、クライアントは `icecandidate` イベントで送信) → Chrome がイベント自体を発火しないので無意味

Chrome 側の挙動を変えられないため、ICE を必要としない経路にスイッチ。

### 5.2 MJPEG ストリーム (`app.py`)

`GET /stream.mjpg` で `multipart/x-mixed-replace; boundary=frame` を返す `StreamingResponse`:

```python
@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    async def gen():
        shm: FrameSHM | None = None
        last_seq = -1
        black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        while True:
            t_start = time.monotonic()
            if shm is None:                              # lazy attach
                try: shm = FrameSHM.attach(shm_name)
                except (FileNotFoundError, RuntimeError): pass
            frame = black
            if shm is not None and (got := shm.read()) is not None:
                fresh, meta = got
                if meta.seq != last_seq:
                    frame = fresh; last_seq = meta.seq
                else:
                    frame = fresh
            ok, jpeg = await asyncio.to_thread(
                cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                yield (f"--frame\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg.tobytes() + b"\r\n"
            await asyncio.sleep(max(0.0, frame_period - (time.monotonic() - t_start)))
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame", ...)
```

要点:

- **lazy attach**: capture-run が後から起動しても自動接続。未 attach の間は `black` を流して接続を保つ
- **JPEG エンコードは off-thread**: `cv2.imencode` は CPU を持って行くので `asyncio.to_thread` で event loop から逃がす (Python 3.9+)
- **rate limit**: `config.yaml > camera.format.fps` (= 30) を frame_period に換算し、`asyncio.sleep` で 30fps 上限
- **品質**: `config.yaml > server.mjpeg_quality` (default 80)。1280×720 で 1 フレーム ~80-120 KB、30fps で 20-30 Mbps 程度
- **STUN / TURN / ICE 一切無し**: plain HTTP only。LAN 越しも `<NucBox>:8080` で素直に動作

### 5.3 WS broadcast (`ws_broadcaster.py`, `yolo_runner.py`, `vlm_runner.py`)

WebRTC 移行で `/ws/bbox` と `/ws/caption` のロジックは無傷:

- `WsBroadcaster`: クライアント集合 + asyncio Lock + JSON broadcast。送信失敗のクライアントを自動 drop
- `YoloRunner`: daemon thread。SHM read → predict → `_latest = {...}` を thread-safe slot で更新
- `_broadcast_bbox_loop`: 30 fps の async task、`frame_seq` で dedup して `/ws/bbox` に push
- `VlmRunner`: 同型、cadence 2 秒。llama-server health check → SHM attach → loop。`ts_ns` で dedup
- `_broadcast_caption_loop`: 0.5 fps cadence

### 5.4 フロントエンド (`src/web/`)

- `<img id="stream" src="/stream.mjpg">` (MJPEG)、`<canvas>` (bbox)、半透明 `<div>` (caption) の 3 層
- `<canvas>` は `width=1280 height=720` 固定、CSS で `<img>` に被せる。bbox 座標が CAL 正規化フレーム空間そのままなので 1 段スケールで描画
- `<img>` の `load` で status を `streaming` 表示、`error` で指数バックオフ (上限 5s) して `?t=<ts>` キャッシュバスター付きで `src` を差し替え再接続
- `overlay.js` / `caption.js` ともに WebSocket disconnect 時は exponential backoff で再接続 (上限 5 秒)

### 5.5 残置ファイル

`src/server/webrtc_track.py` は未使用ですが「将来 STUN/TURN が用意できる環境で再投入する」可能性を残して保持しています。`pyproject.toml` の `[webrtc]` extra も同様の理由で `aiortc` を残してありますが、現行 `serve` は import しません。

---

## 6. 起動・終了スクリプト

### `start_all.sh`

tmux session `llava` に 3 windows:

1. `capture` ← `uv run capture-run`
2. `serve` ← `uv run serve` (FastAPI + YoloRunner + VlmRunner)
3. `vlm` ← `~/llama.cpp/build/bin/llama-server ... --reasoning off`

ROCm 環境変数 (`ROCM_PATH`, `HIP_VISIBLE_DEVICES`) を per-pane で `export` するので `~/.bashrc` に書き忘れていても確実に効きます。`HSA_OVERRIDE_GFX_VERSION` は逆に意図的に **`unset`** します（§9.4-5 参照）。`http://localhost:8080/` がレスポンスを返すまで `curl` で 30 秒ポーリングしてから Chrome (or chromium / xdg-open) を起動。

### `stop_all.sh`

各 window に `Ctrl-C` 送信 → 5 秒待機 → `tmux kill-session`。万一の残留プロセスは `pgrep -f "src.server.app|capture.main|llama-server"` で検出して SIGINT → SIGKILL の 2 段階で後始末。

---

## 7. 実装中に判明した落とし穴 (再発防止メモ)

### 7.1 Capture / SHM

- **SHM seqlock の retry は sleep 必須**: タイトループだとライターの 500 µs 窓を捕まえられない (3.3 節参照)
- **`multiprocessing.shared_memory.SharedMemory` の resource_tracker パッチ**: 二重 unlink 警告を抑制 (3.4 節)
- **`shm.read()` の戻り値は `(frame, meta)` であって `(ts, frame)` ではない**。`benchmark_concurrent.py` 初版で `ts, frame = got` と書いて `ValueError: array truth value ambiguous` を踏んだ
- **`cv2.VideoCapture.read()` は USB 抜き取り時にブロックする**: 別スレッドで read、main thread はタイムアウト監視 (`CaptureReader`)

### 7.2 YOLO

- **dedup の有無でベースラインが ~3 倍違う**: GPU 律速 (97.8 fps) と pipeline 律速 (30.1 fps) を混同しないこと
- **fp16 が並走時に VLM を救う**: 単独 YOLO fps はカメラレートで頭打ちなのに、fp16 にすると並走時の VLM eval_tps が +12% 改善

### 7.3 VLM (llama.cpp)

- **`llama-mtmd-cli` には `--no-display-prompt` フラグが無い**: stdout に prompt がエコーされる → Python 側で先頭 strip
- **stderr に非 UTF-8 バイトが混じる**: モデルロード進捗の制御文字。`subprocess.run(..., errors='replace')` で救済
- **`llama_perf_*` ログの `prompt eval time` と `eval time` を素朴に正規表現すると両方 prompt 行に当たる**: `(?<!prompt )eval time` の negative lookbehind が必要
- **`llama-server` で `--reasoning auto` (default) は *-Reasoning モデルだと thinking が n_predict を食いつぶして content が空になる**: `--reasoning off` を必須化

### 7.4 配信経路 / Frontend

- **Chrome は完全オフラインだと WebRTC ICE candidate を 1 件も emit しない**: Wi-Fi OFF / 非ループバック interface 無し、の組み合わせで host candidate gathering 自体を諦める (`chrome://webrtc-internals` の `onicecandidate` が一切発火しない)。サーバ側 (aioice) で loopback を candidate に乗せても、trickle ICE を実装しても、Chrome 側が出さない以上は無効 → 本プロジェクトでは ICE を必要としない MJPEG に切り替え。§5.1 参照
- **MJPEG 1 接続あたり 20-30 Mbps**: 1280×720 / 30fps / JPEG quality 80 で平均 ~80-120 KB/frame。複数クライアントが繋ぐと比例して帯域も CPU (`cv2.imencode`) も増えるので注意
- **Chrome キャッシュ**: `caption.js` などを更新したのに反映されない時は Ctrl-Shift-R でハードリロード。MJPEG 経路でも `?t=<ts>` 等のクエリで意図的に cache を回避できる
- **`benchmark-*` の `+` / `-` 符号**: ベンチスクリプトで「fps が下がった」を `+71.5%` と表示するなど混乱の元 → 「+ = 改善 / - = 劣化」に統一

---

## 8. NPU バックエンド (YOLO11m を XDNA2 / VitisAI EP で実行)

YOLO11m の物体検出を、従来の **Ultralytics YOLO(GPU/ROCm, `yolo11m.pt`)** から
**NPU 実行(VitisAI EP, `yolo11m_a16w8.onnx`)** に載せ替えた記録（実装・実機検証: Opus, 2026-07-07）。
VLM(Nemotron)・カメラ・MJPEG 配信は無改修。サイドカー方式で実装し、実機 NPU で検出再現・
`xrt-smi` オフロード実証・HTTP 配線の結合テストまで完了。使い方・運用注意は [`READMEJ.md`](./READMEJ.md) を参照。

### 8.1 実装したアーキテクチャ (as-built)

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

**なぜ別プロセスか**: NPU 実行には VitisAI EP 入りの onnxruntime + XRT の `LD_LIBRARY_PATH` が要り、
これは Ryzen AI venv（現在は `~/ryzenai_1_8/venv`。`scripts/rai_env.sh` で source する。§9.5 参照）にしか無い。LLaVA 本体の uv venv（torch-ROCm / ultralytics / fastapi）と
統合すると依存衝突・環境汚染のリスクがある。よって VLM と同じくプロセス分離した。
`src/capture/shm_writer.py` の `FrameSHM` が純 Python（numpy + `multiprocessing.shared_memory` のみ、
torch 非依存）なので、RAI venv からも import して SHM を読めるのが成立の鍵。

### 8.2 追加・変更したファイル

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

### 8.3 実機検証の結果

| 検証 | 結果 |
|---|---|
| 単体テスト(前後処理6件) | ✅ 6/6 pass |
| 実NPU 推論+decode(bus.jpg) | ✅ **person×4 + bus×1**(0.89/0.89/0.89/0.75, bus 0.87)。推論 ~34ms/frame |
| 逆letterbox の座標 | ✅ 全bboxが 810×1080 フレーム内(inbounds)。入力フレーム座標系で正しい |
| NPU オフロード(`xrt-smi`) | ✅ **HW Context=Active・Columns[0-7]・Submissions 5→64→123 と増加** |
| HTTP 配線(サイドカー⇄YoloRunner) | ✅ 結合テストで `/latest`(200/503)・`/health`・`runner.get_latest()` 一致を確認 |
| 構文チェック | ✅ `compileall` / `bash -n` とも OK |

ライブ end-to-end（カメラ→SHM→サイドカー→serve→ブラウザ）は 2026-07-24 の環境復旧後に実機カメラで
確認済み（`start_all.sh` 一発で 4 窓起動、`/latest` が `person conf=0.73` 等を返す。§9 参照）。

### 8.4 設計の背景（なぜこうしたか）

#### 8.4.1 yolotest で確定していた前提
| 事実 | 内容 |
|---|---|
| NPU で YOLO11m は動く | A16W8 量子化 → VitisAI EP で全ノードオフロード、~28 inf/s |
| **A16W8 が必須** | XINT8(活性化8bit)は分類ヘッドを潰し**検出ゼロ**。A16W8(活性化INT16/重みINT8)で FP32 相当に回復 |
| 量子化済みモデルが存在 | `~/yolotest/yolo11m_a16w8.onnx`（再量子化不要でそのまま使える） |
| 前後処理の原型が存在 | `~/yolotest/decode_detect.py`（letterbox640 + decode + NMS） |

一次情報: `~/yolotest/READMEJ.md`（成功手順）, `~/yolotest/HANDOFF.md`（失敗例と xrt-smi 判定基準）。

#### 8.4.2 座標系（逆letterbox）— 最大の実装ポイント
Ultralytics は 1280×720 入力に対し bbox を**入力フレーム座標系**で返していた。NPU 経路は自前で
640 letterbox するので、`decode_detect.py`（640座標のまま表示していた）をそのまま使うと bbox がズレる。
`src/npu_yolo/postprocess.py` で **逆letterbox** を厳密に実装した:
`scale = min(640/h, 640/w)`、`left,top` はパディング量として、`x_orig = (x_lb - left)/scale`,
`y_orig = (y_lb - top)/scale`、その後フレーム範囲に clip。この変換は単体テストで固定してある。
また NMS は Ultralytics 既定に合わせ **class-aware**（別クラス同士は抑制し合わない）にした。

#### 8.4.3 bbox JSON スキーマ（クライアント互換のため不変）
サイドカーは `YoloRunner._publish` と同一スキーマを出し、serve はそれを verbatim で `/ws/bbox` へ流す:
```python
{"frame_seq", "ts_ns", "frame_w", "frame_h", "connected",
 "boxes": [{"label", "conf", "x1", "y1", "x2", "y2"}, ...]}   # 座標は入力フレーム系
```
`frame_seq` は SHM の seq をそのまま載せるので、app.py 側の seq ベース重複排除も従来どおり効く。

### 8.5 スコープ外（今回触っていない）
- VLM(Nemotron / llama-server)経路 — 無改修。
- カメラ capture / SHM writer / MJPEG 配信 / WebSocket 配線 — 無改修（`FrameSHM` はサイドカーから読むだけ）。
- 精度の作り込み — A16W8 で FP32 相当が既に出ているため追加の量子化作業なし。

---

## 9. 環境アップグレード後の NPU 復旧 (Ubuntu 26.04 / ROCm 7.14)

Ubuntu を **26.04** へ、ROCm を **7.14** へアップグレードした直後、NPU(`amdxdna`) が動作しなくなった
（実施: Opus 4.8, 2026-07-24）。独立した複数の問題を切り分けて解消し、一般ユーザ権限で
`xrt-smi examine` が **NPU Strix Halo（Firmware 1.1.2.65）** を認識する状態まで復旧した。
運用者向けの最終検証手順は [`READMEJ.md`](./READMEJ.md) の「NPU 復旧・運用メモ」を参照。

### 9.1 復旧前の症状
- `xrt-smi examine` → **0 devices found**
- `/dev/accel/` が存在しない
- `lsmod | grep xdna` → 空（`amdxdna` 未ロード）
- ただし PCI では認識済み: `c6:00.1 ... Strix Halo Neural Processing Unit`

### 9.2 原因と対処（NPU デバイス側の 3 つの独立した問題）

| # | 問題 | 根本原因 | 対処 |
|---|------|----------|------|
| 1 | `amdxdna` が未ロード | カーネル入れ替え後に自動ロードされていなかった | DKMS 再ビルドで解消（下記 #2 に統合） |
| 2 | `modprobe amdxdna` → `Exec format error` / `disagrees about version of symbol module_layout` | **DKMS モジュールが古いヘッダ状態でビルドされていた**。稼働カーネルは gcc-15 ビルド(`module_layout` CRC `0xe9196a28`)なのに、DKMS 生成物は `0xd954c786` を要求。Ubuntu 26.04 アップグレード過程で「稼働カーネル本体」と「DKMS がビルドに使うヘッダ状態」がズレたのが原因 | 現行カーネルに対して DKMS を作り直し |
| 3 | `xrt-smi examine` → `mmap(len=64MB, offset=4GB) failed (err=-11 EAGAIN)` | **memlock リミットが 8MB（8192KB）** と低く、XRT/NPU が要求する 64MB のピン留めメモリ確保に失敗（root では memlock が緩く成功したことで確定） | `memlock unlimited` を全ユーザに適用 |

```bash
# --- 問題 #1/#2: DKMS を現行カーネルに対して再ビルド ---
sudo dkms remove  xrt-amdxdna/2.21.260102.53.release --all
sudo dkms install xrt-amdxdna/2.21.260102.53.release
sudo depmod -a

# 事前確認: 要求 CRC が稼働カーネルと一致するか（.ko.zst であることに注意）
modprobe --dump-modversions /lib/modules/$(uname -r)/updates/dkms/amdxdna.ko.zst | grep module_layout
#  → 0xe9196a28  module_layout  （= カーネル本体と一致すれば成功見込み）

sudo modprobe amdxdna
ls -l /dev/accel/          # → accel0 が生成される

# --- 問題 #3: memlock を unlimited に（PAM ログインセッションに適用） ---
echo '* - memlock unlimited' | sudo tee /etc/security/limits.d/99-xrt-memlock.conf
# ※ 反映は再ログイン後。現行 root セッションでは既に緩いので検証は sudo 経由で可能:
sudo bash -c 'source /opt/xilinx/xrt/setup.sh; xrt-smi examine'
```

復旧後の確認結果:
```
XRT
  Version              : 2.21.75
  amdxdna Version      : 2.21.260102.53.release_20260309
  NPU Firmware Version : 1.1.2.65

Device(s) Present
  [0000:c6:00.1]  NPU Strix Halo   ✅

dmesg:
  amdxdna 0000:c6:00.1: PASID address mode enabled
  [drm] Initialized amdxdna_accel_driver 1.0.0 for 0000:c6:00.1 on minor 0
```
- カーネル: `7.0.0-28-generic`（gcc 15.2.0 ビルド）
- `ulimit -l`: root では緩く成功。一般ユーザは **再ログイン後に unlimited** となる

### 9.3 パイプライン起動時に判明した 2 つの環境問題（`start_all.sh`）

NPU 自体の復旧後、`./start_all.sh` を初めて実行したところ 4 窓のうち **capture / vlm は正常**、
**serve と npu-yolo が別要因で起動失敗**した。いずれも Ubuntu 26.04 アップグレードの副作用。

| # | 窓 | 症状 | 根本原因 | 対処 |
|---|----|------|----------|------|
| A | serve | `ModuleNotFoundError: No module named 'fastapi'` | `.venv` がアップグレードで **base 依存のみで再作成**され、fastapi 等は `[project.optional-dependencies].webrtc` にあるため未同期。`uv run serve` は暗黙同期で env を base に揃えるので fastapi が入らない | 起動を **`uv run --extra webrtc serve`** に修正（`start_all.sh` 修正済み） |
| B | npu-yolo | `ModuleNotFoundError: No module named 'encodings'` / `init_fs_encoding failed`（インタプリタ自体が起動不能） | ryzenai venv が **旧 `/usr/bin/python3.12`(3.12.3) から `--copies`** で作られていた。26.04 で system Python が **3.14** になり `/usr/bin/python3.12` と `/usr/lib/python3.12`(stdlib) が消失 → コピーされた python バイナリが stdlib を失い起動不能 | **uv 管理の standalone `cpython-3.12.13`** で venv を **in-place upgrade**（インタプリタ/stdlib 参照のみ差し替え、site-packages 330 個は温存） |

問題 B の対処コマンド:
```bash
# 事前調査で確定した事実:
#  - RAI 1.7.1 は公式に Python 3.12.x のみサポート（3.13/3.14 は非対応）
#    → https://ryzenai.docs.amd.com/en/latest/linux.html （"Install Python 3.12.x"）
#  - venv の site-packages 330 個（onnxruntime_vitisai 1.23.3 / voe 1.7.1 等）は
#    すべて cp312 wheel。壊れていたのはインタプリタ + stdlib だけ。
#  - apt には python3.12 が無い（26.04）。uv 管理の 3.12.13 standalone を使う。

STD=/home/araki/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
VENV=/home/araki/ryzenai/ryzenai_venv/venv

# 旧 --copies バイナリを消してから venv を作り直す（site-packages は消えない）
rm -f "$VENV"/bin/python "$VENV"/bin/python3 "$VENV"/bin/python3.12
"$STD" -m venv --without-pip "$VENV"     # bin/ を standalone への symlink で再生成

# 検証: VitisAIExecutionProvider が出れば成功
source /home/araki/ryzenai/ryzenai_venv/setup_ryzenai_env.sh
python -c "import onnxruntime as ort, voe; print(ort.__version__, ort.get_available_providers())"
#  → 1.23.3.dev...  ['VitisAIExecutionProvider', 'CPUExecutionProvider']
```

起動確認（全 4 窓）:
```
capture   : 30fps  1280x720 BGR → SHM
serve     : http://localhost:8080/  HTTP 200
vlm       : llama-server 8081  Nemotron キャプション ~1.2s
npu-yolo  : VitisAIExecutionProvider セッション確立 / warmup 40ms
            http://127.0.0.1:8082/latest → {"boxes":[{"label":"person","conf":0.73,...}]}  ✅ NPU 推論
```

### 9.4 今後のための知見（再発しやすいポイント）

1. **カーネル更新後に NPU が消えたら、まず DKMS 再ビルドを疑う。**
   Ubuntu の版数（例 `7.0.0-28.28`）が同じでも、アップグレード時に稼働カーネル本体と
   ヘッダのビルドがズレると `module_layout` の CRC 不一致（`Exec format error`）が起きる。
   `sudo dkms install xrt-amdxdna/<ver>` で現行カーネルに対して作り直せば解消する。
   - カーネル同梱の **in-tree `amdxdna`** も存在する
     （`/lib/modules/$(uname -r)/kernel/drivers/accel/amdxdna/amdxdna.ko.zst`, CRC は必ずカーネルと一致）。
     DKMS が直らない場合は `sudo dkms uninstall ...` で in-tree 版に切り替える手もあるが、
     XRT 2.21 ユーザ空間との ABI 互換は `xrt-smi examine` の成否で確認すること。今回は DKMS 版で成功。

2. **memlock は unlimited が XRT/NPU の必須要件。**
   低い memlock だと `mmap ... EAGAIN` でデバイス認識に失敗する。
   `/etc/security/limits.d/99-xrt-memlock.conf` に `* - memlock unlimited` を設定済み。
   - この設定は **PAM ログイン経由のセッションにのみ効く**。将来 `start_all.sh` を
     **systemd サービス**化する場合は PAM を通らないため、unit に `LimitMEMLOCK=infinity` が別途必要。

3. **OS メジャーアップグレード後は「system Python 依存の venv」が壊れる。**
   Ubuntu 26.04 で system Python が 3.14 になり、`/usr/bin/python3.12` から
   `--copies` で作った ryzenai venv がインタプリタごと起動不能になった
   （`No module named 'encodings'`）。**RAI は Python 3.12.x 専用**（3.13/3.14 非対応）なので、
   apt に 3.12 が無い 26.04 では **uv 管理の standalone 3.12.13** を使い、
   `python -m venv --without-pip <venv>`（旧 bin/python* を rm してから）で
   **site-packages を温存したまま**インタプリタだけ差し替えるのが最短。cp312 wheel はそのまま動く。

4. **`start_all.sh` の serve は `--extra webrtc` が必須。**
   fastapi/aiortc/uvicorn/requests は `pyproject.toml` の optional group `webrtc` にある。
   OS アップグレード等で `.venv` が base のみに再作成されると、`uv run serve`（extra なし）は
   暗黙同期で env を base に揃え fastapi が消える。**`uv run --extra webrtc serve`** で起動すること
   （2026-07-24 に `start_all.sh` を修正済み）。

5. **この機体では `HSA_OVERRIDE_GFX_VERSION` を設定しない。**
   ROCm PyTorch wheel も llama.cpp ビルド（`-DAMDGPU_TARGETS=gfx1150`）も gfx1150
   ネイティブなので、override に得はない。しかも壊れ方が非対称で、`11.5.0`（= 実際の
   gfx1150）はたまたま無害だが、古い手順書からコピーした値が残っていると致命的になる。
   `HSA_OVERRIDE_GFX_VERSION=11.0.0` では `torch.cuda.is_available()` は `True` のまま
   `gcnArchName` が **`gfx1100`** になり、以降のカーネル起動が全部失敗する
   （`HIP error: invalid device function`）。値は環境変数由来なので、`~/.bashrc` の
   `export` 一行が後々まで GPU 経路を静かに壊し続ける。そのため `start_all.sh` の
   `ENV_PREFIX` では export ではなく `unset` している
   （2026-07-26 変更・`RealtimeDepth` と同じ方針）。

### 9.5 Ryzen AI 1.8 への移行（2026-08-06）

1.7.1 の NPU スタックをアンインストールし（`~/ryzenai_1_8/uninstall_171.sh`: `xrt-npu` /
`xrt_plugin-amdxdna` を purge、`/opt/xilinx` を削除）、**Ryzen AI 1.8 + XRT 2.25.37** に入れ替えた。
その後 `./start_all.sh` は 4 ウィンドウ立ち上がるものの **npu-yolo だけ即死**した。
モデル側には何の問題もなかった。

| 症状 | 根本原因 | 対処 |
|------|----------|------|
| npu-yolo: `No module named 'encodings'` / `init_fs_encoding failed` | `start_all.sh` が今も `$HOME/ryzenai/ryzenai_venv/setup_ryzenai_env.sh` を source していた。ファイル自体は残っているが、背後の 1.7.1 venv にはもう起動可能なインタプリタが無い（かつ onnxruntime-vitisai 1.23.3 は purge 済み XRT 2.21 向けビルド） | `RAI_ENV` を新設の **`scripts/rai_env.sh`**（`~/ryzenai_1_8/venv` を有効化）に向ける |

**A16W8 ONNX の作り直しは不要だった。** 1.7.1 で量子化した `models/yolo11m_a16w8.onnx` は
1.8 でもそのまま読めて動くことを実測で確認済み:

```bash
source scripts/rai_env.sh
python ~/yolotest/decode_detect.py --model models/yolo11m_a16w8.onnx \
    --image ~/yolotest/calib_images/bus.jpg
#  providers: ['VitisAIExecutionProvider', 'CPUExecutionProvider']
#  person x4: [0.89, 0.89, 0.89, 0.75]
#  bus x1:    [0.87]                      ← 1.7.1 当時の基準結果と完全一致
```
1.8 の ORT は **1.27.0**（旧 1.23.3.dev）。VitisAI のコンパイル対象は
`AMD_AIE2P_4x8_CMC_Overlay`、初回セッションのコンパイル約 28 秒、定常の推論は約 37ms。

**なぜ同じコミットが別の 395 マシンでは動いたのか。** git の中身に差は無い。差があるのは、
どちらのリポジトリも追跡していないマシン側の状態（`ryzenai_1_8/.gitignore` は `venv/` を除外、
XRT は apt パッケージ、`/opt/xilinx` は OS 側）。`start_all.sh` は **1.7.1** のインストール先を
ハードコードしていたが、このマシンではその 1.7.1 環境が二重に死んでいた
（`dpkg.log` と `/var/log/dist-upgrade/main.log` より）:

| 日付 | 出来事 |
|------|--------|
| 2026-07-21 | XRT を **2.21.75** に更新。10:38 に `/usr/bin/python3.12 -m venv --copies` で venv 作成、10:56 に A16W8 ONNX 生成 — この時点では NPU 経路は動いていた |
| **2026-08-05 15:12–15:38** | Ubuntu **24.04 → 26.04** の release upgrade。この中で `python3.12` / `libpython3.12t64` / `python3.12-venv` が **削除**され(15:36)、`/usr/lib/python3.12` が消滅 → `--copies` のインタプリタが stdlib を失う。同時に **XRT 2.21.75 も削除** |
| 2026-08-06 | Ryzen AI 1.8 を導入 → XRT **2.25.37** + `~/ryzenai_1_8/venv` |

つまり §9.3 問題 B と同じ手順でインタプリタを直しても、ここでは足りなかった。1.7.1 の
site-packages（`onnxruntime-vitisai 1.23.3`）は purge 済みの XRT 2.21 向けビルドだからである。
向こうのマシンは §9.3 のルート（1.7.1 を修理し XRT 2.21 のまま）を採ったのでハードコードされた
パスが有効なまま、こちらはスタックごと入れ替えた — これが分岐点。

1 つのコミットで両方のマシンを動かすため、`scripts/rai_env.sh` は **1.8 を優先し、無ければ
1.7.1 にフォールバック**する（場所は `RAI18_VENV` / `RAI171_SETUP` で上書き可、選ばれた版は
`RAI_VERSION` として export。`start_all.sh` は起動前にサブシェルで実行して、どちらを使うか表示する）。
両方ある場合に 1.8 を優先するのは、1.8 に移行したマシンにも 1.7.1 のディレクトリと setup
スクリプトが残っており、**ファイルの存在は 1.7.1 が動く証拠にならない**ため。加えて 1.7.1 経路では
先に `venv/bin/python` の起動を試し、死んでいる場合は明示的に報告する（サイドカーが
`No module named 'encodings'` で落ちるのを待たない）。

`scripts/rai_env.sh` が 1.8 の環境を手で組み立てているのは、**RAI 1.8 に `setup_ryzenai_env.sh` が
同梱されていない**ため。中身は `~/ryzenai_1_8/run_quicktest.sh` と同じ構成で、見落としやすい
2 つの回避策を含む:

1. `libonnxruntime_vitisai_ep.so` が `libpeano-lib.so.21.0git` を NEED するが、これは
   `site-packages/lnx64.o/tools/peano/lib` にしか無く、公式手順の `LD_LIBRARY_PATH` に含まれない。
   通さないと EP が**エラーも出さず CPU にフォールバック**する（NPU が使われない）。
2. venv の `activate` が `voe/lib` を `/opt/xilinx/xrt/lib` より前に置き、`voe/lib` には古い
   `libxrt_coreutil.so.2.19.184` が入っている。これを掴むと XRT 2.25.37 の `libxrt_core.so.2` が
   `undefined symbol: xrt_core::smi::get_option_options` で失敗 →「Failed to create runner」→ abort。
   インストール済み XRT のライブラリを必ず先頭にする。

あわせて `start_all.sh` の npu 経路に **memlock の事前チェック**を追加した。`ulimit -H -l` が
`8192` のままだと XRT が EAGAIN で死ぬので、その場で明示的に失敗して
`~/ryzenai_1_8/fix_memlock.sh` を案内する（この設定は実行後に開いた端末にしか効かない）。

### 9.6 サイドカー起動時間 25秒 → 0.7秒（EPContext モデル化, 2026-08-06）

**症状.** 1.8 移行後、npu-yolo ウィンドウが bbox を出し始めるまで約 40 秒かかるようになった
（1.7.1 では約 5 秒）。定常のスループットには影響なし。

**計測.** `ort.InferenceSession(...)` と初回 `run()` を分けて計ると、全部セッション生成側で、
ウォームアップ推論自体は 30ms しかかかっていない:

```
SESSION_INIT 24.76s providers=['VitisAIExecutionProvider', 'CPUExecutionProvider']
WARMUP 0.03s
  run0 32.8ms ...
```

EP のログを見ると理由は明白で、起動のたびに AIE のフルコンパイルが走っている
（`Target architecture: AMD_AIE2P_4x8_CMC_Overlay`、`vaiml_compile_x2_v2 time: 8127 ms`、
`PDI Swap times: 227`）。

**原因.** コンパイル結果がディスクに一切残らない。`~/.cache/vaip` は生成されず、1.7.1 時代の
`cacheDir` / `cacheKey` provider option を渡しても何も変わらない。1.8 のフロー
（`EnableInMemoryMladfCompilePass`、`vaip_config.json` の `enable_cache_file_io_in_mem`）は
コンパイル成果物をメモリ内に保持する方式で、再起動を安くしていた 1.7.1 の `vaip_cache/` に
相当する仕組みが無くなっている。つまり毎回ゼロからの再コンパイルだった。

**対処**（`scripts/npu_yolo_sidecar.py` の `_ensure_ep_context`）。ORT のセッション設定で、
**EPContext** ノードを持つ ONNX に事前コンパイルして保存する:

```python
so.add_session_config_entry("ep.context_enable", "1")
so.add_session_config_entry("ep.context_file_path", str(tmp))
so.add_session_config_entry("ep.context_embed_mode", "1")  # 単一ファイルに埋め込む
```

元モデルの隣に `models/yolo11m_a16w8_ctx.onnx`（28 MB）が作られ、以降の起動はこれを読む。
実機での計測:

| | セッション生成 | 推論 | 出力 |
|---|---|---|---|
| `yolo11m_a16w8.onnx` | 24.6 秒 | 32.9 ms | — |
| `yolo11m_a16w8_ctx.onnx` | **0.67 秒** | 32.9 ms | ビット一致（生の `(1,84,8400)` テンソルを `np.array_equal`） |

キャッシュの陳腐化判定は EP 任せにせず、スタンプ（`models/yolo11m_a16w8_ctx.json`: 元モデルの
サイズと mtime、`ort.__version__`、`RAI_VERSION`）で行う。EP 側のバージョンチェックは C++ の
ロード奥深くで失敗するため。コンパイルは `*.tmp` に書いてから rename するので、中断しても
壊れたキャッシュは残らない。何らかの理由でコンパイルに失敗した場合は警告を出して元モデルに
フォールバックする（起動が遅くなるだけで、起動失敗にはならない）。`--no-ctx-cache` で無効化可。
生成物は両方 gitignore 済み（モデルは既存の `*.onnx`、スタンプ用に `*_ctx.json` を追加）。

---

## 10. 関連ドキュメント

- [`HANDOFF.md`](./HANDOFF.md) — Claude.ai で行った設計検討の引き継ぎ文書 (本実装の元設計)
- [`READMEJ.md`](./READMEJ.md) — git clone から動作までのセットアップ手順
- [`docs/01_pipeline_architecture.svg`](./docs/01_pipeline_architecture.svg) / [`docs/02_camera_abstraction_layer.svg`](./docs/02_camera_abstraction_layer.svg) — 設計図
- [`docs/LLaVA設計図.pptx`](./docs/LLaVA設計図.pptx) — Chrome 上での画面レイアウト原案
