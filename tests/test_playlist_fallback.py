from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from src.capture.playlist import PlaylistPlayer, load_playlist
from src.capture.shm_writer import FrameSHM


def _video(path: Path, value: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 16))
    assert writer.isOpened()
    writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()


def test_playlist_credit_and_loop(tmp_path: Path) -> None:
    _video(tmp_path / "a.mp4", 40)
    _video(tmp_path / "b.mp4", 160)
    playlist = tmp_path / "playlist.m3u8"
    playlist.write_text(
        "#EXTM3U\n#CREDIT:作者 A\na.mp4\n"
        "#CREDITS:クレジット表記不要\nb.mp4\n", encoding="utf-8"
    )
    entries = load_playlist(playlist)
    assert [entry.credit for entry in entries] == ["作者 A", ""]
    player = PlaylistPlayer(entries)
    try:
        assert player.read(0)[1] == 1
        assert player.read(0.05) is None  # source FPS is respected
        assert player.read(0.11)[1] == 2
        assert player.read(0.22)[1] == 1
    finally:
        player.close()


def test_source_index_roundtrip() -> None:
    shm = FrameSHM.create(f"test_playlist_{uuid4().hex}", frame_w=16, frame_h=16)
    try:
        shm.write(np.zeros((16, 16, 3), dtype=np.uint8), 16, 16, 0, 0, 1.0, 1,
                  connected=True, source_index=2)
        assert shm.read_source_index() == 2
        assert shm.read()[1].source_index == 2
        shm.write(np.zeros((16, 16, 3), dtype=np.uint8), 0, 0, 0, 0, 0.0, 2,
                  connected=False)
        assert shm.read_source_index() == 0
    finally:
        shm.close()
