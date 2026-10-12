"""Loop local video files listed in an extended M3U playlist."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaylistEntry:
    path: Path
    credit: str


def load_playlist(path: Path) -> list[PlaylistEntry]:
    entries: list[PlaylistEntry] = []
    credit = ""
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith(("#CREDIT:", "#CREDITS:")):
            credit = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#"):
            video = (path.parent / line).resolve()
            if video.is_file():
                entries.append(PlaylistEntry(video, "" if credit == "クレジット表記不要" else credit))
            else:
                log.warning("playlist video missing: %s", video)
            credit = ""
    if len(entries) > 255:
        raise ValueError("playlist supports at most 255 videos")
    return entries


class PlaylistPlayer:
    def __init__(self, entries: list[PlaylistEntry]) -> None:
        self.entries = entries
        self.index = 0
        self._cap: cv2.VideoCapture | None = None
        self._period = 1 / 30
        self._next_frame_at = 0.0
        self._retry_at = 0.0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _advance(self) -> None:
        self.close()
        self.index = (self.index + 1) % len(self.entries)
        self._next_frame_at = 0.0

    def read(self, now: float | None = None) -> tuple[np.ndarray, int] | None:
        """Return a frame when due; index is one based for SHM (zero means camera/black)."""
        if not self.entries:
            return None
        now = time.monotonic() if now is None else now
        if now < self._next_frame_at or now < self._retry_at:
            return None
        for _ in range(len(self.entries)):
            if self._cap is None:
                entry = self.entries[self.index]
                self._cap = cv2.VideoCapture(str(entry.path))
                if not self._cap.isOpened():
                    log.warning("cannot open playlist video: %s", entry.path)
                    self._advance()
                    continue
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                self._period = 1 / fps if 1 <= fps <= 120 else 1 / 30
                log.info("playlist: playing %s @ %.2f fps", entry.path.name, 1 / self._period)
            ok, frame = self._cap.read()
            if ok and frame is not None:
                # If processing stalled, resume from now instead of racing through frames.
                self._next_frame_at = now + self._period
                return frame, self.index + 1
            self._advance()
        self._retry_at = now + 1.0
        return None
