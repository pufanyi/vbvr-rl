import os
import time
from pathlib import Path

import pytest

from src.data import remote_io


class _FakeAossClient:
    def __init__(self):
        self.downloads: list[tuple[str, str]] = []

    def download_file(self, uri: str, dst: str) -> None:
        self.downloads.append((uri, dst))
        Path(dst).write_bytes(b"media")


def _set_remote_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WAN_TRAINER_REMOTE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("WAN_TRAINER_REMOTE_DOWNLOAD_WARN_SECONDS", "0")


def test_remote_cache_removes_stale_lock_and_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_remote_env(monkeypatch, tmp_path)
    monkeypatch.setenv("WAN_TRAINER_REMOTE_LOCK_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("WAN_TRAINER_REMOTE_LOCK_STALE_SECONDS", "0.01")
    fake_client = _FakeAossClient()
    monkeypatch.setattr(remote_io, "_get_aoss_client", lambda _uri: fake_client)

    uri = "s3://bucket/path/video.mp4"
    local_path = remote_io._remote_cache_path(uri)
    lock_dir = local_path.with_suffix(local_path.suffix + ".lock")
    lock_dir.mkdir(parents=True)
    old_time = time.time() - 60
    os.utime(lock_dir, (old_time, old_time))

    assert Path(remote_io.localize_media_path(uri)).read_bytes() == b"media"
    assert fake_client.downloads
    assert not lock_dir.exists()


def test_remote_cache_lock_timeout_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_remote_env(monkeypatch, tmp_path)
    monkeypatch.setenv("WAN_TRAINER_REMOTE_LOCK_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("WAN_TRAINER_REMOTE_LOCK_STALE_SECONDS", "3600")

    uri = "s3://bucket/path/video.mp4"
    local_path = remote_io._remote_cache_path(uri)
    lock_dir = local_path.with_suffix(local_path.suffix + ".lock")
    lock_dir.mkdir(parents=True)

    with pytest.raises(TimeoutError, match="remote cache lock"):
        remote_io.localize_media_path(uri)


def test_remote_cache_hit_does_not_remove_existing_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _set_remote_env(monkeypatch, tmp_path)

    uri = "s3://bucket/path/video.mp4"
    local_path = remote_io._remote_cache_path(uri)
    local_path.parent.mkdir(parents=True)
    local_path.write_bytes(b"cached")
    lock_dir = local_path.with_suffix(local_path.suffix + ".lock")
    lock_dir.mkdir()

    assert remote_io.localize_media_path(uri) == str(local_path)
    assert lock_dir.exists()
