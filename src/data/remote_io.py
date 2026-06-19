"""Helpers for localizing remote media paths before video/image decoding."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

_S3_PREFIX = "s3://"


def is_remote_path(path: str) -> bool:
    return path.startswith(_S3_PREFIX)


def resolve_media_path(path: str, root: str | Path | None = None) -> str:
    """Resolve local or S3 media paths without mangling S3 URI separators."""
    if path.startswith(_S3_PREFIX) or Path(path).is_absolute():
        return path
    if root:
        root_str = str(root)
        if root_str.startswith(_S3_PREFIX):
            return root_str.rstrip("/") + "/" + path.lstrip("/")
        return str(Path(root_str) / path)
    return path


def localize_media_path(path: str) -> str:
    """Return a local path for media decoders.

    Local paths are returned unchanged. S3 paths are downloaded with AOSS into a
    cache directory. Configuration is intentionally environment-driven so data
    credentials stay out of training configs:

    - WAN_TRAINER_AOSS_CONF_RULES: JSON list of {"pattern", "conf_path"}
    - WAN_TRAINER_AOSS_CONF_PATH: fallback conf path
    - WAN_TRAINER_REMOTE_CACHE_DIR: download cache root
    """
    if not is_remote_path(path):
        return path
    return str(_download_s3_to_cache(path))


def _download_s3_to_cache(uri: str) -> Path:
    cache_dir = Path(os.environ.get("WAN_TRAINER_REMOTE_CACHE_DIR", "storage/aoss_cache"))
    parsed = urlparse(uri)
    suffix = Path(parsed.path).suffix or ".bin"
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    local_path = cache_dir / parsed.netloc / digest[:2] / f"{digest}{suffix}"
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = local_path.with_suffix(local_path.suffix + ".lock")
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if local_path.exists() and local_path.stat().st_size > 0:
                return local_path
            time.sleep(0.25)

    try:
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        tmp_path = local_path.with_name(f".{local_path.name}.{os.getpid()}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        client = _get_aoss_client(uri)
        client.download_file(uri, str(tmp_path))
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"AOSS download produced an empty file for {uri}")
        tmp_path.replace(local_path)
        return local_path
    finally:
        with suppress(FileNotFoundError):
            lock_dir.rmdir()


def _get_aoss_client(uri: str):
    conf_path = _select_aoss_conf(uri)
    if conf_path is None:
        raise RuntimeError(
            "S3 media requires WAN_TRAINER_AOSS_CONF_PATH or WAN_TRAINER_AOSS_CONF_RULES to select an AOSS config"
        )
    return _get_aoss_client_for_conf(conf_path)


@lru_cache(maxsize=16)
def _get_aoss_client_for_conf(conf_path: str):
    from aoss_client.client import Client

    return Client(conf_path=conf_path)


def _select_aoss_conf(uri: str) -> str | None:
    rules_raw = os.environ.get("WAN_TRAINER_AOSS_CONF_RULES")
    if rules_raw:
        rules = json.loads(rules_raw)
        for rule in rules:
            pattern = rule.get("pattern")
            conf_path = rule.get("conf_path")
            if pattern and conf_path and re.search(pattern, uri):
                return str(conf_path)
    return os.environ.get("WAN_TRAINER_AOSS_CONF_PATH")
