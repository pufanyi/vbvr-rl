"""Globally shuffle a VBVR webdataset and write new SFT / RL tar shards.

Tuned for 128-core / 2TB RAM boxes:
  * All ~2TB of input shards are ``mmap``'d in the parent process. The OS
    page cache absorbs the full working set after one read pass, so the
    random-access pattern of a true global shuffle costs ≈ memcpy speed.
  * Writer processes are created with ``fork`` so they inherit the parent's
    mappings — no re-opening, no LRU churn, no per-read syscalls.
  * Tar members are emitted with zero-copy writes from the mmap buffers
    (no intermediate BytesIO round-trip).
  * Defaults to ``num_workers = os.cpu_count()``.

Example:

    uv run python scripts/data/shuffle_and_split_wds.py \
        --src data/vbvr/latents/webdataset \
        --sft-dst data/vbvr/latents/splits/sft \
        --rl-dst data/vbvr/latents/splits/rl \
        --sft-ratio 0.8 \
        --seed 1337 \
        --samples-per-shard 1000

Output is a drop-in replacement for the source layout: ``shard-NNNNNN.tar``
containing paired ``{key}.safetensors`` + ``{key}.json`` members. Point
``latent_webdataset_dir`` at the new dir and train as-is.
"""

from __future__ import annotations

import argparse
import logging
import mmap
import multiprocessing as mp
import os
import tarfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# (src_shard_idx, st_offset, st_size, json_offset, json_size)
Sample = tuple[int, int, int, int, int]

# Populated in parent before forking writer pool; inherited by workers.
_SRC_MMAPS: list[mmap.mmap] = []


# ----------------------------------------------------------------------
# Scanning (parallel, spawn context — no mmap needed yet)
# ----------------------------------------------------------------------


def _scan_shard(task: tuple[int, str]) -> list[Sample]:
    shard_idx, path = task
    entries: dict[str, dict[str, tuple[int, int]]] = {}
    with tarfile.open(path, "r") as tf:
        for m in tf.getmembers():
            if "." not in m.name:
                continue
            stem, ext = m.name.rsplit(".", 1)
            if ext not in ("safetensors", "json"):
                continue
            entries.setdefault(stem, {})[ext] = (m.offset_data, m.size)
    out: list[Sample] = []
    for stem, e in entries.items():
        if "safetensors" in e and "json" in e:
            st_off, st_size = e["safetensors"]
            j_off, j_size = e["json"]
            out.append((shard_idx, st_off, st_size, j_off, j_size))
        else:
            logger.warning("shard %d: incomplete sample %s — skipping", shard_idx, stem)
    return out


# ----------------------------------------------------------------------
# Writing (fork context — workers inherit parent's mmaps)
# ----------------------------------------------------------------------


def _mmap_all(paths: list[str]) -> None:
    global _SRC_MMAPS
    _SRC_MMAPS = []
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            length = os.fstat(fd).st_size
            _SRC_MMAPS.append(mmap.mmap(fd, length, prot=mmap.PROT_READ))
        finally:
            os.close(fd)  # mmap keeps the mapping alive


def _write_tar_member(out_f, name: str, data: memoryview) -> None:
    """Append one tar member, zero-copy from the source mmap slice."""
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    out_f.write(info.tobuf(format=tarfile.USTAR_FORMAT))
    out_f.write(data)
    pad = (-len(data)) % 512
    if pad:
        out_f.write(b"\x00" * pad)


_TAR_EOF = b"\x00" * 1024  # two zero blocks mark end-of-archive


def _write_shard(task: tuple[str, list[Sample], int]) -> int:
    out_path_str, samples, global_offset = task
    out_path = Path(out_path_str)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb", buffering=32 << 20) as out_f:
        for local_idx, (shard_idx, st_off, st_size, j_off, j_size) in enumerate(samples):
            mv = memoryview(_SRC_MMAPS[shard_idx])
            key = f"{global_offset + local_idx:09d}"
            _write_tar_member(out_f, f"{key}.safetensors", mv[st_off : st_off + st_size])
            _write_tar_member(out_f, f"{key}.json", mv[j_off : j_off + j_size])
        out_f.write(_TAR_EOF)
    tmp.rename(out_path)
    return len(samples)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--sft-dst", required=True)
    ap.add_argument("--rl-dst", required=True)
    ap.add_argument("--sft-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--samples-per-shard", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=os.cpu_count() or 32)
    ap.add_argument("--scan-workers", type=int, default=None, help="defaults to min(num-workers, 64)")
    args = ap.parse_args()

    if not 0.0 < args.sft_ratio < 1.0:
        raise ValueError(f"--sft-ratio must be in (0, 1), got {args.sft_ratio}")

    src = Path(args.src)
    sft_dst = Path(args.sft_dst)
    rl_dst = Path(args.rl_dst)
    for d in (sft_dst, rl_dst):
        d.mkdir(parents=True, exist_ok=True)
        if any(d.glob("shard-*.tar")):
            raise FileExistsError(f"{d} already contains shard-*.tar — refusing to overwrite")

    src_shards = sorted(src.glob("shard-*.tar"))
    if not src_shards:
        raise FileNotFoundError(f"No shards in {src}")
    src_paths = [str(p) for p in src_shards]

    scan_workers = args.scan_workers or min(args.num_workers, 64)
    logger.info("scanning %d input shards (workers=%d)", len(src_shards), scan_workers)
    scan_ctx = mp.get_context("spawn")
    tasks = [(i, p) for i, p in enumerate(src_paths)]
    all_samples: list[Sample] = []
    with scan_ctx.Pool(scan_workers) as pool:
        for i, per in enumerate(pool.imap_unordered(_scan_shard, tasks, chunksize=4)):
            all_samples.extend(per)
            if (i + 1) % 100 == 0:
                logger.info("  scanned %d/%d", i + 1, len(src_shards))
    n = len(all_samples)
    logger.info("indexed %d samples", n)

    logger.info("global shuffle (seed=%d)", args.seed)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    shuffled = [all_samples[i] for i in perm]

    cut = int(n * args.sft_ratio)
    subsets = [("sft", sft_dst, shuffled[:cut]), ("rl", rl_dst, shuffled[cut:])]
    logger.info("split: sft=%d rl=%d", cut, n - cut)

    logger.info(
        "mmap'ing %d input shards in parent (~%.1f GB virtual)",
        len(src_paths),
        sum(os.path.getsize(p) for p in src_paths) / 1e9,
    )
    _mmap_all(src_paths)

    write_ctx = mp.get_context("fork")
    sps = args.samples_per_shard
    for name, dst, subset in subsets:
        n_out = (len(subset) + sps - 1) // sps
        logger.info("%s: writing %d shards → %s (workers=%d)", name, n_out, dst, args.num_workers)
        write_tasks = [
            (
                str(dst / f"shard-{i:06d}.tar"),
                subset[i * sps : (i + 1) * sps],
                i * sps,
            )
            for i in range(n_out)
        ]
        done = 0
        with write_ctx.Pool(args.num_workers) as pool:
            for i, n_w in enumerate(pool.imap_unordered(_write_shard, write_tasks)):
                done += n_w
                if (i + 1) % 20 == 0 or (i + 1) == n_out:
                    logger.info("  %s: %d/%d shards (%d samples)", name, i + 1, n_out, done)
        logger.info("finished %s: %d samples", name, done)

    for mm in _SRC_MMAPS:
        mm.close()


if __name__ == "__main__":
    main()
