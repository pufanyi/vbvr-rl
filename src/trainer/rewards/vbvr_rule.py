"""VBVR EvalKit rule reward for GRPO.

This reward mirrors the rule-based evaluation path used for VBVR reporting:
decode generated latents to temporary videos, then call the vendored EvalKit
task-specific evaluator for the sample's task. GT latents are decoded only
when original GT video/first/final-frame paths are not available in metadata.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import torch
from loguru import logger

from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.registry import register_reward

_EVALKIT = Path(__file__).resolve().parents[3] / "third_party" / "VBVR-EvalKit"


def _resolve_evalkit_path(evalkit_dir: str | None = None) -> Path:
    return Path(evalkit_dir).expanduser().resolve() if evalkit_dir else _EVALKIT


def _ensure_evalkit_path(evalkit_dir: str | None = None) -> Path:
    evalkit = _resolve_evalkit_path(evalkit_dir)
    if not (evalkit / "vbvr_bench").exists():
        raise FileNotFoundError(f"VBVR EvalKit path does not contain vbvr_bench: {evalkit}")
    if str(evalkit) not in sys.path:
        sys.path.insert(0, str(evalkit))
    return evalkit


def evalkit_supported_task_names(evalkit_dir: str | None = None) -> frozenset[str]:
    _ensure_evalkit_path(evalkit_dir)

    from vbvr_bench.evaluators import TASK_EVALUATOR_MAP

    return frozenset(TASK_EVALUATOR_MAP)


@register_reward("vbvr_rule")
class VBVRRuleReward(BaseReward):
    """Reward = VBVR EvalKit task-specific rule score in [0, 1]."""

    requires_vae: ClassVar[bool] = True

    def __init__(self, trainer, cfg) -> None:
        super().__init__(trainer, cfg)
        self._evalkit = _ensure_evalkit_path(cfg.vbvr_reward_evalkit_dir)

        from vbvr_bench.evaluators import get_evaluator

        self._task_evaluator_map = evalkit_supported_task_names(str(self._evalkit))
        self._get_evalkit_evaluator = get_evaluator
        self._evaluators: dict[str, Any] = {}
        self._thread_local = threading.local()
        self._error_lock = threading.Lock()
        self._score_workers = max(1, int(cfg.vbvr_reward_cpu_workers))
        self._score_executor = (
            ThreadPoolExecutor(
                max_workers=self._score_workers,
                thread_name_prefix=f"vbvr-rule-rank{trainer.rank}",
            )
            if self._score_workers > 1
            else None
        )
        self._warned_unsupported: set[str] = set()
        self._error_count = 0
        self._tmp_root = Path(cfg.vbvr_reward_tmp_dir) / f"rank{trainer.rank}"
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def __call__(
        self,
        generated_latents: torch.Tensor,
        gt_video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        indices: torch.Tensor | None = None,
        expert_filter: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del condition, prompt_embeds, indices

        B = generated_latents.shape[0]
        device = generated_latents.device

        # EP double-count guard: the expert-parallel GRPO path sums high+low rewards.
        if expert_filter == "high":
            return torch.zeros(B, device=device, dtype=torch.float32)

        if meta is None or "sample_tar" not in meta:
            raise RuntimeError(
                "VBVRRuleReward requires WebDataset JSON metadata with 'tar'. "
                "Expected VBVRLatentDataset to expose it as 'sample_tar'."
            )

        rewards = torch.full(
            (B,),
            float(self.cfg.vbvr_reward_unsupported_score),
            device=device,
            dtype=torch.float32,
        )
        decode_bs = max(1, int(self.cfg.vbvr_reward_decode_batch_size))
        task_names: list[str] = []
        supported_indices: list[int] = []
        for i in range(B):
            task_name = self._task_name(meta, i)
            task_names.append(task_name)
            if self._is_supported(task_name):
                supported_indices.append(i)

        score_jobs: list[tuple[int, dict[str, Any]]] = []
        for start in range(0, len(supported_indices), decode_bs):
            batch_indices = supported_indices[start : start + decode_bs]
            latent_indices = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
            gt_video_paths = [self._meta_path(meta, "sample_gt_video_path", i) for i in batch_indices]
            gt_first_frame_paths = [self._meta_path(meta, "sample_gt_first_frame", i) for i in batch_indices]
            gt_final_frame_paths = [self._meta_path(meta, "sample_gt_final_frame", i) for i in batch_indices]
            needs_gt_decode = [
                video_path is None or first_path is None or final_path is None
                for video_path, first_path, final_path in zip(
                    gt_video_paths,
                    gt_first_frame_paths,
                    gt_final_frame_paths,
                    strict=True,
                )
            ]
            gen_latents = generated_latents.index_select(0, latent_indices)
            if any(needs_gt_decode):
                decoded = self.trainer.model.decode_latents(
                    torch.cat(
                        [
                            gen_latents,
                            gt_video_latents.index_select(0, latent_indices),
                        ],
                        dim=0,
                    )
                )
            else:
                decoded = self.trainer.model.decode_latents(gen_latents)
            videos = self._to_uint8_videos(decoded)
            n = len(batch_indices)
            gen_videos = videos[:n]
            gt_videos = videos[n:] if any(needs_gt_decode) else [None] * n

            for local_i, i in enumerate(batch_indices):
                task_name = task_names[i]
                prompt = self._meta_item(meta, "sample_prompt", i, default="")
                score_jobs.append(
                    (
                        i,
                        {
                            "task_name": task_name,
                            "prompt": str(prompt or ""),
                            "gen_video": gen_videos[local_i],
                            "gt_video": gt_videos[local_i],
                            "gt_video_path": gt_video_paths[local_i],
                            "gt_first_frame_path": gt_first_frame_paths[local_i],
                            "gt_final_frame_path": gt_final_frame_paths[local_i],
                            "sample_id": self._sample_id(meta, i),
                        },
                    )
                )

            del decoded

        for i, score in self._score_jobs(score_jobs):
            rewards[i] = score

        return rewards

    def _is_supported(self, task_name: str) -> bool:
        if task_name in self._task_evaluator_map:
            return True
        if task_name not in self._warned_unsupported:
            self._warned_unsupported.add(task_name)
            if self.trainer.rank == 0:
                logger.warning(
                    "VBVR rule reward: task '{}' has no EvalKit evaluator; using unsupported_score={}",
                    task_name,
                    self.cfg.vbvr_reward_unsupported_score,
                )
        return False

    def _evaluator(self, task_name: str):
        if self._score_workers > 1:
            evaluators = getattr(self._thread_local, "evaluators", None)
            if evaluators is None:
                evaluators = {}
                self._thread_local.evaluators = evaluators
            evaluator = evaluators.get(task_name)
            if evaluator is None:
                evaluator = self._get_evalkit_evaluator(task_name, self.cfg.vbvr_reward_device)
                evaluators[task_name] = evaluator
            return evaluator

        evaluator = self._evaluators.get(task_name)
        if evaluator is None:
            evaluator = self._get_evalkit_evaluator(task_name, self.cfg.vbvr_reward_device)
            self._evaluators[task_name] = evaluator
        return evaluator

    def _score_jobs(self, jobs: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, float]]:
        if not jobs:
            return []
        if self._score_executor is None or len(jobs) == 1:
            return [(i, self._score_video_pair(**kwargs)) for i, kwargs in jobs]
        futures = [(i, self._score_executor.submit(self._score_video_pair, **kwargs)) for i, kwargs in jobs]
        return [(i, future.result()) for i, future in futures]

    def _score_video_pair(
        self,
        *,
        task_name: str,
        prompt: str,
        gen_video,
        gt_video,
        gt_video_path: str | None,
        gt_first_frame_path: str | None,
        gt_final_frame_path: str | None,
        sample_id: str,
    ) -> float:
        if self.cfg.vbvr_reward_keep_tmp:
            workdir = self._tmp_root / f"{sample_id}-{uuid.uuid4().hex[:8]}"
            workdir.mkdir(parents=True, exist_ok=False)
            return self._score_in_dir(
                workdir,
                task_name,
                prompt,
                gen_video,
                gt_video,
                gt_video_path=gt_video_path,
                gt_first_frame_path=gt_first_frame_path,
                gt_final_frame_path=gt_final_frame_path,
            )

        with tempfile.TemporaryDirectory(prefix=f"{sample_id}-", dir=self._tmp_root) as tmp:
            return self._score_in_dir(
                Path(tmp),
                task_name,
                prompt,
                gen_video,
                gt_video,
                gt_video_path=gt_video_path,
                gt_first_frame_path=gt_first_frame_path,
                gt_final_frame_path=gt_final_frame_path,
            )

    def _score_in_dir(
        self,
        workdir: Path,
        task_name: str,
        prompt: str,
        gen_video,
        gt_video,
        *,
        gt_video_path: str | None,
        gt_first_frame_path: str | None,
        gt_final_frame_path: str | None,
    ) -> float:
        gen_path = workdir / "generated.mp4"
        fallback_gt_path = workdir / "ground_truth.mp4"
        fallback_first_path = workdir / "first_frame.png"
        fallback_final_path = workdir / "final_frame.png"

        try:
            self._write_video(gen_path, gen_video)
            if gt_video_path is None:
                if gt_video is None:
                    raise RuntimeError("Cannot write fallback GT video without decoded GT frames")
                self._write_video(fallback_gt_path, gt_video)
                gt_video_path = str(fallback_gt_path)
            if gt_first_frame_path is None:
                if gt_video is None:
                    raise RuntimeError("Cannot write fallback GT first frame without decoded GT frames")
                self._write_image(fallback_first_path, gt_video[0])
                gt_first_frame_path = str(fallback_first_path)
            if gt_final_frame_path is None:
                if gt_video is None:
                    raise RuntimeError("Cannot write fallback GT final frame without decoded GT frames")
                self._write_image(fallback_final_path, gt_video[-1])
                gt_final_frame_path = str(fallback_final_path)

            result = self._evaluator(task_name).evaluate(
                {
                    "video_path": str(gen_path),
                    "task_name": task_name,
                    "gt_video_path": gt_video_path,
                    "gt_first_frame": gt_first_frame_path,
                    "gt_final_frame": gt_final_frame_path,
                    "prompt": prompt,
                    "no_ssim_fallback": True,
                },
                task_specific_only=bool(self.cfg.vbvr_reward_task_specific_only),
            )
            return float(max(0.0, min(1.0, result.get("score", 0.0))))
        except Exception as exc:
            with self._error_lock:
                self._error_count += 1
                error_count = self._error_count
            if self.trainer.rank == 0 and error_count <= 10:
                logger.warning("VBVR rule reward failed for task '{}': {}", task_name, exc)
            return 0.0

    def _write_video(self, path: Path, video) -> None:
        import cv2

        if video.ndim != 4:
            raise ValueError(f"Expected video shape (T,H,W,C), got {video.shape}")
        height, width = video.shape[1], video.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(self.cfg.vbvr_reward_fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open cv2.VideoWriter for {path}")
        try:
            for frame in video:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    @staticmethod
    def _write_image(path: Path, frame) -> None:
        import cv2

        ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write image {path}")

    @staticmethod
    def _to_uint8_videos(decoded: torch.Tensor):
        videos = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        return videos.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()

    def _task_name(self, meta: dict[str, Any], index: int) -> str:
        for key in ("sample_task_name", "task_name"):
            task_name = str(self._meta_item(meta, key, index, default="") or "")
            if task_name:
                return task_name
        tar_name = str(self._meta_item(meta, "sample_tar", index, default=""))
        task_name = Path(tar_name).stem
        if not task_name:
            raise RuntimeError("VBVRRuleReward could not infer task_name from sample_tar metadata")
        return task_name

    def _sample_id(self, meta: dict[str, Any], index: int) -> str:
        task = self._task_name(meta, index)
        sample_index = self._meta_item(meta, "sample_index_in_tar", index, default=index)
        return f"{task}-{sample_index}"

    @staticmethod
    def _meta_item(meta: dict[str, Any], key: str, index: int, *, default: Any = None) -> Any:
        value = meta.get(key, default)
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            item = value[index]
            return item.item() if item.ndim == 0 else item
        if isinstance(value, list | tuple):
            return value[index]
        return value

    def _meta_path(self, meta: dict[str, Any], key: str, index: int) -> str | None:
        value = self._meta_item(meta, key, index, default=None)
        if value is None:
            return None
        path = str(value)
        return path if path and Path(path).exists() else None
