"""VBVR EvalKit rule reward for GRPO.

This reward mirrors the rule-based evaluation path used for VBVR reporting:
decode generated and GT latents to temporary videos, then call the vendored
EvalKit task-specific evaluator for the sample's task.
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, ClassVar

import torch
from loguru import logger

from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.registry import register_reward

_EVALKIT = Path(__file__).resolve().parents[3] / "third_party" / "VBVR-EvalKit"


@register_reward("vbvr_rule")
class VBVRRuleReward(BaseReward):
    """Reward = VBVR EvalKit task-specific rule score in [0, 1]."""

    requires_vae: ClassVar[bool] = True

    def __init__(self, trainer, cfg) -> None:
        super().__init__(trainer, cfg)
        if str(_EVALKIT) not in sys.path:
            sys.path.insert(0, str(_EVALKIT))

        from vbvr_bench.evaluators import TASK_EVALUATOR_MAP, get_evaluator

        self._task_evaluator_map = TASK_EVALUATOR_MAP
        self._get_evalkit_evaluator = get_evaluator
        self._evaluators: dict[str, Any] = {}
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

        for start in range(0, B, decode_bs):
            end = min(B, start + decode_bs)
            decoded = self.trainer.model.decode_latents(
                torch.cat(
                    [
                        generated_latents[start:end],
                        gt_video_latents[start:end],
                    ],
                    dim=0,
                )
            )
            videos = self._to_uint8_videos(decoded)
            n = end - start
            gen_videos = videos[:n]
            gt_videos = videos[n:]

            for local_i in range(n):
                i = start + local_i
                task_name = self._task_name(meta, i)
                if not self._is_supported(task_name):
                    continue

                prompt = self._meta_item(meta, "sample_prompt", i, default="")
                score = self._score_video_pair(
                    task_name=task_name,
                    prompt=str(prompt or ""),
                    gen_video=gen_videos[local_i],
                    gt_video=gt_videos[local_i],
                    sample_id=self._sample_id(meta, i),
                )
                rewards[i] = score

            del decoded

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
        evaluator = self._evaluators.get(task_name)
        if evaluator is None:
            evaluator = self._get_evalkit_evaluator(task_name, self.cfg.vbvr_reward_device)
            self._evaluators[task_name] = evaluator
        return evaluator

    def _score_video_pair(
        self,
        *,
        task_name: str,
        prompt: str,
        gen_video,
        gt_video,
        sample_id: str,
    ) -> float:
        if self.cfg.vbvr_reward_keep_tmp:
            workdir = self._tmp_root / f"{sample_id}-{uuid.uuid4().hex[:8]}"
            workdir.mkdir(parents=True, exist_ok=False)
            return self._score_in_dir(workdir, task_name, prompt, gen_video, gt_video)

        with tempfile.TemporaryDirectory(prefix=f"{sample_id}-", dir=self._tmp_root) as tmp:
            return self._score_in_dir(Path(tmp), task_name, prompt, gen_video, gt_video)

    def _score_in_dir(self, workdir: Path, task_name: str, prompt: str, gen_video, gt_video) -> float:
        gen_path = workdir / "generated.mp4"
        gt_path = workdir / "ground_truth.mp4"
        first_path = workdir / "first_frame.png"
        final_path = workdir / "final_frame.png"

        try:
            self._write_video(gen_path, gen_video)
            self._write_video(gt_path, gt_video)
            self._write_image(first_path, gt_video[0])
            self._write_image(final_path, gt_video[-1])

            result = self._evaluator(task_name).evaluate(
                {
                    "video_path": str(gen_path),
                    "task_name": task_name,
                    "gt_video_path": str(gt_path),
                    "gt_first_frame": str(first_path),
                    "gt_final_frame": str(final_path),
                    "prompt": prompt,
                    "no_ssim_fallback": True,
                },
                task_specific_only=bool(self.cfg.vbvr_reward_task_specific_only),
            )
            return float(max(0.0, min(1.0, result.get("score", 0.0))))
        except Exception as exc:
            self._error_count += 1
            if self.trainer.rank == 0 and self._error_count <= 10:
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
