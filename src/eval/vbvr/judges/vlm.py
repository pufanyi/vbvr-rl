"""VLM-based judge — asks a vision-language model whether the video completes the task."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from ..frames import sample_frames
from ..types import EvalSample, SampleScore
from .base import Judge

_JUDGE_SYSTEM = (
    "You are an expert evaluator of video-based visual reasoning. "
    "You will be shown (1) the task description, (2) the expected final frame "
    "(ground truth), and (3) several frames sampled uniformly from a generated "
    "video. Decide how well the generated video completes the task.\n\n"
    "Respond with ONLY a JSON object, no extra text:\n"
    '  {"score": <integer 0-10>, "reasoning": "<one short sentence>"}\n'
    "Score rubric (strict):\n"
    "  10 — task clearly solved; final state matches GT intent\n"
    "   7 — mostly correct; minor deviation from GT\n"
    "   4 — partially correct; key step missing or wrong\n"
    "   1 — attempted but fails the task\n"
    "   0 — irrelevant, broken, or no reasoning shown"
)


def _parse_score(response: str) -> tuple[float, str]:
    """Extract score ∈ [0,1] and reasoning from judge text. Robust to extra prose."""
    # Preferred: strict JSON object somewhere in the output
    m = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            raw = float(obj.get("score", -1))
            if 0 <= raw <= 10:
                return raw / 10.0, str(obj.get("reasoning", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: "X/10"
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", response)
    if m:
        raw = float(m.group(1))
        return max(0.0, min(10.0, raw)) / 10.0, ""
    # Last resort: "score: X"
    m = re.search(r"score\s*[:=]\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
    if m:
        raw = float(m.group(1))
        return max(0.0, min(10.0, raw)) / 10.0, ""
    return 0.0, ""


class VLMJudge(Judge):
    """
    Score a video by asking a VLM whether it solves the task.

    The VLM sees: task prompt, GT final frame, and K uniformly-sampled frames
    from the generated video. Output is parsed as JSON.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-26B-A4B-it",
        num_frames: int = 6,
        include_gt_first_frame: bool = False,
        max_new_tokens: int = 128,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str | None = None,
    ) -> None:
        self.name = f"vlm:{model_id}"
        self.num_frames = num_frames
        self.include_gt_first_frame = include_gt_first_frame
        self.max_new_tokens = max_new_tokens
        self.device = device

        logger.info("Loading VLM judge: {}", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        load_kwargs: dict = {"torch_dtype": torch_dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
        if device_map is None:
            self.model = self.model.to(device)
        self.model.eval()

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _build_messages(self, sample: EvalSample, frames: list[Image.Image]) -> list[dict]:
        content: list[dict] = [
            {"type": "text", "text": f"Task description:\n{sample.prompt}"},
            {"type": "text", "text": "Expected final frame (ground truth):"},
            {"type": "image", "image": self._load_image(sample.gt_final_frame)},
        ]
        if self.include_gt_first_frame:
            content.append({"type": "text", "text": "Starting frame (given to the model):"})
            content.append({"type": "image", "image": self._load_image(sample.gt_first_frame)})
        content.append(
            {
                "type": "text",
                "text": f"Frames sampled uniformly from the generated video ({len(frames)} frames, earliest first):",
            }
        )
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": "Now output the JSON verdict."})
        return [
            {"role": "system", "content": [{"type": "text", "text": _JUDGE_SYSTEM}]},
            {"role": "user", "content": content},
        ]

    @torch.inference_mode()
    def _generate(self, messages: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_tokens = outputs[0, input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def score(self, sample: EvalSample) -> SampleScore:
        try:
            frames = sample_frames(sample.video_path, self.num_frames)
            if not frames:
                return SampleScore(
                    task_name=sample.task_name,
                    video_idx=sample.video_idx,
                    split=sample.split,
                    domain=sample.domain,
                    score=0.0,
                    error="empty video",
                )
            messages = self._build_messages(sample, frames)
            response = self._generate(messages)
            score, reasoning = _parse_score(response)
            return SampleScore(
                task_name=sample.task_name,
                video_idx=sample.video_idx,
                split=sample.split,
                domain=sample.domain,
                score=score,
                judge_response=response,
                details={"reasoning": reasoning, "num_frames": len(frames)},
            )
        except Exception as e:
            logger.exception("Judge failed on {}/{}", sample.task_name, sample.video_idx)
            return SampleScore(
                task_name=sample.task_name,
                video_idx=sample.video_idx,
                split=sample.split,
                domain=sample.domain,
                score=0.0,
                error=f"{type(e).__name__}: {e}",
            )
