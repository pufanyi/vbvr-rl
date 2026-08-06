from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.trainer.config import RLConfig
from src.trainer.rewards.vbvr_vlm import (
    EVAL_PROMPTS_SOURCE_SHA256,
    TASK_VLM_JUDGE_OUTPUT_REMINDER,
    VBVRVLMReward,
    parse_task_vlm_judge_score,
    parse_vlm_judge_score,
    task_vlm_judge_output_regex,
)
from src.trainer.rewards.vbvr_vlm_eval_prompts import EVAL_PROMPTS

G21_TASK = "G-21_multiple_occlusions_vertical_data-generator"


class _IdentityDecoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(latents.shape[0]))
        return latents


def _config(**overrides) -> SimpleNamespace:
    values = {
        "vlm_reward_base_url": "http://127.0.0.1:18080/v1",
        "vlm_reward_model": "qwen3.6-27b",
        "vlm_reward_api_key": "EMPTY",
        "vlm_reward_prompt_mode": "task_specific",
        "vlm_reward_system_prompt": None,
        "vlm_reward_system_prompt_path": None,
        "vlm_reward_num_frames": 3,
        "vlm_reward_include_gt_first_frame": True,
        "vlm_reward_decode_batch_size": 2,
        "vlm_reward_concurrency": 2,
        "vlm_reward_max_pending_jobs": 4,
        "vlm_reward_request_timeout_seconds": 5.0,
        "vlm_reward_max_retries": 0,
        "vlm_reward_retry_backoff_seconds": 0.0,
        "vlm_reward_max_new_tokens": 1024,
        "vlm_reward_image_max_edge": 32,
        "vlm_reward_jpeg_quality": 85,
        "vlm_reward_score_max": 100.0,
        "vlm_reward_use_structured_output": False,
        "vlm_reward_validate_service": False,
        "vlm_reward_fail_on_error": True,
        "vlm_reward_error_score": 0.0,
        "vlm_reward_log_first_n": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _trainer() -> SimpleNamespace:
    return SimpleNamespace(
        rank=0,
        tensor_parallel_enabled=False,
        tp_rank=0,
        model=_IdentityDecoder(),
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"score": 83.5, "reasoning": "mostly correct"}', (0.835, "mostly correct")),
        ('prefix {"reasoning": "done", "score": 100} suffix', (1.0, "done")),
        ("total_score: 75\nreason: acceptable", (0.75, "acceptable")),
        ("score: 25", (0.25, "")),
    ],
)
def test_parse_vlm_judge_score(response: str, expected: tuple[float, str]):
    score, reasoning = parse_vlm_judge_score(response)
    assert score == pytest.approx(expected[0])
    assert reasoning == expected[1]


@pytest.mark.parametrize(
    "response",
    [
        '{"score": 101, "reasoning": "out of range"}',
        '{"score": "nan", "reasoning": "not finite"}',
        "no json here",
    ],
)
def test_parse_vlm_judge_score_rejects_invalid_output(response: str):
    with pytest.raises(ValueError, match="does not contain a score"):
        parse_vlm_judge_score(response)


def test_vendored_task_prompts_are_pinned_and_well_formed():
    prompt_path = Path(__file__).parents[1] / "src/trainer/rewards/vbvr_vlm_eval_prompts.py"
    assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == EVAL_PROMPTS_SOURCE_SHA256
    assert len(EVAL_PROMPTS) == 100
    assert G21_TASK in EVAL_PROMPTS
    assert all("You will receive: the first frame (input) and the generated video." in p for p in EVAL_PROMPTS.values())
    assert all("Output format (exactly these lines):" in p for p in EVAL_PROMPTS.values())


def test_parse_task_vlm_judge_score_validates_dynamic_fields_and_weights():
    response = """mask_path_vadility_score: 70
occlusion_correctness_score: 80
elements_preservation_score: 90
mask_path_vadility_weight: 40
occlusion_correctness_weight: 35
elements_preservation_weight: 25
total_score: 79.5
reason: The mask mostly follows the requested path while preserving the scene."""
    score, reasoning = parse_task_vlm_judge_score(response, task_prompt=EVAL_PROMPTS[G21_TASK])
    assert score == pytest.approx(0.795)
    assert reasoning.startswith("The mask")
    output_regex = task_vlm_judge_output_regex(EVAL_PROMPTS[G21_TASK])
    assert re.fullmatch(output_regex, response)
    assert re.fullmatch(output_regex, "Analysis first.\n" + response) is None

    with pytest.raises(ValueError, match="exactly one 'reason' line"):
        parse_task_vlm_judge_score(response.rsplit("\n", maxsplit=1)[0], task_prompt=EVAL_PROMPTS[G21_TASK])
    with pytest.raises(ValueError, match="weights must sum to 100"):
        bad_weights = response.replace("elements_preservation_weight: 25", "elements_preservation_weight: 20")
        parse_task_vlm_judge_score(bad_weights, task_prompt=EVAL_PROMPTS[G21_TASK])


def test_task_specific_payload_uses_exact_prompt_start_and_generated_frames_only(monkeypatch: pytest.MonkeyPatch):
    reward = VBVRVLMReward(_trainer(), _config())
    captured: dict = {}
    response = """mask_path_vadility_score: 70
occlusion_correctness_score: 80
elements_preservation_score: 90
mask_path_vadility_weight: 40
occlusion_correctness_weight: 35
elements_preservation_weight: 25
total_score: 79.5
reason: Mostly correct."""

    def fake_request(method: str, path: str, payload: dict | None = None):
        captured.update({"method": method, "path": path, "payload": payload})
        return {"choices": [{"message": {"content": response}}]}

    monkeypatch.setattr(reward, "_request_json", fake_request)
    frames = [Image.new("RGB", (48, 24), color=(index * 20, 0, 0)) for index in range(3)]
    try:
        messages = reward._build_task_messages(
            task_prompt=EVAL_PROMPTS[G21_TASK],
            first_frame=Image.new("RGB", (48, 24), color="white"),
            generated_frames=frames,
        )
        assert reward._chat_completion(messages, task_prompt=EVAL_PROMPTS[G21_TASK]) == response
    finally:
        reward.close()

    payload = captured["payload"]
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert "response_format" not in payload
    assert payload["structured_outputs"] == {"regex": task_vlm_judge_output_regex(EVAL_PROMPTS[G21_TASK])}
    user_content = payload["messages"][0]["content"]
    assert user_content[0] == {"type": "text", "text": EVAL_PROMPTS[G21_TASK].strip()}
    assert sum(block["type"] == "image_url" for block in user_content) == 4
    assert not any("Expected final frame" in block.get("text", "") for block in user_content)
    assert user_content[-1] == {"type": "text", "text": TASK_VLM_JUDGE_OUTPUT_REMINDER}


def test_custom_payload_interleaves_references_and_chronological_frames(monkeypatch: pytest.MonkeyPatch):
    reward = VBVRVLMReward(
        _trainer(),
        _config(vlm_reward_prompt_mode="custom", vlm_reward_use_structured_output=True),
    )
    captured: dict = {}

    def fake_request(method: str, path: str, payload: dict | None = None):
        captured.update({"method": method, "path": path, "payload": payload})
        return {"choices": [{"message": {"content": '{"score": 62.5, "reasoning": "partial"}'}}]}

    monkeypatch.setattr(reward, "_request_json", fake_request)
    frames = [Image.new("RGB", (48, 24), color=(index * 20, 0, 0)) for index in range(3)]
    try:
        messages = reward._build_custom_messages(
            task_name="G-1",
            prompt="move the block",
            first_frame=Image.new("RGB", (48, 24), color="white"),
            final_frame=Image.new("RGB", (48, 24), color="black"),
            generated_frames=frames,
        )
        response = reward._chat_completion(messages)
    finally:
        reward.close()

    assert json.loads(response)["score"] == 62.5
    assert captured["method"] == "POST"
    assert captured["path"] == "/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == "qwen3.6-27b"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["json_schema"]["strict"] is True
    user_content = payload["messages"][1]["content"]
    assert sum(block["type"] == "image_url" for block in user_content) == 5
    generated_labels = [
        block["text"] for block in user_content if block.get("type") == "text" and "Generated frame" in block["text"]
    ]
    assert generated_labels == ["Generated frame 1/3:", "Generated frame 2/3:", "Generated frame 3/3:"]


def test_image_encoding_preserves_native_size_and_only_downscales():
    reward = VBVRVLMReward(_trainer(), _config(vlm_reward_image_max_edge=512))
    try:
        native_content = reward._image_content(Image.new("RGB", (512, 384), color="white"))
        large_content = reward._image_content(Image.new("RGB", (1024, 512), color="white"))
    finally:
        reward.close()

    def decoded_size(content: dict) -> tuple[int, int]:
        data_url = content["image_url"]["url"]
        encoded = data_url.split(",", maxsplit=1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            return image.size

    assert decoded_size(native_content) == (512, 384)
    assert decoded_size(large_content) == (512, 256)


def test_vlm_reward_decodes_in_batches_and_preserves_sample_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first_path = tmp_path / "first.png"
    final_path = tmp_path / "final.png"
    Image.new("RGB", (8, 8), color="white").save(first_path)
    Image.new("RGB", (8, 8), color="black").save(final_path)

    trainer = _trainer()
    reward = VBVRVLMReward(trainer, _config())
    generated = torch.stack(
        [
            torch.full((3, 5, 4, 4), -1.0),
            torch.full((3, 5, 4, 4), 0.0),
            torch.full((3, 5, 4, 4), 1.0),
        ]
    )
    gt = torch.zeros_like(generated)
    meta = {
        "sample_prompt": ["p0", "p1", "p2"],
        "sample_task_name": ["t0", "t1", "t2"],
        "sample_gt_first_frame": [str(first_path)] * 3,
        # Task-specific mode deliberately does not consume GT final frames.
        "sample_gt_final_frame": [None] * 3,
    }

    def fake_score(**kwargs) -> float:
        return float(kwargs["generated_video"][0, 0, 0, 0]) / 255.0

    monkeypatch.setattr(reward, "_score_video", fake_score)
    try:
        pending = reward.submit(generated, gt, torch.empty(0), torch.empty(0), meta=meta)
        scores = pending.result()
    finally:
        reward.close()

    assert trainer.model.batch_sizes == [2, 1]
    assert scores.tolist() == pytest.approx([0.0, 128.0 / 255.0, 1.0])


def test_vlm_reward_error_policy_can_fail_closed_or_use_fallback():
    video = np.zeros((3, 8, 8, 3), dtype=np.uint8)

    fallback = VBVRVLMReward(
        _trainer(),
        _config(
            vlm_reward_prompt_mode="custom",
            vlm_reward_include_gt_first_frame=False,
            vlm_reward_fail_on_error=False,
            vlm_reward_error_score=0.2,
        ),
    )
    try:
        assert fallback._score_video(
            task_name="missing-ref",
            prompt="test",
            generated_video=video,
            gt_video=None,
            gt_first_frame_path=None,
            gt_final_frame_path=None,
        ) == pytest.approx(0.2)
    finally:
        fallback.close()

    strict = VBVRVLMReward(
        _trainer(),
        _config(vlm_reward_prompt_mode="custom", vlm_reward_include_gt_first_frame=False),
    )
    try:
        with pytest.raises(RuntimeError, match="VBVR VLM reward failed"):
            strict._score_video(
                task_name="missing-ref",
                prompt="test",
                generated_video=video,
                gt_video=None,
                gt_first_frame_path=None,
                gt_final_frame_path=None,
            )
    finally:
        strict.close()


def test_rl_config_validates_vlm_reward_fields():
    config = RLConfig(grpo_reward_fn="vbvr_vlm")
    assert config.vlm_reward_model == "qwen3.6-27b"
    assert config.vlm_reward_prompt_mode == "task_specific"
    with pytest.raises(ValueError, match="vlm_reward_jpeg_quality"):
        RLConfig(vlm_reward_jpeg_quality=0)
    with pytest.raises(ValueError, match="nonnegative integer fields"):
        RLConfig(vlm_reward_max_retries=-1)
    with pytest.raises(ValueError, match="requires vlm_reward_base_url"):
        RLConfig(grpo_reward_fn="vbvr_vlm", vlm_reward_base_url="")
    with pytest.raises(ValueError, match="require vlm_reward_include_gt_first_frame=true"):
        RLConfig(grpo_reward_fn="vbvr_vlm", vlm_reward_include_gt_first_frame=False)
    with pytest.raises(ValueError, match="require vlm_reward_score_max=100"):
        RLConfig(grpo_reward_fn="vbvr_vlm", vlm_reward_score_max=10)
    with pytest.raises(ValueError, match="require vlm_reward_use_structured_output=false"):
        RLConfig(grpo_reward_fn="vbvr_vlm", vlm_reward_use_structured_output=True)
