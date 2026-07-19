from types import SimpleNamespace

import pytest
import torch

from src.trainer.base_grpo_trainer import BaseGRPOTrainer
from src.trainer.base_rl_trainer import BaseRLTrainer
from src.trainer.config import RLConfig


def _tp_cfg(**overrides) -> RLConfig:
    values = {
        "tensor_parallel_size": 2,
        "fsdp": True,
        "hsdp": False,
        "lora_rank": 0,
        "use_liger_kernel": False,
        "torch_compile": False,
        "train_text_encoder": False,
        "expert_parallel": False,
        "batch_size": 4,
        "grpo_shared_prompt_batch": False,
    }
    values.update(overrides)
    return RLConfig(**values)


def _trainer_state(rank: int = 0, world_size: int = 8):
    return SimpleNamespace(
        rl_split_enabled=False,
        expert_parallel=False,
        world_size=world_size,
        rank=rank,
        dp_rank=rank,
        dp_size=world_size,
    )


@pytest.mark.parametrize(
    ("rank", "expected_dp_rank", "expected_tp_rank"),
    [(0, 0, 0), (1, 0, 1), (2, 1, 0), (5, 2, 1), (7, 3, 1)],
)
def test_tp2_rank_mapping(rank: int, expected_dp_rank: int, expected_tp_rank: int):
    trainer = _trainer_state(rank=rank)
    BaseRLTrainer._init_tensor_parallel(trainer, _tp_cfg())

    assert trainer.tensor_parallel_enabled
    assert trainer.dp_size == 4
    assert trainer.dp_rank == expected_dp_rank
    assert trainer.tp_rank == expected_tp_rank


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"fsdp": False}, "fsdp=False"),
        ({"hsdp": True}, "hsdp=True"),
        ({"lora_rank": 8}, "LoRA"),
        ({"use_liger_kernel": True}, "Liger RMSNorm"),
        ({"torch_compile": True}, "torch_compile"),
        ({"train_text_encoder": True}, "train_text_encoder"),
    ],
)
def test_tp_rejects_unsupported_combinations(override: dict, message: str):
    trainer = _trainer_state()
    with pytest.raises(ValueError, match=message):
        BaseRLTrainer._init_tensor_parallel(trainer, _tp_cfg(**override))


def test_tp_requires_world_size_divisibility():
    trainer = _trainer_state(world_size=7)
    with pytest.raises(ValueError, match="must be divisible"):
        BaseRLTrainer._init_tensor_parallel(trainer, _tp_cfg())


def test_shared_prompt_batch_is_bounded_by_dp_size():
    trainer = _trainer_state()
    with pytest.raises(ValueError, match="batch_size must be <= DP size"):
        BaseRLTrainer._init_tensor_parallel(
            trainer,
            _tp_cfg(grpo_shared_prompt_batch=True, batch_size=8),
        )


def test_tensor_parallel_size_must_be_positive():
    with pytest.raises(ValueError, match="tensor_parallel_size must be > 0"):
        RLConfig(tensor_parallel_size=0)


def test_replay_offload_moves_frozen_inference_models_to_cpu():
    class RecordingModule:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))
            return self

    text_encoder = RecordingModule()
    vae = RecordingModule()
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(grpo_offload_inference_models=True),
        model=SimpleNamespace(text_encoder=text_encoder, vae=vae),
        device=torch.device("cpu"),
    )

    BaseGRPOTrainer._offload_inference_models_for_replay(trainer)

    assert text_encoder.devices == ["cpu"]
    assert vae.devices == ["cpu"]


def test_reward_restores_offloaded_vae_for_latent_batches():
    class RecordingModule:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))
            return self

    class VAEReward:
        requires_vae = True
        requires_policy_forward = False

        def __call__(self, generated, _gt, _condition, _prompt, *, meta):
            assert meta == {}
            return torch.zeros(generated.shape[0])

    vae = RecordingModule()
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(grpo_offload_inference_models=True),
        reward_fn=VAEReward(),
        tensor_parallel_enabled=False,
        tp_rank=0,
        model=SimpleNamespace(vae=vae),
        device=torch.device("cpu"),
    )
    generated = torch.zeros(2, 1)

    rewards = BaseGRPOTrainer._compute_reward(
        trainer,
        generated,
        generated,
        generated,
        generated,
        meta={},
    )

    assert vae.devices == ["cpu"]
    assert rewards.shape == (2,)
