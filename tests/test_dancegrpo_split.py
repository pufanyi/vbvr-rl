from types import SimpleNamespace

import torch

from src.trainer.base_grpo_trainer import BaseGRPOTrainer
from src.trainer.dancegrpo_trainer import _split_group_indices
from src.trainer.dancegrpo_trainer import DanceGRPOTrainer


def test_split_group_indices_exact_fanout():
    assigned = [_split_group_indices(24, rank, 24) for rank in range(24)]

    assert all(len(groups) == 1 for groups in assigned)
    assert sorted(group for groups in assigned for group in groups) == list(range(24))


def test_split_group_indices_more_actors_than_group():
    assigned = [_split_group_indices(24, rank, 56) for rank in range(56)]

    assert sorted(group for groups in assigned for group in groups) == list(range(24))
    assert sum(1 for groups in assigned if groups) == 24


def test_split_group_indices_fewer_actors_than_group():
    assigned = [_split_group_indices(24, rank, 7) for rank in range(7)]

    assert sorted(group for groups in assigned for group in groups) == list(range(24))
    assert [len(groups) for groups in assigned] == [4, 4, 4, 3, 3, 3, 3]


class _DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_hidden_shapes = []
        self.seen_timestep_shapes = []

    def forward(self, *, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.seen_hidden_shapes.append(tuple(hidden_states.shape))
        self.seen_timestep_shapes.append(tuple(timestep.shape))
        return (torch.zeros_like(hidden_states),)


class _DummyExpandTimestepModel:
    boundary_timestep = 900

    def __init__(self, transformer):
        self.transformer = transformer
        self.latent_shape_called = False
        self.build_model_input_called = False
        self.build_timestep_input_called = False
        self.prepare_call_called = False

    def latent_shape_from_condition(self, condition):
        self.latent_shape_called = True
        return tuple(condition.shape)

    def _build_model_input(self, latent, condition):
        self.build_model_input_called = True
        if condition.shape != latent.shape:
            raise ValueError("expand-timestep condition and latent shapes must match")
        return latent + condition

    def _build_timestep_input(self, timesteps, latent, transformer):
        self.build_timestep_input_called = True
        return torch.ones(latent.shape[0], 3, device=latent.device, dtype=torch.bfloat16) * timesteps[:, None]

    def _prepare_transformer_call(self, transformer, hidden_states, timestep, encoder_hidden_states):
        self.prepare_call_called = True
        return hidden_states, timestep, encoder_hidden_states

    def _get_expert_for_timestep(self, timestep_val):
        return self.transformer


def test_dancegrpo_shared_initial_latents_use_model_latent_shape_for_5b():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    transformer = _DummyTransformer()
    model = _DummyExpandTimestepModel(transformer)
    trainer.model = model
    trainer.cfg = SimpleNamespace(dancegrpo_share_group_init_noise=True)

    condition = torch.zeros(2, 48, 41, 24, 24)
    latents = trainer._sample_group_initial_latents(condition)

    assert model.latent_shape_called
    assert tuple(latents.shape) == tuple(condition.shape)


def test_dancegrpo_policy_forward_uses_model_input_helpers_for_5b():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    transformer = _DummyTransformer()
    model = _DummyExpandTimestepModel(transformer)
    trainer.model = model
    trainer.cfg = SimpleNamespace(grpo_cfg_scale=1.0)

    latent = torch.zeros(2, 48, 3, 4, 4)
    condition = torch.ones_like(latent)
    prompt_embeds = torch.zeros(2, 8, 16)

    out = trainer._policy_forward(transformer, latent, condition, prompt_embeds, timestep_val=500.0)

    assert model.build_model_input_called
    assert model.build_timestep_input_called
    assert model.prepare_call_called
    assert tuple(out.shape) == tuple(latent.shape)
    assert transformer.seen_hidden_shapes == [tuple(latent.shape)]
    assert transformer.seen_timestep_shapes == [(2, 3)]


def test_grpo_reference_forward_uses_model_input_helpers_for_5b_fallback_ref():
    trainer = BaseGRPOTrainer.__new__(BaseGRPOTrainer)
    transformer = _DummyTransformer()
    model = _DummyExpandTimestepModel(transformer)
    trainer.model = model
    trainer.is_lora = False
    trainer.expert_parallel = False
    trainer.ref_transformers = {"transformer": transformer}

    latent = torch.zeros(2, 48, 3, 4, 4)
    condition = torch.ones_like(latent)
    prompt_embeds = torch.zeros(2, 8, 16)

    out = trainer._ref_forward(latent, condition, prompt_embeds, timestep_val=100.0)

    assert model.build_model_input_called
    assert model.build_timestep_input_called
    assert model.prepare_call_called
    assert tuple(out.shape) == tuple(latent.shape)
    assert transformer.seen_hidden_shapes == [tuple(latent.shape)]
    assert transformer.seen_timestep_shapes == [(2, 3)]
