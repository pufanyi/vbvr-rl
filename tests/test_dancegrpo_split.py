from types import SimpleNamespace

import torch

from src.models.wan_i2v import WanI2VForTraining
from src.trainer.base_grpo_trainer import BaseGRPOTrainer
from src.trainer.config import RLConfig
from src.trainer.dancegrpo_trainer import (
    DanceGRPOTrainer,
    _interleave_actor_ranks_by_node,
    _shared_prompt_assignment,
    _split_group_indices,
)
from src.trainer.rewards.neg_loss import NegLossReward


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


def test_interleave_actor_ranks_by_node_spreads_dispatch_across_nodes():
    actors = list(range(32, 64))

    ordered = _interleave_actor_ranks_by_node(actors, local_world_size=8)

    assert ordered[:8] == [32, 40, 48, 56, 33, 41, 49, 57]
    assert [rank // 8 for rank in ordered[:4]] == [4, 5, 6, 7]
    assert sorted(ordered) == actors


def test_shared_prompt_assignment_shards_prompts_and_groups_across_world():
    assignments = [_shared_prompt_assignment(rank, 64, 8, 24) for rank in range(64)]

    assert [item[0] for item in assignments[:8]] == list(range(8))
    assert assignments[0] == (0, 0, 8, [0, 8, 16])
    assert assignments[8] == (0, 1, 8, [1, 9, 17])
    for prompt_idx in range(8):
        groups = sorted(group for item in assignments if item[0] == prompt_idx for group in item[3])
        assert groups == list(range(24))


def test_rl_config_allows_full_actor_sync_without_lora():
    cfg = RLConfig(rl_actor_weight_sync="full", lora_rank=0)

    assert cfg.rl_actor_weight_sync == "full"
    assert cfg.lora_rank == 0


def test_rl_config_allows_flowcps_sde_formula():
    cfg = RLConfig(grpo_sde_formula="flowcps")

    assert cfg.grpo_sde_formula == "flowcps"


def test_flowcps_transition_matches_coefficients_preserving_formula():
    sample = torch.tensor([[[[[0.4, -0.2]]]]], dtype=torch.float32)
    model_output = torch.tensor([[[[[0.5, -0.3]]]]], dtype=torch.float32)
    sigma = 0.6
    sigma_prev = 0.25
    noise_level = 0.3

    mean, std = WanI2VForTraining._flowcps_transition_mean(
        sample=sample,
        model_output=model_output,
        sigma=sigma,
        sigma_prev=sigma_prev,
        noise_level=noise_level,
    )

    expected_std = sigma_prev * torch.sin(torch.tensor(noise_level * torch.pi / 2.0)).item()
    expected_coeff = (sigma_prev**2 - expected_std**2) ** 0.5
    expected_x0 = sample - sigma * model_output
    expected_x1 = sample + model_output * (1.0 - sigma)
    expected_mean = expected_x0 * (1.0 - sigma_prev) + expected_x1 * expected_coeff

    assert abs(std - expected_std) < 1e-6
    assert torch.allclose(mean, expected_mean)


def test_flowcps_log_prob_and_kl_use_official_unscaled_surrogates():
    mean = torch.zeros(2, 1, 1, 1, 2)
    sample = torch.ones_like(mean) * 0.25
    ref_mean = torch.ones_like(mean) * 0.5

    log_prob = WanI2VForTraining._sde_transition_log_prob("flowcps", sample, mean, noise_scale=0.1)
    kl_loss = WanI2VForTraining._sde_transition_kl_loss("flowcps", mean, ref_mean, noise_scale=0.1)

    assert torch.allclose(log_prob, torch.full((2,), -(0.25**2)))
    assert torch.allclose(kl_loss, torch.tensor(0.25))


def test_split_rollout_actor_skips_full_ft_reference_copies():
    trainer = BaseGRPOTrainer.__new__(BaseGRPOTrainer)
    trainer.rl_split_enabled = True
    trainer.is_inference_rank = True

    trainer._pre_fsdp_setup(SimpleNamespace(lora_rank=0, grpo_kl_coeff=0.004))

    assert trainer.is_lora is False
    assert trainer.ref_transformers == {}


def test_streamed_full_policy_dtype_parser_accepts_torch_names():
    assert DanceGRPOTrainer._dtype_from_name("torch.float32") is torch.float32
    assert DanceGRPOTrainer._dtype_from_name("bfloat16") is torch.bfloat16


class _DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_weight = torch.nn.Parameter(torch.empty((), dtype=torch.float32), requires_grad=False)
        self.seen_hidden_shapes = []
        self.seen_timestep_shapes = []
        self.seen_hidden_dtypes = []

    def forward(self, *, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.seen_hidden_shapes.append(tuple(hidden_states.shape))
        self.seen_timestep_shapes.append(tuple(timestep.shape))
        self.seen_hidden_dtypes.append(hidden_states.dtype)
        return (torch.zeros_like(hidden_states),)


class _DummyExpandTimestepModel:
    boundary_timestep = 900

    def __init__(self, transformer):
        self.transformer = transformer
        self.transformer_2 = None
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

    def _iter_transformer_selections(self, timesteps):
        selected = torch.arange(timesteps.shape[0], device=timesteps.device)
        return [("single", selected, self.transformer)]


class _DummyNegLossModel(_DummyExpandTimestepModel):
    boundary_timestep = 900
    num_train_timesteps = 2

    def _get_training_buffers(self, device):
        return (
            torch.tensor([0.2, 0.8], device=device),
            torch.tensor([950.0, 950.0], device=device),
            None,
        )

    def _prepare_transformer_call(self, transformer, hidden_states, timestep, encoder_hidden_states):
        self.prepare_call_called = True
        dtype = next(transformer.parameters()).dtype
        return hidden_states.to(dtype=dtype), timestep, encoder_hidden_states.to(dtype=dtype)


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


def test_neg_loss_reward_uses_model_helpers_and_transformer_dtype_for_5b():
    transformer = _DummyTransformer().to(dtype=torch.bfloat16)
    model = _DummyNegLossModel(transformer)
    trainer = SimpleNamespace(model=model)
    reward = NegLossReward(trainer, SimpleNamespace(fsdp=False))

    latents = torch.zeros(2, 48, 3, 4, 4, dtype=torch.float32)
    condition = torch.ones_like(latents)
    prompt_embeds = torch.zeros(2, 8, 16, dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.long)

    out = reward(latents, latents, condition, prompt_embeds, indices=indices)

    assert model.build_model_input_called
    assert model.build_timestep_input_called
    assert model.prepare_call_called
    assert tuple(out.shape) == (2,)
    assert transformer.seen_hidden_shapes == [tuple(latents.shape)]
    assert transformer.seen_timestep_shapes == [(2, 3)]
    assert transformer.seen_hidden_dtypes == [torch.bfloat16]


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
