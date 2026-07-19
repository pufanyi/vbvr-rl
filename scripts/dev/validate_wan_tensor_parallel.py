"""Small distributed numerical/composition checks for Wan tensor parallelism.

Examples:

    PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=2 \
        scripts/dev/validate_wan_tensor_parallel.py --mode tp

    PYTHONPATH=. .venv/bin/torchrun --standalone --nproc_per_node=8 \
        scripts/dev/validate_wan_tensor_parallel.py --mode tp-fsdp
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tempfile
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from src.trainer.tensor_parallel import parallelize_wan_transformer
from src.trainer.utils import shard_transformer


def _build_model() -> WanTransformer3DModel:
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=32,
    )


def _inputs(device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden_states = torch.randn((1, 4, 1, 4, 4), device=device, generator=generator)
    encoder_hidden_states = torch.randn((1, 6, 16), device=device, generator=generator)
    timestep = torch.tensor([500.0], device=device)
    return hidden_states, encoder_hidden_states, timestep


def _forward(
    model: WanTransformer3DModel,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    hidden_states, encoder_hidden_states, timestep = inputs
    return model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        return_dict=False,
    )[0]


def _run_tp_numerics(device: torch.device) -> None:
    if dist.get_world_size() != 2:
        raise RuntimeError("--mode tp requires exactly two ranks")
    torch.manual_seed(1234)
    model = _build_model().to(device=device, dtype=torch.float32)
    reference = deepcopy(model)
    inputs = _inputs(device, 777)

    reference_output = _forward(reference, inputs)
    reference_output.square().mean().backward()

    tp_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
    stats = parallelize_wan_transformer(model, tp_mesh)
    tp_output = _forward(model, inputs)
    tp_output.square().mean().backward()

    output_diff = (reference_output - tp_output).abs().max()
    grad_diff = torch.zeros((), device=device)
    reference_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        reference_grad = reference_params[name].grad
        tp_grad = parameter.grad
        if reference_grad is None or tp_grad is None:
            if reference_grad is not None or tp_grad is not None:
                raise RuntimeError(f"Gradient presence mismatch for {name}")
            continue
        full_tp_grad = tp_grad.full_tensor() if isinstance(tp_grad, DTensor) else tp_grad
        grad_diff = torch.maximum(grad_diff, (reference_grad - full_tp_grad).abs().max())

    result = torch.stack([output_diff, grad_diff])
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(
            f"{stats} output_max_abs_diff={result[0].item():.8e} grad_max_abs_diff={result[1].item():.8e}",
            flush=True,
        )
    if result[0].item() > 2e-5 or result[1].item() > 2e-5:
        raise RuntimeError(f"TP numerical mismatch: {result.tolist()}")


def _run_tp_fsdp(device: torch.device) -> None:
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2:
        raise RuntimeError("--mode tp-fsdp requires an even world size of at least four")
    rank = dist.get_rank()
    tp_size = 2
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    torch.manual_seed(1234)
    model = _build_model()

    mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    tp_mesh = mesh_2d["tp"]
    parallelize_wan_transformer(model, tp_mesh)
    shard_transformer(
        model,
        mesh_2d["dp"],
        MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    output = _forward(model, _inputs(device, 777 + dp_rank))
    output.square().mean().backward()
    optimizer.step()

    # Exercise the same model/optimizer state-dict conversion used by DCP and
    # rank-local optimizer checkpoint shards.
    model_state = get_model_state_dict(model)
    optimizer_state = get_optimizer_state_dict(model, optimizer)
    if not model_state or not optimizer_state.get("state"):
        raise RuntimeError("TP+FSDP state-dict conversion returned an empty state")
    serialized_optimizer = io.BytesIO()
    torch.save(optimizer_state, serialized_optimizer)
    serialized_optimizer.seek(0)
    roundtrip_optimizer_state = torch.load(serialized_optimizer, map_location="cpu", weights_only=False)
    set_optimizer_state_dict(model, optimizer, roundtrip_optimizer_state)

    # DCP is the model-weight format used by production checkpoints. Use one
    # shared temporary directory to verify the composed 2D DTensor layout can
    # be planned, written, read, and installed back into the model.
    checkpoint_dir = [tempfile.mkdtemp(prefix="wan-tp-fsdp-dcp-") if rank == 0 else None]
    dist.broadcast_object_list(checkpoint_dir, src=0)
    checkpoint_path = checkpoint_dir[0]
    if checkpoint_path is None:
        raise RuntimeError("Rank 0 did not publish the temporary DCP path")
    dcp.save({"model": model_state}, checkpoint_id=checkpoint_path)
    loaded_state = {"model": get_model_state_dict(model)}
    dcp.load(loaded_state, checkpoint_id=checkpoint_path)
    set_model_state_dict(model, loaded_state["model"])
    dist.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path)

    checksum = output.float().sum()
    pair_checksums = [torch.empty_like(checksum) for _ in range(tp_size)]
    dist.all_gather(pair_checksums, checksum, group=tp_mesh.get_group())
    pair_diff = torch.stack(pair_checksums).sub(pair_checksums[0]).abs().max()
    dist.all_reduce(pair_diff, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"TP{tp_size}xFSDP{dp_size} AdamW/optimizer/DCP check passed; "
            f"tp_output_diff={pair_diff.item():.3e} "
            f"peak_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.1f}",
            flush=True,
        )
    if pair_diff.item() > 1e-6:
        raise RuntimeError(f"TP pair outputs diverged: {pair_diff.item()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tp", "tp-fsdp"), required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        if args.mode == "tp":
            _run_tp_numerics(device)
        else:
            _run_tp_fsdp(device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
