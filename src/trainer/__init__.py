# Workaround: In some PyTorch versions, DeviceMesh.size() passes string
# dimension names directly to _MeshLayout.__getitem__ which only accepts int
# indices, causing TypeError.  Patch size() to convert name → index first.
from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh

_orig_device_mesh_size = _DeviceMesh.size

def _device_mesh_size_patched(self, mesh_dim=None):
    if isinstance(mesh_dim, str) and self.mesh_dim_names:
        mesh_dim = list(self.mesh_dim_names).index(mesh_dim)
    return _orig_device_mesh_size(self, mesh_dim)

_DeviceMesh.size = _device_mesh_size_patched

from src.trainer.base_grpo_trainer import BaseGRPOTrainer
from src.trainer.base_rl_trainer import BaseRLTrainer
from src.trainer.checkpoint import TrainState
from src.trainer.config import RLConfig, SFTConfig, TrainConfig
from src.trainer.cos_trainer import COSTrainer
from src.trainer.dancegrpo_trainer import DanceGRPOTrainer
from src.trainer.grpo_trainer import GRPOTrainer
from src.trainer.i2v_trainer import I2VTrainer

__all__ = [
    "BaseGRPOTrainer",
    "BaseRLTrainer",
    "COSTrainer",
    "DanceGRPOTrainer",
    "GRPOTrainer",
    "I2VTrainer",
    "RLConfig",
    "SFTConfig",
    "TrainConfig",
    "TrainState",
]
