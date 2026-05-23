import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as st_load

from src.precompute.maze_generator import (
    MazeSpec,
    build_line_waypoint_from_sample,
    build_maze_sample,
    render_video_from_metadata,
)
from src.precompute.maze_webdataset import GenConfig, _sample_to_json_blob, _write_encoded_samples


class TestMazeCOSChain(unittest.TestCase):
    def _sample_pair(self):
        rng = np.random.default_rng(123)
        spec = MazeSpec(
            cell_h=6,
            cell_w=6,
            cell_px=8,
            num_frames=9,
            difficulty_names=("easy",),
            render_mode="moving_ball",
            max_generation_attempts=512,
        )
        final_video, final_sample = build_maze_sample(spec, rng, sample_seed=123)
        line_video, line_sample = build_line_waypoint_from_sample(final_sample, completion_fraction=0.5)
        return line_video, line_sample, final_video, final_sample

    def test_line_waypoint_completes_halfway(self):
        line_video, line_sample, final_video, final_sample = self._sample_pair()

        self.assertEqual(line_video.shape, final_video.shape)
        self.assertEqual(line_sample.render_mode, "growing_path_line")
        self.assertEqual(line_sample.render_metadata["line_completion_frame"], 4)
        self.assertEqual(tuple(line_sample.frame_positions_cell[4]), tuple(float(x) for x in final_sample.goal))
        self.assertEqual(tuple(line_sample.frame_positions_cell[-1]), tuple(float(x) for x in final_sample.goal))

        reconstructed = render_video_from_metadata(_sample_to_json_blob(line_sample))
        np.testing.assert_array_equal(reconstructed, line_video)

    def test_encoded_samples_write_multi_latent_schema(self):
        _, line_sample, _, final_sample = self._sample_pair()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GenConfig(
                output_dir=tmpdir,
                model_path="unused",
                num_samples=1,
                cos_chain_mode="line_to_moving_ball",
            )
            tar_path = Path(tmpdir) / "sample.tar"
            encoded = [
                (
                    [torch.zeros(2, 1, 1, 1), torch.ones(2, 1, 1, 1)],
                    torch.zeros(1, 1, 1, 1),
                    torch.zeros(3, 4),
                    [line_sample, final_sample],
                    0,
                )
            ]
            with tarfile.open(tar_path, "w") as tar:
                _write_encoded_samples(cfg, tar, "sft", encoded, split_offset=0, local_start=0)

            with tarfile.open(tar_path, "r") as tar:
                st_member = tar.extractfile("0000000.safetensors")
                json_member = tar.extractfile("0000000.json")
                assert st_member is not None
                assert json_member is not None
                tensors = st_load(st_member.read())
                meta = json.loads(json_member.read().decode("utf-8"))

        self.assertIn("latents_0", tensors)
        self.assertIn("latents_1", tensors)
        self.assertNotIn("latents", tensors)
        self.assertEqual(meta["num_latents"], 2)
        self.assertEqual(meta["cos_chain_mode"], "line_to_moving_ball")
        self.assertEqual(len(meta["maze_chain"]), 2)


if __name__ == "__main__":
    unittest.main()
