import unittest

import torch

from src.models.cos_path import compute_cos_path


class TestCOSPathValidation(unittest.TestCase):
    def test_target_cosine_accepts_matching_chain_lengths(self):
        sigma = torch.rand(2, 1, 1, 1, 1)
        noise = torch.randn(2, 3, 2, 4, 4)
        waypoints = [torch.randn(2, 3, 2, 4, 4) for _ in range(3)]

        x_t, target = compute_cos_path("target_cosine", sigma, [0.9, 0.8], noise, waypoints)

        self.assertEqual(x_t.shape, noise.shape)
        self.assertEqual(target.shape, noise.shape)

    def test_target_cosine_rejects_mismatched_tau_count(self):
        sigma = torch.rand(1, 1, 1, 1, 1)
        noise = torch.randn(1, 1, 1, 1, 1)
        waypoints = [torch.randn(1, 1, 1, 1, 1) for _ in range(3)]

        with self.assertRaisesRegex(ValueError, r"len\(taus\) == len\(waypoints\) - 1"):
            compute_cos_path("target_cosine", sigma, [0.9, 0.8, 0.7], noise, waypoints)

    def test_target_cosine_supports_single_waypoint_chain(self):
        sigma = torch.rand(2, 1, 1, 1, 1)
        noise = torch.randn(2, 3, 2, 4, 4)
        final = torch.randn(2, 3, 2, 4, 4)

        x_t, target = compute_cos_path("target_cosine", sigma, [], noise, [final])

        self.assertEqual(x_t.shape, noise.shape)
        self.assertEqual(target.shape, noise.shape)


if __name__ == "__main__":
    unittest.main()
