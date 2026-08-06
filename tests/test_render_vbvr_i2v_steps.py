from types import SimpleNamespace

import pytest
import torch

from src.cli.render_vbvr_i2v_steps import _OdeStepRecorder, parse_args


class _FakeScheduler:
    def __init__(self, *, predict_x0: bool):
        self.config = SimpleNamespace(predict_x0=predict_x0)
        self.sigmas = torch.tensor([0.75, 0.0])
        self.step_index = None
        self.original_calls = 0

    def _init_step_index(self, timestep):
        self.step_index = 0

    def convert_model_output(self, model_output, *, sample):
        return sample.float() - self.sigmas[self.step_index] * model_output.float()

    def step(self, model_output, timestep, sample, return_dict=False):
        self.original_calls += 1
        if self.step_index is None:
            self._init_step_index(timestep)
        self.step_index += 1
        return (sample - model_output,)


class _FakePipe:
    def __init__(self, *, predict_x0: bool):
        self.scheduler = _FakeScheduler(predict_x0=predict_x0)

    def prepare_latents(self):
        latents = torch.full((1, 1, 2, 1, 1), 9.0)
        condition = torch.tensor([[[[[3.0]], [[4.0]]]]])
        mask = torch.tensor([[[[[0.0]], [[1.0]]]]])
        return latents, condition, mask


@pytest.mark.parametrize("predict_x0", [False, True])
def test_ode_recorder_captures_clean_endpoint_and_pins_conditioned_frame(predict_x0: bool):
    pipe = _FakePipe(predict_x0=predict_x0)
    recorder = _OdeStepRecorder(pipe)
    recorder.install()
    try:
        pipe.prepare_latents()
        sample = torch.tensor([[[[[8.0]], [[8.0]]]]])
        velocity = torch.tensor([[[[[2.0]], [[4.0]]]]])
        output = pipe.scheduler.step(velocity, torch.tensor(750.0), sample, return_dict=False)
    finally:
        recorder.restore()

    assert pipe.scheduler.original_calls == 1
    assert torch.equal(output[0], sample - velocity)
    assert recorder.sigmas == [0.75]
    assert recorder.timesteps == [750.0]
    assert torch.equal(recorder.pred_x0[0], torch.tensor([[[[[3.0]], [[5.0]]]]]))


def test_render_cli_requires_exactly_cps_noise_level(tmp_path):
    common = [
        "--eval_json",
        str(tmp_path / "eval.json"),
        "--model_path",
        str(tmp_path / "model"),
        "--output_dir",
        str(tmp_path / "output"),
    ]

    with pytest.raises(SystemExit):
        parse_args([*common, "--sampler", "cps"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--sampler", "unipc", "--noise_level", "0.7"])
    assert parse_args([*common, "--sampler", "cps", "--noise_level", "0.3"]).noise_level == 0.3
