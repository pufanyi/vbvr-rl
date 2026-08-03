from src.cli.validate_grpo_runtime import _selected_runtime


def test_selected_runtime_reads_config_and_cli_overrides(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("grpo_reward_fn: vbvr_rule\nattention_backend: _flash_3_hub\n")

    assert _selected_runtime(["--config", str(config)]) == ("vbvr_rule", "_flash_3_hub")
    assert _selected_runtime(
        [
            "--config",
            str(config),
            "--grpo_reward_fn",
            "neg_loss",
            "--attention_backend",
            "_native_flash",
        ]
    ) == ("neg_loss", "_native_flash")
