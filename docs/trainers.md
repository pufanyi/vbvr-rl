# Trainer Deep Dive

## 1. I2VTrainer (`src/trainer/i2v_trainer.py`)
**Purpose**: Standard Supervised Fine-Tuning.
**Logic**:
- Samples a random $\sigma \in [0, 1]$.
- Predicts $v = \frac{dx}{d\sigma}$.
- Minimizes MSE between predicted and target velocity.
**Usage**:
```bash
.venv/bin/python -m src.cli.train_i2v --config configs/train_sft.yaml
```

## 2. COSTrainer (`src/trainer/cos_trainer.py`)
**Purpose**: Chain-of-Step (Piecewise Flow Matching).
**Logic**:
- Splits the path into segments: `Noise -> Search -> Final`.
- Segment 1 ($\sigma > \tau$): Targeted at intermediate reasoning (e.g., maze BFS).
- Segment 2 ($\sigma < \tau$): Targeted at final execution (e.g., ball moving).
- **Dual Expert Step**: Automatically ensures both high and low experts receive training signal in every batch by sampling $\sigma$ from both ranges.
**Usage**:
```bash
.venv/bin/python -m src.cli.train_cos --config configs/train_cos.yaml
```

## 3. GRPOTrainer (`src/trainer/grpo_trainer.py`)
**Purpose**: Reinforcement Learning for Video.
**Logic**:
- **Sampling**: Performs a full SDE rollout to generate videos.
- **Advantage**: Calculates rewards for a group of $G$ videos, then computes z-scores.
- **Optimization**: Standard PPO-style clipped loss with a KL penalty against a frozen reference policy.
**Usage**:
```bash
.venv/bin/python -m src.cli.train_grpo --config configs/train_grpo.yaml
```

## 4. CorrectionTrainer (`src/trainer/i2v_correction_trainer.py`)
**Purpose**: Drift Correction.
**Logic**:
- Uses an EMA Teacher to generate a "drifting" path.
- The Student is trained to "correct" the drift by pointing back to the Ground Truth video from the teacher's noisy state.
- Combines standard SFT loss with a weighted correction loss.
**Usage**:
```bash
.venv/bin/python -m src.cli.train_i2v_correction --config configs/train_correction.yaml
```
