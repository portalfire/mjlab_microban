# MjLab Microban

<img width="35%" align="right" alt="image" src="https://github.com/user-attachments/assets/38848ed5-ef34-44f3-ad44-f2137ac0347b" />

[![License: Apache-2.0](https://img.shields.io/badge/Software-Apache--2.0-yellow.svg)](LICENSE)

This repository contains Reinforcement Learning (RL) environments for Microban, a compact, low-cost, fully open-source small humanoid robot. 

If you are interested in learning more about Microban, or even building your own, check out the [Microban repository](https://github.com/MarcDcls/microban).

The environments are built using the [MjLab](https://github.com/mujocolab/mjlab) framework.
A velocity control task is currently implemented, allowing the robot to follow target linear and angular velocities while resisting external disturbances.

A [quadruped locomotion task](#teaching-a-quadruped-to-walk) is also included, built on the Unitree Go1 that ships with MjLab.

<br>
<br>

## Install

To install the repository, you need the uv package manager.
If you don't have it yet, you can install it by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Then, clone this repository and run the following command in your terminal:
```
uv sync
```

## Using a velocity agent

<p align="center">
  <img width="480" alt="MicrobanSimu" src="https://github.com/user-attachments/assets/fa79d712-e2ff-4452-b3ef-7ac41b87ff13" />
</p>

You can use a pre-trained agent directly in its MjLab environment (GPU required), where random velocity commands are given to the robot at regular intervals.
Linear velocity commands are represented by a blue arrow, while angular velocity commands are represented by a green vertical one.

```
uv run play Mjlab-Velocity-Microban --checkpoint-file src/mjlab_microban/agents/velocity.pt
```

To push the robot while playing, double-click on the trunk in the simulation window, then hold the left-ctrl key and right-click and drag to apply a force.

## Transferring to the real robot

The transfer on the real robot is always a challenge due to the sim-to-real gap. However, the policies trained in this repository have been successfully transferred to the real Microban robot. It is possible due to a combination of domain randomization and a well-tuned modelisation of the actuators (delays, friction, voltage drop, current clipping, etc.). This modelisation is done using the [BAM](https://github.com/Rhoban/bam) library.

Here is a video of the trained agent being transferred to the real robot: [https://youtu.be/1pnFrT_jfXQ](https://youtu.be/1pnFrT_jfXQ)

<p align="center">
  <img width="70%" alt="image" src="https://github.com/user-attachments/assets/dd91b082-faf0-4c73-a216-fe9b633f51b3" />
</p>

## Training your own agent

You can modify the environment configuration at `src/mjlab_microban/tasks/microban_velocity_env_cfg.py`.

To test the environment before training, play with a zero or random agent:

```
uv run play Mjlab-Velocity-Microban --agent zero
uv run play Mjlab-Velocity-Microban --agent random
```

Start the training with:

```
uv run train Mjlab-Velocity-Microban --env.scene.num-envs 4096
```

Once training is complete, play back a checkpoint with:

```
uv run play Mjlab-Velocity-Microban --checkpoint-file [path to your checkpoint]
```

Where `[path to your checkpoint]` is typically located at `logs/rsl_rl/mjlab_microban_velocity/[date]/model_[number].pt`.

You can also play back the last checkpoint in wandb with:

```
uv run play Mjlab-Velocity-Microban --wandb-run-path [path to your wandb run]
```

Where `[path to your wandb run]` is available in the Overview tab of your wandb run.

## Teaching a quadruped to walk

Alongside the Microban biped, this repository contains a quadruped locomotion task built on the [Unitree Go1](https://github.com/mujocolab/mjlab) that ships inside MjLab, so no extra assets are needed. The policy starts from random weights and is rewarded only for tracking a commanded body twist — there is no reference gait and no motion capture. The trot emerges from the velocity reward together with the foot clearance, swing height and slip shaping terms.

Three tasks are registered:

| Task | Terrain | Use it for |
| --- | --- | --- |
| `Mjlab-Velocity-Quadruped` | flat | Learning to walk from scratch. **Start here.** |
| `Mjlab-Velocity-Quadruped-Rough` | generated rough terrain | Fine-tuning a flat-ground policy on uneven ground. |
| `Mjlab-Velocity-Quadruped-Deploy` | flat | Retraining under hardware-realistic sensing and actuation. |

No pre-trained quadruped checkpoint is shipped with the repository — the point of the task is to train one.

### Check the environment before you train

Training needs a GPU, but the configuration itself can be checked on CPU in a few seconds. This catches the class of mistake that actually happens in a locomotion config: a regex that matches no joint, a renamed foot site, a reward wired to a sensor that is not in the scene.

```
uv run python src/mjlab_microban/scripts/check_quadruped_env.py
```

It compiles the robot, verifies every name the task refers to, holds the default stance under gravity to confirm the robot supports its own weight, and builds all four config variants. Add `--rollout 20` to also construct the environment and step it — that path runs on CPU too, just slower.

### Train

```
uv run train Mjlab-Velocity-Quadruped --env.scene.num-envs 4096
```

The commanded velocity range widens in three stages as training progresses (see `WALK_VELOCITY_STAGES` in `src/mjlab_microban/tasks/quadruped_velocity_env_cfg.py`). A policy that has never stood up cannot track 2 m/s, so the first stage asks for little more than a slow walk and the range opens once a gait exists.

What to watch in the logs:

- `Episode_Reward/track_linear_velocity` is the one that matters. It should climb steadily; a plateau near zero while the episode length stays short means the robot is falling rather than walking.
- `Episode_Length` rising towards the 20 s episode limit means the robot has stopped falling over.
- `Curriculum/command_vel/lin_vel_x_max` confirms the command curriculum has advanced.

Then play the result back:

```
uv run play Mjlab-Velocity-Quadruped --checkpoint-file logs/rsl_rl/mjlab_quadruped_velocity/[date]/model_[number].pt
```

### If the gait looks wrong

The reward weights are inherited from MjLab's own Go1 configuration, which is known to produce a walking policy on this robot — so treat them as the baseline to beat rather than the first thing to change. When the gait does go wrong, these are the knobs that matter, in the order worth trying:

- **Shuffling, feet never leaving the ground.** Raise `air_time` (weight `0.0` by default for this robot) to reward longer swing phases, or make `foot_clearance` more negative.
- **Falling over constantly, reward flat.** The first curriculum stage is asking too much; narrow the stage-0 ranges in `WALK_VELOCITY_STAGES`.
- **Walking but twitchy.** Make `action_rate_l2` more negative.

### Training on a cloud GPU

mjlab needs an NVIDIA GPU, so training does not run on a Mac. The full 10,000-iteration flat run is a single-GPU job of roughly 3–6 hours, which is a few dollars on an RTX 4090 — pick a provider on convenience, not price. [RunPod](https://www.runpod.io/product/cloud-gpus)'s Community Cloud (~$0.34/hr for a 4090, billed by the second) is a reasonable default.

On a fresh box:

```
curl -fsSL https://raw.githubusercontent.com/portalfire/mjlab_microban/claude/quadruped-walk-sim-5z0ff3/src/mjlab_microban/scripts/runpod_bootstrap.sh | bash
```

The script installs uv, clones into `/workspace` (the volume that survives a pod restart), syncs dependencies, validates the environment on the GPU, and starts training. It is safe to re-run — every step checks before it acts. Set `SETUP_ONLY=1` to stop after validation, or `NUM_ENVS`, `MAX_ITER`, `LOGGER` and `TASK` to change the run.

Four things it handles that will otherwise cost you an afternoon:

- **`CUDA_VISIBLE_DEVICES`.** mjlab's train script falls back to `device="cpu"` when this is empty, silently — a three-hour run becomes a multi-day one with no error. The script pins it from `nvidia-smi`.
- **The `bam` dependency is declared over SSH.** A fresh box has no GitHub SSH key, so `uv sync` fails. `rhoban/bam` is public, so the script rewrites the transport to HTTPS via git's `insteadOf`, and falls back to editing `pyproject.toml` if that is not enough.
- **The `bam` package-name fix is on a branch, not `main`.** The script merges `fix/bam-dep-name-and-protobuf-onnx` unless the checkout already contains it.
- **Logging defaults to W&B.** The script uses TensorBoard unless you set `LOGGER=wandb` (and `WANDB_API_KEY`).

Checkpoints land in `logs/rsl_rl/mjlab_quadruped_velocity/` inside `/workspace`. Pull one down and play it back locally — `play` builds the same environment, which also runs on CPU.

### Rough terrain and hardware

Once a flat policy walks, `Mjlab-Velocity-Quadruped-Rough` adds generated terrain with MjLab's terrain-level curriculum, which promotes environments to harder tiles as they succeed.

`Mjlab-Velocity-Quadruped-Deploy` is the configuration to train if the policy is meant to leave simulation. It differs from the flat task in four ways:

- The base **linear velocity is removed from the actor** observation, because a real quadruped cannot measure it reliably. The critic keeps it — it only runs during training (asymmetric actor-critic).
- The IMU channels get up to 15 ms of latency, on top of the noise the base task already applies.
- The actuators get 5–15 ms of command delay.
- Joint armature, joint friction and the trunk's inertial parameters are randomized per environment.

Expect lower reward than the flat task. That is the cost of robustness, not a bug.

## Exporting a policy to ONNX

A ONNX is generated during training with the latest checkpoint, but if you want to export a specific checkpoint, you can do so with the following command:

```
uv run python src/mjlab_microban/scripts/export_onnx.py --checkpoint [path to your checkpoint]
```

## License

Copyright (c) 2026 Marc Duclusaud

This software is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
