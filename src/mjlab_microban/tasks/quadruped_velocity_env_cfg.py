# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Quadruped velocity environment.

Teaching a quadruped to walk from scratch: the policy starts from random
weights and is rewarded for tracking a commanded body twist (forward, lateral
and yaw velocity). No reference gait, no motion capture — the trot emerges from
the velocity reward plus the foot clearance / slip / swing-height shaping terms.

Three configurations are built from one factory:

``flat``
    The starting point. Flat ground, a command curriculum that widens as
    training progresses, and no sensor degradation. Use this first: it is the
    fastest way to confirm the whole loop works and to get a policy that walks.

``rough``
    Same task on generated rough terrain, with mjlab's terrain-level curriculum
    so environments are promoted to harder tiles as they succeed. Train flat
    first, then fine-tune here.

``deploy``
    Flat ground plus the modelling needed for a policy that could run on real
    hardware: the base linear velocity is removed from the actor (a real robot
    cannot measure it reliably), the IMU channels get latency on top of the
    noise they already carry, the actuators get command delay, and the
    inertial/joint parameters are randomized. Expect lower reward than ``flat`` — that is the price of
    robustness, not a bug.

The robot itself is defined in ``mjlab_microban.robot.quadruped_constants``.
"""

from copy import deepcopy
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.go1.env_cfgs import (
    unitree_go1_flat_env_cfg,
    unitree_go1_rough_env_cfg,
)

from mjlab_microban.robot.quadruped_constants import (
    ACTION_SCALE,
    TRUNK_BODY,
    get_quadruped_robot_cfg,
)

TerrainType = Literal["flat", "rough"]

# --------------------------------------------------------------------------- #
# Command curriculum
#
# Steps are counted in environment steps: one PPO iteration is
# `num_steps_per_env` (24) steps, so `1500 * 24` is iteration 1500.
#
# A policy that has never stood up cannot track 2 m/s, and sampling commands it
# cannot possibly follow only adds noise to the advantage estimates. So the
# first stage asks for little more than a slow walk, and the range opens up
# once the gait exists. The final stage matches the range mjlab's own Go1
# config trains at.
# --------------------------------------------------------------------------- #

WALK_VELOCITY_STAGES = [
    {
        "step": 0,
        "lin_vel_x": (-0.5, 0.8),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.6, 0.6),
    },
    {
        "step": 1500 * 24,
        "lin_vel_x": (-1.0, 1.5),
        "lin_vel_y": (-0.7, 0.7),
        "ang_vel_z": (-0.7, 0.7),
    },
    {
        "step": 4000 * 24,
        "lin_vel_x": (-1.5, 2.5),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-0.7, 0.7),
    },
]

# Actuator command latency for the deploy config, in physics timesteps. The
# physics runs at 200 Hz (5 ms), so this is 5-15 ms between the policy emitting
# a joint target and the motor acting on it.
DEPLOY_ACTUATOR_DELAY = (1, 3)

# Observation latency for the deploy config, in physics timesteps: up to 15 ms
# of IMU pipeline lag, resampled every 64 steps so a policy cannot learn the
# exact delay of any given episode.
DEPLOY_OBS_DELAY = (0, 3)
DEPLOY_OBS_DELAY_UPDATE_PERIOD = 64


def make_quadruped_velocity_env_cfg(
    play: bool = False,
    terrain: TerrainType = "flat",
    deploy: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the quadruped velocity environment configuration.

    Args:
        play: Build the playback variant — no observation corruption, no
            pushes, effectively unbounded episodes.
        terrain: ``"flat"`` for plane ground, ``"rough"`` for generated terrain
            with a terrain-level curriculum.
        deploy: Add the sim-to-real modelling described in the module docstring.
    """
    if terrain == "flat":
        cfg = unitree_go1_flat_env_cfg(play=play)
    else:
        cfg = unitree_go1_rough_env_cfg(play=play)

    cfg.scene.entities = {
        "robot": get_quadruped_robot_cfg(
            command_delay_steps=DEPLOY_ACTUATOR_DELAY if deploy else None
        )
    }

    # Take the action scale from our own robot module rather than leaving the
    # one the upstream Go1 config baked in, so swapping the robot there is
    # enough to move this task to a different quadruped.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ACTION_SCALE

    # ----------------------------- Curriculum ------------------------------ #
    # Play mode clears the curriculum upstream so the command ranges stay at
    # whatever the play config asks for; don't put it back.
    if not play:
        cfg.curriculum["command_vel"] = CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": WALK_VELOCITY_STAGES,
            },
        )

    if not deploy:
        return cfg

    # ---------------------------- Sim-to-real ------------------------------ #

    # A real quadruped has no reliable estimate of its own base linear
    # velocity, so the actor must walk without it. The critic keeps it: it only
    # runs during training, and the extra state makes value estimation easier
    # (asymmetric actor-critic).
    #
    # The critic's term dict is built from the actor's upstream, so the two
    # groups share ObservationTermCfg objects. Deleting a key from one dict is
    # safe, but any term we modify has to be copied first or the critic's view
    # of the world degrades along with the actor's.
    del cfg.observations["actor"].terms["base_lin_vel"]

    # The IMU pipeline on real hardware is not instantaneous. The upstream
    # config already corrupts these two channels with noise; what it does not
    # model is latency, so add it here, resampled periodically so the policy
    # cannot settle on one fixed delay. The critic keeps its prompt copy.
    for term_name in ("base_ang_vel", "projected_gravity"):
        term = deepcopy(cfg.observations["actor"].terms[term_name])
        term.delay_min_lag, term.delay_max_lag = DEPLOY_OBS_DELAY
        term.delay_update_period = DEPLOY_OBS_DELAY_UPDATE_PERIOD
        cfg.observations["actor"].terms[term_name] = term

    # Actuator response varies unit to unit and with temperature.
    cfg.events["joint_armature"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.8, 1.2),
        },
    )
    cfg.events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.8, 1.2),
        },
    )

    # Trunk payload: mass and inertia scale together by exp(2 * alpha), so this
    # is roughly a +/- 10% density change that stays physically consistent.
    cfg.events["trunk_inertia"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(TRUNK_BODY,)),
            "alpha_range": (-0.05, 0.05),
        },
    )

    return cfg


def make_quadruped_velocity_rl_cfg(
    experiment_name: str = "mjlab_quadruped_velocity",
    max_iterations: int = 10_000,
) -> RslRlOnPolicyRunnerCfg:
    """PPO configuration for the quadruped velocity task.

    These are the hyperparameters mjlab ships for the Go1; they are known to
    produce a walking policy on this robot, so treat them as the baseline to
    beat rather than the first thing to tune.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        wandb_project=experiment_name,
        experiment_name=experiment_name,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )
