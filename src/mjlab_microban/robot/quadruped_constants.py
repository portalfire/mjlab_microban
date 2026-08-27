# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Quadruped robot configuration.

The quadruped task is built on the Unitree Go1 that ships inside mjlab's asset
zoo. Its meshes, inertias and actuator model are already validated upstream, so
a policy can be trained from scratch without any asset work here.

This module is the single place the task looks at the robot. To train a
different quadruped, return your own ``EntityCfg`` from
:func:`get_quadruped_robot_cfg` and update the name tables below; nothing in
``quadruped_velocity_env_cfg`` refers to the robot by any other means.
"""

from dataclasses import replace

from mjlab.asset_zoo.robots import GO1_ACTION_SCALE, get_go1_robot_cfg
from mjlab.entity import EntityCfg

# --------------------------------------------------------------------------- #
# Naming
#
# The Go1 names each leg by the corner it sits on: F/R (front/rear) followed by
# R/L (right/left). Every leg has three joints: `hip` (abduction, rolls the leg
# sideways), `thigh` (hip flexion) and `calf` (knee).
# --------------------------------------------------------------------------- #

TRUNK_BODY = "trunk"

LEGS = ("FR", "FL", "RR", "RL")

# One site per foot, used by the foot clearance / slip rewards and the foot
# height scanner. On the Go1 the site shares the leg's name.
FOOT_SITES = LEGS

FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)
THIGH_GEOMS = tuple(f"{leg}_thigh_collision{i}" for leg in LEGS for i in (1, 2, 3))

# Joint groups, as regexes, for per-group reward tuning.
ABDUCTION_AND_HIP_JOINTS = r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*"
KNEE_JOINTS = r".*(FR|FL|RR|RL)_calf_joint.*"

# Trunk height in the default standing keyframe [m].
NOMINAL_HEIGHT = 0.278

# Per-actuator action scale, derived upstream from effort limit / stiffness so
# that an action of 1.0 corresponds to a quarter of the actuator's torque range.
ACTION_SCALE = GO1_ACTION_SCALE


def get_quadruped_robot_cfg(
    command_delay_steps: tuple[int, int] | None = None,
) -> EntityCfg:
    """Build a fresh quadruped robot configuration.

    Returns a new instance on every call, so callers can mutate it freely.

    Args:
        command_delay_steps: If given, ``(min, max)`` actuator command latency in
            physics timesteps, sampled per step. Models the delay between the
            policy emitting a target and the motor acting on it. Leave at
            ``None`` for a clean sim-only baseline; set it when training a
            policy meant to run on hardware.
    """
    cfg = get_go1_robot_cfg()

    if command_delay_steps is not None:
        min_lag, max_lag = command_delay_steps
        articulation = cfg.articulation
        assert articulation is not None
        articulation.actuators = tuple(
            replace(actuator, delay_min_lag=min_lag, delay_max_lag=max_lag)
            for actuator in articulation.actuators
        )

    return cfg


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    viewer.launch(Entity(get_quadruped_robot_cfg()).spec.compile())
