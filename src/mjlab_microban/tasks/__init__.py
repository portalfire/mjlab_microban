# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_microban.tasks.microban_velocity_env_cfg import (
    make_microban_velocity_env_cfg,
    MicrobanVelocityRlCfg,
)
from mjlab_microban.tasks.quadruped_velocity_env_cfg import (
    make_quadruped_velocity_env_cfg,
    make_quadruped_velocity_rl_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Microban",
    env_cfg=make_microban_velocity_env_cfg(),
    play_env_cfg=make_microban_velocity_env_cfg(play=True),
    rl_cfg=MicrobanVelocityRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Quadruped locomotion. Start with the flat task: it is the one that takes a
# randomly initialized policy to a walking gait. `Rough` fine-tunes that policy
# on generated terrain, and `Deploy` retrains it under the sensing and actuation
# limits of real hardware.

register_mjlab_task(
    task_id="Mjlab-Velocity-Quadruped",
    env_cfg=make_quadruped_velocity_env_cfg(terrain="flat"),
    play_env_cfg=make_quadruped_velocity_env_cfg(terrain="flat", play=True),
    rl_cfg=make_quadruped_velocity_rl_cfg("mjlab_quadruped_velocity"),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Quadruped-Rough",
    env_cfg=make_quadruped_velocity_env_cfg(terrain="rough"),
    play_env_cfg=make_quadruped_velocity_env_cfg(terrain="rough", play=True),
    rl_cfg=make_quadruped_velocity_rl_cfg("mjlab_quadruped_velocity_rough"),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Quadruped-Deploy",
    env_cfg=make_quadruped_velocity_env_cfg(terrain="flat", deploy=True),
    play_env_cfg=make_quadruped_velocity_env_cfg(
        terrain="flat", deploy=True, play=True
    ),
    rl_cfg=make_quadruped_velocity_rl_cfg("mjlab_quadruped_velocity_deploy"),
    runner_cls=VelocityOnPolicyRunner,
)
