# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Sanity-check the quadruped environment without a GPU.

Training needs a GPU, but almost everything that goes wrong in a locomotion
config goes wrong before the first gradient step: a regex that matches no
joint, a site name that moved, a reward wired to a sensor that isn't in the
scene. This script catches that class of mistake on CPU in seconds.

Usage:
    # Model and config checks only (fast).
    uv run python src/mjlab_microban/scripts/check_quadruped_env.py

    # Also build the environment and step it on CPU. Slower (warp compiles
    # kernels on first run) but exercises the real simulation loop.
    uv run python src/mjlab_microban/scripts/check_quadruped_env.py --rollout 20
"""

import argparse
import re

import mujoco
import numpy as np
from mjlab.entity.entity import Entity

from mjlab_microban.robot.quadruped_constants import (
    ABDUCTION_AND_HIP_JOINTS,
    FOOT_GEOMS,
    FOOT_SITES,
    KNEE_JOINTS,
    NOMINAL_HEIGHT,
    THIGH_GEOMS,
    TRUNK_BODY,
    get_quadruped_robot_cfg,
)
from mjlab_microban.tasks.quadruped_velocity_env_cfg import (
    make_quadruped_velocity_env_cfg,
)

# Tolerance on the settled standing height [m]. The Go1 sags roughly 40 mm into
# its position actuators holding its own weight, so this is deliberately loose:
# it is here to catch a robot that collapses or launches, not to measure posture.
HEIGHT_TOLERANCE = 0.08


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rollout",
        type=int,
        default=0,
        metavar="N",
        help="Build the env on CPU and step it N times (0 = skip).",
    )
    return p.parse_args()


def check_robot_model() -> None:
    """Compile the robot and confirm every name the task refers to exists."""
    print("== robot model ==")
    model = Entity(get_quadruped_robot_cfg()).spec.compile()

    def names(obj_type: mujoco.mjtObj, count: int) -> set[str]:
        found = (mujoco.mj_id2name(model, obj_type, i) for i in range(count))
        return {n for n in found if n is not None}

    bodies = names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    sites = names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    geoms = names(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    joints = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)

    missing: list[str] = []
    for label, wanted, available in (
        ("body", {TRUNK_BODY}, bodies),
        ("site", set(FOOT_SITES), sites),
        ("geom", set(FOOT_GEOMS) | set(THIGH_GEOMS), geoms),
    ):
        for name in sorted(wanted - available):
            missing.append(f"{label} '{name}'")

    # The reward config selects joints by regex; an expression that matches
    # nothing silently disables the reward it belongs to.
    for label, pattern in (
        ("abduction/hip", ABDUCTION_AND_HIP_JOINTS),
        ("knee", KNEE_JOINTS),
    ):
        matched = [j for j in joints if re.fullmatch(pattern, j)]
        if not matched:
            missing.append(f"joint regex {label} ({pattern!r}) matched nothing")
        else:
            print(f"  {label:14s} regex matches {len(matched)} joints")

    print(f"  bodies={model.nbody} joints={model.njnt} actuators={model.nu}")
    print(f"  total mass {model.body_mass.sum():.2f} kg")

    if missing:
        raise SystemExit("MISSING FROM MODEL:\n  " + "\n  ".join(missing))
    print("  all referenced names present")


def check_standing() -> None:
    """Hold the default pose under gravity and see where the trunk settles.

    Purely MuJoCo on CPU: no warp, no RL. If the robot cannot hold itself up
    with its own position actuators at the default targets, no reward function
    is going to save the policy.

    The entity spec is the robot alone, so a ground plane has to be added here
    for it to have anything to stand on.
    """
    print("== standing ==")
    spec = Entity(get_quadruped_robot_cfg()).spec
    spec.worldbody.add_geom(
        name="check_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, 0.0],
    )
    model = spec.compile()

    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if key_id < 0:
        raise SystemExit("no 'init_state' keyframe on the compiled robot")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Command each position actuator to hold the joint angle it starts at.
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        data.ctrl[i] = data.qpos[model.jnt_qposadr[joint_id]]

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY)
    start_height = float(data.xpos[trunk_id, 2]) if trunk_id >= 0 else float("nan")

    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    height = float(data.xpos[trunk_id, 2])
    print(f"  trunk height: {start_height:.3f} m -> {height:.3f} m after 2 s")
    print(f"  nominal: {NOMINAL_HEIGHT:.3f} m")

    if not np.isfinite(height):
        raise SystemExit("simulation diverged (non-finite trunk height)")
    if abs(height - NOMINAL_HEIGHT) > HEIGHT_TOLERANCE:
        raise SystemExit(
            f"robot did not hold its stance: settled at {height:.3f} m, "
            f"expected {NOMINAL_HEIGHT:.3f} +/- {HEIGHT_TOLERANCE:.3f} m"
        )
    print("  robot holds its stance")


def check_configs() -> None:
    """Build every variant of the task config and summarize it."""
    variants = {
        "flat": dict(terrain="flat"),
        "rough": dict(terrain="rough"),
        "deploy": dict(terrain="flat", deploy=True),
        "flat/play": dict(terrain="flat", play=True),
    }
    for name, kwargs in variants.items():
        cfg = make_quadruped_velocity_env_cfg(**kwargs)  # type: ignore[arg-type]
        actor = sorted(cfg.observations["actor"].terms)
        print(f"== config: {name} ==")
        print(f"  terrain     {cfg.scene.terrain.terrain_type}")
        print(f"  actor obs   {', '.join(actor)}")
        print(f"  rewards     {len(cfg.rewards)}  events {len(cfg.events)}")
        print(f"  curriculum  {sorted(cfg.curriculum) or '(none)'}")
        print(f"  terminate   {sorted(cfg.terminations)}")

        if name == "deploy":
            assert "base_lin_vel" not in actor, (
                "deploy config must not feed base linear velocity to the actor"
            )
            assert "base_lin_vel" in cfg.observations["critic"].terms, (
                "critic should keep base linear velocity (asymmetric actor-critic)"
            )


def check_rollout(steps: int) -> None:
    """Instantiate the environment on CPU and step it."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    print(f"== rollout ({steps} steps on cpu) ==")
    cfg = make_quadruped_velocity_env_cfg(terrain="flat")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

    obs, _ = env.reset()
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(steps):
        obs, reward, terminated, truncated, _ = env.step(action)

    print(f"  actions     {env.action_manager.total_action_dim}")
    for group, tensor in obs.items():
        print(f"  obs[{group}]  {tuple(tensor.shape)}")
    print(f"  reward      {reward.tolist()}")
    if not torch.isfinite(reward).all():
        raise SystemExit("non-finite reward during rollout")
    env.close()
    print("  rollout OK")


def main() -> None:
    args = parse_args()
    check_robot_model()
    check_standing()
    check_configs()
    if args.rollout:
        check_rollout(args.rollout)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
