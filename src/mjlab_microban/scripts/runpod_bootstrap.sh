#!/usr/bin/env bash
# Copyright 2026 Marc Duclusaud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Bring a bare CUDA box (RunPod and friends) to a running quadruped training
# job. Safe to re-run: every step checks before it acts.
#
# On the box:
#
#   curl -fsSL https://raw.githubusercontent.com/portalfire/mjlab_microban/claude/quadruped-walk-sim-5z0ff3/src/mjlab_microban/scripts/runpod_bootstrap.sh | bash
#
# or, from an existing clone:
#
#   ./src/mjlab_microban/scripts/runpod_bootstrap.sh
#
# Tunables, all via environment variables:
#
#   WORKDIR=/workspace     Where to clone. On RunPod, /workspace is the volume
#                          that survives a pod restart -- do not use /root.
#   TASK=Mjlab-Velocity-Quadruped
#   NUM_ENVS=4096
#   MAX_ITER=              Empty means the task default (10000).
#   LOGGER=tensorboard     Or `wandb`, which needs WANDB_API_KEY set.
#   SETUP_ONLY=0           1 to stop after validation, printing the train command.
#
# Example:
#   SETUP_ONLY=1 ./runpod_bootstrap.sh
#   NUM_ENVS=1024 LOGGER=wandb ./runpod_bootstrap.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/portalfire/mjlab_microban.git}"
BRANCH="${BRANCH:-claude/quadruped-walk-sim-5z0ff3}"
# The bam dependency-name fix lives on its own branch and is not on main. Until
# it is merged, the build needs it or `uv sync` fails on metadata validation.
DEP_FIX_BRANCH="${DEP_FIX_BRANCH:-fix/bam-dep-name-and-protobuf-onnx}"
WORKDIR="${WORKDIR:-/workspace}"
TASK="${TASK:-Mjlab-Velocity-Quadruped}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-}"
LOGGER="${LOGGER:-tensorboard}"
SETUP_ONLY="${SETUP_ONLY:-0}"

REPO_DIR="$WORKDIR/mjlab_microban"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
log "1/7  GPU preflight"
# --------------------------------------------------------------------------- #
command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi: this box has no NVIDIA GPU. mjlab cannot train here."
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
    || die "nvidia-smi failed: the GPU is not usable from inside this container."

# mjlab's train script reads CUDA_VISIBLE_DEVICES and falls back to device="cpu"
# when it is empty -- silently, with no error. That turns a three-hour run into
# a multi-day one, so pin it here rather than trust the container's defaults.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    export CUDA_VISIBLE_DEVICES
    warn "CUDA_VISIBLE_DEVICES was empty; set to '$CUDA_VISIBLE_DEVICES'."
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# --------------------------------------------------------------------------- #
log "2/7  System packages"
# --------------------------------------------------------------------------- #
missing=()
for pkg in git curl; do
    command -v "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "installing: ${missing[*]}"
    apt-get update -qq && apt-get install -y -qq "${missing[@]}"
else
    echo "git and curl present"
fi

# --------------------------------------------------------------------------- #
log "3/7  uv"
# --------------------------------------------------------------------------- #
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv install failed; \$HOME/.local/bin may not be on PATH"
uv --version

# --------------------------------------------------------------------------- #
log "4/7  Repository"
# --------------------------------------------------------------------------- #
# The bam dependency is declared as ssh://git@github.com/rhoban/bam. A fresh box
# has no GitHub SSH key, so that clone fails. rhoban/bam is public, so rewriting
# the transport to HTTPS resolves it with no credentials. Harmless if a key does
# exist.
git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
git config --global --get user.email >/dev/null 2>&1 || git config --global user.email "bootstrap@localhost"
git config --global --get user.name  >/dev/null 2>&1 || git config --global user.name  "runpod bootstrap"

mkdir -p "$WORKDIR"
if [ -d "$REPO_DIR/.git" ]; then
    echo "reusing $REPO_DIR"
    git -C "$REPO_DIR" fetch --all --prune
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || warn "could not fast-forward $BRANCH; continuing with local state"

# Merge the dependency fix unless this branch already contains it.
if git rev-parse --verify --quiet "origin/$DEP_FIX_BRANCH" >/dev/null; then
    if git merge-base --is-ancestor "origin/$DEP_FIX_BRANCH" HEAD; then
        echo "dependency fix already present"
    else
        echo "merging origin/$DEP_FIX_BRANCH (bam dist name + onnx/protobuf overrides)"
        git merge --no-edit "origin/$DEP_FIX_BRANCH" \
            || die "merge conflict with $DEP_FIX_BRANCH -- resolve by hand and re-run"
    fi
else
    warn "branch origin/$DEP_FIX_BRANCH not found; if uv sync fails on the bam package name, that is why"
fi
echo "HEAD: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# --------------------------------------------------------------------------- #
log "5/7  Dependencies"
# --------------------------------------------------------------------------- #
if ! uv sync; then
    warn "uv sync failed; retrying with the bam source rewritten to HTTPS in pyproject.toml"
    # Fallback for a uv build that does not honour git's insteadOf rewrite.
    sed -i 's|ssh://git@github.com/rhoban/bam|https://github.com/rhoban/bam|' pyproject.toml
    uv sync || die "uv sync failed. Check the error above; the usual causes are the bam
package name (needs the $DEP_FIX_BRANCH branch) or no network access to GitHub."
fi

# --------------------------------------------------------------------------- #
log "6/7  Validation on the real device"
# --------------------------------------------------------------------------- #
# Same checker used on CPU, pointed at the GPU: compiles the robot, checks every
# name and regex the task refers to, holds the stance under gravity, builds all
# config variants, then builds and steps the env on CUDA. Catches a broken box
# in ~a minute instead of at iteration 1.
uv run python src/mjlab_microban/scripts/check_quadruped_env.py --rollout 20 --device cuda:0 \
    || die "validation failed -- do not start a training run until this passes"

# --------------------------------------------------------------------------- #
log "7/7  Training"
# --------------------------------------------------------------------------- #
train_cmd=(uv run train "$TASK" --env.scene.num-envs "$NUM_ENVS" --agent.logger "$LOGGER")
[ -n "$MAX_ITER" ] && train_cmd+=(--agent.max-iterations "$MAX_ITER")

if [ "$LOGGER" = "wandb" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    warn "LOGGER=wandb but WANDB_API_KEY is unset; the run will stop to ask you to log in."
fi

if [ "$SETUP_ONLY" = "1" ]; then
    log "Setup complete. To train:"
    printf '  cd %s && %s\n' "$REPO_DIR" "${train_cmd[*]}"
    exit 0
fi

log "Starting training: ${train_cmd[*]}"
echo "Checkpoints land in $REPO_DIR/logs/rsl_rl/mjlab_quadruped_velocity/"
echo "Detach-safe alternative: nohup ${train_cmd[*]} > train.log 2>&1 &"
exec "${train_cmd[@]}"
