#!/usr/bin/env bash
# Pull a Docker image and convert it to an Apptainer SIF file.
#
# Usage:
#   bash pull_image.sh [docker_uri] [sif_path]
#
# Defaults:
#   docker_uri = docker-daemon://gcbench-desktop:latest
#   sif_path   = gcbench-desktop.sif
#                (or $APPTAINER_ENV_IMAGE if set)

set -euo pipefail

DOCKER_URI="${1:-docker-daemon://gcbench-desktop:latest}"
SIF_PATH="${2:-${APPTAINER_ENV_IMAGE:-gcbench-desktop.sif}}"

APPTAINER_BIN="${APPTAINER_CLI:-}"
if [[ -z "$APPTAINER_BIN" ]]; then
    APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
fi
if [[ -z "$APPTAINER_BIN" || ! -x "$APPTAINER_BIN" ]]; then
    echo "apptainer/singularity not executable; set APPTAINER_CLI to the cluster-provided executable path" >&2
    exit 1
fi

SIF_DIR="$(dirname -- "$SIF_PATH")"
mkdir -p "$SIF_DIR"

: "${APPTAINER_CACHEDIR:=${SIF_DIR}/.apptainer_cache}"
: "${APPTAINER_TMPDIR:=${SIF_DIR}/.apptainer_tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR APPTAINER_TMPDIR

echo "Pulling $DOCKER_URI -> $SIF_PATH"
echo "  cache: $APPTAINER_CACHEDIR"
echo "  tmp:   $APPTAINER_TMPDIR"

# --force lets us re-pull over an existing file when refreshing the image.
"$APPTAINER_BIN" pull --force "$SIF_PATH" "$DOCKER_URI"

echo "Done: $SIF_PATH ($(du -h "$SIF_PATH" | cut -f1))"
