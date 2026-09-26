#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COLLECTION_ENV="$SCRIPT_DIR/hgpt_collection.env"

# shellcheck disable=SC1090
source "$COLLECTION_ENV"

has_option() {
    local requested="$1"
    shift
    local raw option
    for raw in "$@"; do
        option="${raw%%=*}"
        option="${option//_/-}"
        if [[ "$option" == "$requested" ]]; then
            return 0
        fi
    done
    return 1
}

DEFAULT_ARGS=()
if ! has_option --camera-endpoint "$@"; then
    DEFAULT_ARGS+=(--camera-endpoint "$HGPT_CAMERA_ENDPOINT")
fi
if ! has_option --pose-endpoint "$@"; then
    DEFAULT_ARGS+=(--pose-endpoint "$HGPT_POSE_ENDPOINT")
fi
if ! has_option --state-action-endpoint "$@"; then
    DEFAULT_ARGS+=(--state-action-endpoint "$HGPT_STATE_ACTION_ENDPOINT")
fi
if ! has_option --hand-status-endpoint "$@"; then
    DEFAULT_ARGS+=(--hand-status-endpoint "$HGPT_HAND_STATUS_ENDPOINT")
fi

PYTHON_BIN="${COLLECTOR_PYTHON:-$REPO_ROOT/.venv_teleop/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${COLLECTOR_PYTHON_FALLBACK:-python3}"
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m collector.humanoid_gpt.run_collector \
    "${DEFAULT_ARGS[@]}" "$@"
