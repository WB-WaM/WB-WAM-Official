#!/usr/bin/env bash

LAUNCHER_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LAUNCHER_CONFIG_FILE="${DEFAULT_LAUNCHER_CONFIG_FILE:-$LAUNCHER_HELPER_DIR/collector_pc.env}"
LAUNCHER_CONFIG_FILE="${LAUNCHER_CONFIG_FILE:-${COLLECTOR_CONFIG_FILE:-$DEFAULT_LAUNCHER_CONFIG_FILE}}"

if [[ -f "$LAUNCHER_CONFIG_FILE" ]]; then
    echo "Loading launcher config: $LAUNCHER_CONFIG_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$LAUNCHER_CONFIG_FILE"
    set +a
fi

# Keep the legacy variable in sync so older wrappers can still print a useful path.
COLLECTOR_CONFIG_FILE="$LAUNCHER_CONFIG_FILE"

require_wuji_glove_addresses() {
    local name value
    local left="${WUJI_GLOVE_LEFT_ADDRESS:-}"
    local right="${WUJI_GLOVE_RIGHT_ADDRESS:-}"
    while (($#)); do
        case "$1" in
            --left-address|--wuji-glove-left-address)
                left="${2:-}"
                if (($# > 1)); then shift; fi
                ;;
            --right-address|--wuji-glove-right-address)
                right="${2:-}"
                if (($# > 1)); then shift; fi
                ;;
            --left-address=*|--wuji-glove-left-address=*) left="${1#*=}" ;;
            --right-address=*|--wuji-glove-right-address=*) right="${1#*=}" ;;
        esac
        shift
    done
    for name in left right; do
        value="${!name}"
        if [[ -z "$value" || "$value" == *"<"*">"* ]]; then
            echo "Set WUJI_GLOVE_${name^^}_ADDRESS in $LAUNCHER_CONFIG_FILE or pass --${name}-address" >&2
            return 1
        fi
    done
}
