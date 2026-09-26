#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COLLECTOR_DIR/../.." && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage:
  collector/sonic/scripts/save_manus_calibration.sh left [source.mcal]
  collector/sonic/scripts/save_manus_calibration.sh right [source.mcal]

The MANUS SDK Client usually saves the latest calibration as:
  ~/Documents/manus-calibrations/Calibration.mcal
This script copies that file into the project-local calibration directory.
EOF
}

side="${1:-}"
source_file="${2:-${MANUS_SOURCE_CALIBRATION_FILE:-$HOME/Documents/manus-calibrations/Calibration.mcal}}"
case "$side" in
    left|right) ;;
    *) usage; exit 2 ;;
esac

calibration_dir="${MANUS_PROJECT_CALIBRATION_DIR:-$REPO_ROOT/third_party/MANUS_SDK/calibrations}"
dest_file="$calibration_dir/Calibration_${side}.mcal"

if [[ ! -f "$source_file" ]]; then
    echo "Calibration source file not found: $source_file" >&2
    exit 1
fi

install -d "$calibration_dir"
cp -f "$source_file" "$dest_file"
cp -f "$source_file" "$calibration_dir/Calibration.mcal"

echo "Saved MANUS $side calibration"
echo "  source: $source_file"
echo "  dest:   $dest_file"
echo "  latest: $calibration_dir/Calibration.mcal"
