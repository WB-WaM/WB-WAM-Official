#!/usr/bin/env bash
# Shared WB-WAM environment pins.
#
# Source this file from setup scripts instead of repeating version strings.

export WBWAM_PYTHON_VERSION="${WBWAM_PYTHON_VERSION:-3.10}"
export WBWAM_NUMPY_VERSION="${WBWAM_NUMPY_VERSION:-1.26.4}"
# Wuji retargeting requires Pinocchio 3.8, whose wheels require NumPy 2.
export WBWAM_TELEOP_NUMPY_VERSION="${WBWAM_TELEOP_NUMPY_VERSION:-2.2.6}"
export WBWAM_OPENCV_VERSION="${WBWAM_OPENCV_VERSION:-4.11.0.86}"
