#pragma once

#include <array>
#include <cstddef>

inline constexpr std::size_t kWujiFingerCount = 5;
inline constexpr std::size_t kWujiJointsPerFinger = 4;
inline constexpr std::size_t kWujiHandDoF = kWujiFingerCount * kWujiJointsPerFinger;

using WujiHandJointArray = std::array<double, kWujiHandDoF>;

inline constexpr WujiHandJointArray kDefaultWujiHandPose = {
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
};

// Replay command limits in the SDK's flat [finger][joint] order. The upper
// and lower bounds come from the vendored Wuji hand URDF and match
// bridge.sonic.action_schema.apply_wuji_qpos_limits().
inline constexpr WujiHandJointArray kWujiReplayCommandLowerLimits = {
     0.0475, -0.1387, -0.4642, -0.4699,
    -0.1585, -0.3700, -0.4777, -0.4683,
    -0.1644, -0.3700, -0.4739, -0.4684,
    -0.1554, -0.3700, -0.4765, -0.4777,
    -0.1626, -0.3700, -0.4768, -0.4683,
};

inline constexpr WujiHandJointArray kWujiReplayCommandUpperLimits = {
    1.6033, 0.9324, 1.5623, 1.5568,
    1.5604, 0.3700, 1.5485, 1.5753,
    1.5516, 0.3700, 1.5512, 1.5745,
    1.5585, 0.3700, 1.5487, 1.5634,
    1.5585, 0.3700, 1.5490, 1.5735,
};

inline constexpr double kWujiReplayLimitTolerance = 1e-5;
