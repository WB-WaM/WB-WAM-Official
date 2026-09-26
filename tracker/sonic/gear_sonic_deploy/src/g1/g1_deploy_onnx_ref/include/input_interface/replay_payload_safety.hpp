/**
 * @file replay_payload_safety.hpp
 * @brief Pure validation helpers for explicit SONIC replay packets.
 */

#ifndef REPLAY_PAYLOAD_SAFETY_HPP
#define REPLAY_PAYLOAD_SAFETY_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "../wuji_hand_constants.hpp"

namespace sonic_replay {

inline constexpr std::size_t kG1ReplayJointCount = 29;
inline constexpr double kReplayJointPositionHardDelta = 0.8;
inline constexpr double kReplayBodyQuaternionHardAngle = 1.2;
inline constexpr double kReplayNumericLimitTolerance = 1e-6;
// Matches config/manager_env/base_env.yaml. The decoder emits a normalized
// policy action which is converted into a PD position setpoint. That setpoint
// may legitimately lie outside the mechanical joint range to create restoring
// torque, so command validation uses this action envelope while measured
// positions continue to use the URDF/XML hard limits below.
inline constexpr double kReplayPolicyActionAbsLimit = 20.0;

inline bool ReplayIsFinite(double value) {
    static_assert(sizeof(value) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

// Replay joint_pos is in policy/IsaacLab order. These position limits come from
// gear_sonic_deploy/g1/g1_29dof.xml and are reordered with the repository's
// mujoco_to_isaaclab mapping. Velocity limits use the conservative minimum of
// the non-rev URDF, rev-1.0 URDF, and SONIC model-12 config.
inline constexpr std::array<double, kG1ReplayJointCount>
    kG1ReplayJointPositionLowerLimits = {
        -2.5307, -2.5307, -2.618, -0.5236, -2.9671, -0.52,
        -2.7576, -2.7576, -0.52, -0.087267, -0.087267, -3.0892,
        -3.0892, -0.87267, -0.87267, -1.5882, -2.2515, -0.2618,
        -0.2618, -2.618, -2.618, -1.0472, -1.0472, -1.97222,
        -1.97222, -1.61443, -1.61443, -1.61443, -1.61443,
    };

inline constexpr std::array<double, kG1ReplayJointCount>
    kG1ReplayJointPositionUpperLimits = {
        2.8798, 2.8798, 2.618, 2.9671, 0.5236, 0.52,
        2.7576, 2.7576, 0.52, 2.8798, 2.8798, 2.6704,
        2.6704, 0.5236, 0.5236, 2.2515, 1.5882, 0.2618,
        0.2618, 2.618, 2.618, 2.0944, 2.0944, 1.97222,
        1.97222, 1.61443, 1.61443, 1.61443, 1.61443,
    };

inline constexpr std::array<double, kG1ReplayJointCount>
    kG1ReplayJointVelocityLimits = {
        20.0, 20.0, 32.0, 20.0, 20.0, 30.0, 32.0, 32.0,
        30.0, 20.0, 20.0, 37.0, 37.0, 30.0, 30.0, 37.0,
        37.0, 30.0, 30.0, 37.0, 37.0, 37.0, 37.0, 37.0,
        37.0, 22.0, 22.0, 22.0, 22.0,
    };

struct ReplayMeasuredSafety {
    bool finite = true;
    bool joint_position_violation = false;
    bool joint_velocity_violation = false;
    bool fall_detected = false;
};

struct ReplayArraySpec {
    std::vector<std::size_t> shape;
    std::string dtype;
    std::size_t payload_bytes = 0;
};

class ReplayPayloadSafety {
public:
    static ReplayMeasuredSafety EvaluateMeasuredState(
        const std::array<double, kG1ReplayJointCount>& joint_pos,
        const std::array<double, kG1ReplayJointCount>& joint_vel,
        const std::array<double, 4>& base_quat) {
        ReplayMeasuredSafety result;
        for (std::size_t joint = 0; joint < kG1ReplayJointCount; ++joint) {
            const double q = joint_pos[joint];
            const double dq = joint_vel[joint];
            if (!ReplayIsFinite(q)) {
                result.finite = false;
                result.joint_position_violation = true;
            } else if (q < kG1ReplayJointPositionLowerLimits[joint] -
                               kReplayNumericLimitTolerance ||
                       q > kG1ReplayJointPositionUpperLimits[joint] +
                               kReplayNumericLimitTolerance) {
                result.joint_position_violation = true;
            }
            if (!ReplayIsFinite(dq)) {
                result.finite = false;
                result.joint_velocity_violation = true;
            } else if (std::abs(dq) > kG1ReplayJointVelocityLimits[joint] +
                                           kReplayNumericLimitTolerance) {
                result.joint_velocity_violation = true;
            }
        }

        double norm_squared = 0.0;
        for (const double component : base_quat) {
            if (!ReplayIsFinite(component)) {
                result.finite = false;
                return result;
            }
            norm_squared += component * component;
        }
        if (!ReplayIsFinite(norm_squared) ||
            norm_squared <= std::numeric_limits<double>::epsilon()) {
            result.finite = false;
            return result;
        }
        const double inverse_norm = 1.0 / std::sqrt(norm_squared);
        const double w = base_quat[0] * inverse_norm;
        const double x = base_quat[1] * inverse_norm;
        const double y = base_quat[2] * inverse_norm;
        const double z = base_quat[3] * inverse_norm;
        const double roll = std::atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y));
        const double sin_pitch = std::clamp(
            2.0 * (w * y - z * x), -1.0, 1.0);
        const double pitch = std::asin(sin_pitch);
        if (!ReplayIsFinite(roll) || !ReplayIsFinite(pitch)) {
            result.finite = false;
            return result;
        }
        constexpr double kFallTiltRadians =
            0.78539816339744830962;  // 45 degrees.
        result.fall_detected =
            std::max(std::abs(roll), std::abs(pitch)) >= kFallTiltRadians;
        return result;
    }

    static bool PolicyActionWithinEnvelope(double action) {
        return ReplayIsFinite(action) &&
               std::abs(action) <=
                   kReplayPolicyActionAbsLimit + kReplayNumericLimitTolerance;
    }

    static bool ValidateFloatArray(
        const ReplayArraySpec& spec,
        const std::vector<std::size_t>& expected_shape,
        std::string_view field_name,
        std::string& reason) {
        if (spec.shape != expected_shape) {
            reason = std::string(field_name) + " must have shape " +
                     FormatShape(expected_shape) + ", got " + FormatShape(spec.shape);
            return false;
        }
        const std::size_t element_bytes = FloatElementBytes(spec.dtype);
        if (element_bytes == 0) {
            reason = std::string(field_name) +
                     " must use dtype f32 or f64, got '" + spec.dtype + "'";
            return false;
        }
        const std::size_t expected_bytes = ElementCount(expected_shape) * element_bytes;
        if (spec.payload_bytes != expected_bytes) {
            reason = std::string(field_name) + " payload has " +
                     std::to_string(spec.payload_bytes) + " bytes, expected " +
                     std::to_string(expected_bytes);
            return false;
        }
        return true;
    }

    static bool ValidateI64Array(
        const ReplayArraySpec& spec,
        const std::vector<std::size_t>& expected_shape,
        std::string_view field_name,
        std::string& reason) {
        if (spec.shape != expected_shape || spec.dtype != "i64") {
            reason = std::string(field_name) + " must be i64" +
                     FormatShape(expected_shape);
            return false;
        }
        const std::size_t expected_bytes =
            ElementCount(expected_shape) * sizeof(std::int64_t);
        if (spec.payload_bytes != expected_bytes) {
            reason = std::string(field_name) + " payload has " +
                     std::to_string(spec.payload_bytes) + " bytes, expected " +
                     std::to_string(expected_bytes);
            return false;
        }
        return true;
    }

    static bool ValidateBoolScalar(
        const ReplayArraySpec& spec,
        std::string_view field_name,
        std::string& reason) {
        if (spec.shape != std::vector<std::size_t>{1} || spec.dtype != "bool" ||
            spec.payload_bytes != 1) {
            reason = std::string(field_name) +
                     " must be bool[1] with one payload byte";
            return false;
        }
        return true;
    }

    static bool ValidateReplayTarget(
        const ReplayArraySpec& spec,
        std::uint8_t encoded_value,
        bool expected_real,
        std::string& reason) {
        if (!ValidateBoolScalar(spec, "replay_target_real", reason)) {
            return false;
        }
        if (encoded_value > 1) {
            reason = "replay_target_real must encode false as 0 or true as 1";
            return false;
        }
        const bool payload_real = encoded_value != 0;
        if (payload_real != expected_real) {
            reason = std::string("replay target mismatch: native=") +
                     (expected_real ? "real" : "sim") + ", payload=" +
                     (payload_real ? "real" : "sim");
            return false;
        }
        return true;
    }

    static bool ValidateReplayHandFrame(
        const ReplayArraySpec& spec,
        std::int64_t hand_frame_index,
        std::int64_t source_frame,
        std::string& reason) {
        if (!ValidateI64Array(
                spec, std::vector<std::size_t>{1}, "hand_frame_index",
                reason)) {
            return false;
        }
        if (hand_frame_index != source_frame) {
            reason = "hand_frame_index must equal frame_index (hand=" +
                     std::to_string(hand_frame_index) + ", source=" +
                     std::to_string(source_frame) + ")";
            return false;
        }
        return true;

    }
    static bool ValidateV1CoreSchema(
        const ReplayArraySpec& joint_pos,
        const ReplayArraySpec& joint_vel,
        const ReplayArraySpec& body_quat_w,
        const ReplayArraySpec& frame_index,
        std::size_t expected_rows,
        std::string& reason) {
        if (expected_rows == 0) {
            reason = "v1 replay expected row count must be positive";
            return false;
        }
        const std::vector<std::size_t> joint_shape{
            expected_rows, kG1ReplayJointCount};
        if (!ValidateFloatArray(joint_pos, joint_shape, "joint_pos", reason) ||
            !ValidateFloatArray(joint_vel, joint_shape, "joint_vel", reason)) {
            return false;
        }
        if (joint_pos.dtype != joint_vel.dtype) {
            reason = "joint_pos and joint_vel dtype must match";
            return false;
        }
        if (!ValidateFloatArray(
                body_quat_w, {expected_rows, 1, 4},
                "body_quat_w", reason)) {
            return false;
        }
        return ValidateI64Array(
            frame_index, {expected_rows}, "frame_index", reason);
    }

    static bool ValidateJointSnapshot(
        const std::vector<std::vector<double>>& joint_pos,
        const std::vector<std::vector<double>>& joint_vel,
        std::size_t expected_rows,
        std::string& reason) {
        if (expected_rows == 0 || joint_pos.size() != expected_rows ||
            joint_vel.size() != expected_rows) {
            reason = "joint_pos/joint_vel must contain exactly " +
                     std::to_string(expected_rows) + " replay rows";
            return false;
        }
        for (std::size_t row = 0; row < expected_rows; ++row) {
            if (joint_pos[row].size() != kG1ReplayJointCount ||
                joint_vel[row].size() != kG1ReplayJointCount) {
                reason = "joint_pos/joint_vel replay rows must contain exactly 29 joints";
                return false;
            }
            for (std::size_t joint = 0; joint < kG1ReplayJointCount; ++joint) {
                const double q = joint_pos[row][joint];
                const double dq = joint_vel[row][joint];
                if (!ReplayIsFinite(q) ||
                    q < kG1ReplayJointPositionLowerLimits[joint] -
                            kReplayNumericLimitTolerance ||
                    q > kG1ReplayJointPositionUpperLimits[joint] +
                            kReplayNumericLimitTolerance) {
                    reason = "joint_pos out of replay bound at row " +
                             std::to_string(row) + ", joint " +
                             std::to_string(joint);
                    return false;
                }
                if (!ReplayIsFinite(dq) ||
                    std::abs(dq) > kG1ReplayJointVelocityLimits[joint] +
                                       kReplayNumericLimitTolerance) {
                    reason = "joint_vel out of replay bound at row " +
                             std::to_string(row) + ", joint " +
                             std::to_string(joint);
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Validate raw, globally ordered replay rows without modifying them.
     *
     * The last adjacent pair is the previous snapshot last overlapping row
     * followed by this packet newly appended tail. Consequently this checks
     * the cross-packet tail transition without ever comparing the previous
     * packet future tail against the next packet current head.
     */
    static bool ValidateRawSnapshotTransitions(
        const std::vector<std::int64_t>& frame_indices,
        const std::vector<std::vector<double>>& joint_pos,
        const std::vector<std::vector<std::array<double, 4>>>& body_quat,
        std::size_t expected_rows,
        std::string& reason) {
        if (expected_rows == 0 || frame_indices.size() != expected_rows ||
            joint_pos.size() != frame_indices.size() ||
            body_quat.size() != frame_indices.size()) {
            reason = "replay transition validation requires exactly " +
                     std::to_string(expected_rows) + " aligned rows";
            return false;
        }

        for (std::size_t row = 1; row < frame_indices.size(); ++row) {
            if (frame_indices[row] != frame_indices[row - 1] + 1) {
                reason = "replay transition rows are not globally contiguous";
                return false;
            }
            if (joint_pos[row].size() != kG1ReplayJointCount ||
                joint_pos[row - 1].size() != kG1ReplayJointCount) {
                reason = "replay transition joint rows must contain exactly 29 joints";
                return false;
            }
            for (std::size_t joint = 0; joint < kG1ReplayJointCount; ++joint) {
                const double delta =
                    std::abs(joint_pos[row][joint] - joint_pos[row - 1][joint]);
                if (!ReplayIsFinite(delta) ||
                    delta > kReplayJointPositionHardDelta +
                                kReplayNumericLimitTolerance) {
                    reason = "joint_pos hard jump between source frames " +
                             std::to_string(frame_indices[row - 1]) + " and " +
                             std::to_string(frame_indices[row]) + ", joint " +
                             std::to_string(joint);
                    return false;
                }
            }

            if (body_quat[row].size() != body_quat[row - 1].size() ||
                body_quat[row].empty()) {
                reason = "replay transition quaternion row shape changed";
                return false;
            }
            for (std::size_t body = 0; body < body_quat[row].size(); ++body) {
                double angle = 0.0;
                if (!QuaternionAngle(
                        body_quat[row - 1][body], body_quat[row][body], angle)) {
                    reason = "invalid body quaternion between source frames " +
                             std::to_string(frame_indices[row - 1]) + " and " +
                             std::to_string(frame_indices[row]);
                    return false;
                }
                if (angle > kReplayBodyQuaternionHardAngle +
                                kReplayNumericLimitTolerance) {
                    reason = "body quaternion hard jump between source frames " +
                             std::to_string(frame_indices[row - 1]) + " and " +
                             std::to_string(frame_indices[row]) + ", body " +
                             std::to_string(body);
                    return false;
                }
            }
        }
        return true;
    }

    /** Validate that every repeated global row is unchanged. */
    static bool ValidateRawOverlap(
        const std::vector<std::int64_t>& frame_indices,
        const std::vector<std::vector<double>>& joint_pos,
        const std::vector<std::vector<double>>& joint_vel,
        const std::vector<std::vector<std::array<double, 4>>>& body_quat,
        const std::vector<std::int64_t>& previous_frame_indices,
        const std::vector<std::vector<double>>& previous_joint_pos,
        const std::vector<std::vector<double>>& previous_joint_vel,
        const std::vector<std::vector<std::array<double, 4>>>& previous_body_quat,
        std::size_t expected_rows,
        std::string& reason) {
        if (expected_rows == 0 || frame_indices.size() != expected_rows ||
            joint_pos.size() != expected_rows ||
            joint_vel.size() != expected_rows ||
            body_quat.size() != expected_rows) {
            reason = "current replay overlap snapshot requires exactly " +
                     std::to_string(expected_rows) + " aligned rows";
            return false;
        }
        if (previous_frame_indices.empty()) {
            return true;
        }
        if (previous_frame_indices.size() != expected_rows ||
            previous_joint_pos.size() != expected_rows ||
            previous_joint_vel.size() != expected_rows ||
            previous_body_quat.size() != expected_rows) {
            reason = "previous replay overlap snapshot requires exactly " +
                     std::to_string(expected_rows) + " aligned rows";
            return false;
        }
        for (std::size_t current = 0; current < frame_indices.size(); ++current) {
            const auto it = std::lower_bound(
                previous_frame_indices.begin(), previous_frame_indices.end(),
                frame_indices[current]);
            if (it == previous_frame_indices.end() || *it != frame_indices[current]) {
                continue;
            }
            const std::size_t previous = static_cast<std::size_t>(
                std::distance(previous_frame_indices.begin(), it));
            if (current >= joint_pos.size() ||
                current >= joint_vel.size() ||
                current >= body_quat.size() ||
                previous >= previous_joint_pos.size() ||
                previous >= previous_joint_vel.size() ||
                previous >= previous_body_quat.size() ||
                joint_pos[current].size() != previous_joint_pos[previous].size() ||
                joint_vel[current].size() != previous_joint_vel[previous].size() ||
                body_quat[current].size() != previous_body_quat[previous].size()) {
                reason = "overlapping replay row shape changed";
                return false;
            }
            for (std::size_t joint = 0; joint < joint_pos[current].size(); ++joint) {
                if (!NearlyEqual(
                        joint_pos[current][joint],
                        previous_joint_pos[previous][joint]) ||
                    !NearlyEqual(
                        joint_vel[current][joint],
                        previous_joint_vel[previous][joint])) {
                    reason = "overlapping replay joint row changed at source frame " +
                             std::to_string(frame_indices[current]);
                    return false;
                }
            }
            for (std::size_t body = 0; body < body_quat[current].size(); ++body) {
                for (std::size_t component = 0; component < 4; ++component) {
                    if (!NearlyEqual(
                            body_quat[current][body][component],
                            previous_body_quat[previous][body][component])) {
                        reason =
                            "overlapping replay quaternion row changed at source frame " +
                            std::to_string(frame_indices[current]);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    static bool ValidateWujiCommand(
        const WujiHandJointArray& command,
        std::string_view field_name,
        std::string& reason) {
        std::size_t invalid_index = 0;
        for (; invalid_index < command.size(); ++invalid_index) {
            const double value = command[invalid_index];
            if (!ReplayIsFinite(value) ||
                value < kWujiReplayCommandLowerLimits[invalid_index] -
                            kWujiReplayLimitTolerance ||
                value > kWujiReplayCommandUpperLimits[invalid_index] +
                            kWujiReplayLimitTolerance) {
                break;
            }
        }
        if (invalid_index == command.size()) {
            return true;
        }
        reason = std::string(field_name) + "[" + std::to_string(invalid_index) +
                 "] is non-finite or outside [" +
                 std::to_string(kWujiReplayCommandLowerLimits[invalid_index]) +
                 ", " +
                 std::to_string(kWujiReplayCommandUpperLimits[invalid_index]) + "]";
        return false;
    }

private:
    static bool NearlyEqual(double lhs, double rhs) {
        return ReplayIsFinite(lhs) && ReplayIsFinite(rhs) &&
               std::abs(lhs - rhs) <= kReplayNumericLimitTolerance;
    }

    static bool QuaternionAngle(
        const std::array<double, 4>& from,
        const std::array<double, 4>& to,
        double& angle) {
        double from_norm_squared = 0.0;
        double to_norm_squared = 0.0;
        double dot = 0.0;
        for (std::size_t component = 0; component < 4; ++component) {
            if (!ReplayIsFinite(from[component]) ||
                !ReplayIsFinite(to[component])) {
                return false;
            }
            from_norm_squared += from[component] * from[component];
            to_norm_squared += to[component] * to[component];
            dot += from[component] * to[component];
        }
        if (from_norm_squared <= std::numeric_limits<double>::epsilon() ||
            to_norm_squared <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        dot /= std::sqrt(from_norm_squared * to_norm_squared);
        dot = std::clamp(std::abs(dot), 0.0, 1.0);
        angle = 2.0 * std::acos(dot);
        return ReplayIsFinite(angle);
    }

    static std::size_t FloatElementBytes(std::string_view dtype) {
        if (dtype == "f32") {
            return sizeof(float);
        }
        if (dtype == "f64") {
            return sizeof(double);
        }
        return 0;
    }

    static std::size_t ElementCount(const std::vector<std::size_t>& shape) {
        std::size_t count = 1;
        for (const std::size_t dimension : shape) {
            count *= dimension;
        }
        return count;
    }

    static std::string FormatShape(const std::vector<std::size_t>& shape) {
        std::ostringstream stream;
        stream << "[";
        for (std::size_t index = 0; index < shape.size(); ++index) {
            if (index != 0) {
                stream << ",";
            }
            stream << shape[index];
        }
        stream << "]";
        return stream.str();
    }
};

/** Thread-safe validity for replay hands; an accepted packet replaces both sides. */
class ReplayHandValidity {
public:
    void Reset() {
        active_.store(false, std::memory_order_release);
        left_.store(false, std::memory_order_release);
        right_.store(false, std::memory_order_release);
    }

    void ApplyAcceptedPacket(bool left_valid, bool right_valid) {
        left_.store(left_valid, std::memory_order_release);
        right_.store(right_valid, std::memory_order_release);
        active_.store(true, std::memory_order_release);
    }

    bool Active() const {
        return active_.load(std::memory_order_acquire);
    }

    bool Valid(bool is_left) const {
        return is_left ? left_.load(std::memory_order_acquire) : right_.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool> active_{false};
    std::atomic<bool> left_{false};
    std::atomic<bool> right_{false};
};

}  // namespace sonic_replay

#endif  // REPLAY_PAYLOAD_SAFETY_HPP
