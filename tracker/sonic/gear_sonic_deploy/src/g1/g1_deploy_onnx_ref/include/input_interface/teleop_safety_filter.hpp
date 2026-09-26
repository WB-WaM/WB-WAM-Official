/**
 * @file teleop_safety_filter.hpp
 * @brief Lightweight per-frame safety gate for streamed teleop poses.
 */

#ifndef TELEOP_SAFETY_FILTER_HPP
#define TELEOP_SAFETY_FILTER_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../wuji_hand_constants.hpp"

/**
 * @class TeleopSafetyFilter
 * @brief Rejects non-finite teleop data and clamps small discontinuities.
 *
 * The filter intentionally keeps only the previous accepted sample per field.
 * This makes it cheap enough for the deploy input path while still catching
 * headset tracking jumps, dropped-frame spikes, and hand-target discontinuities.
 */
class TeleopSafetyFilter {
public:
    struct Limits {
        double vr_position_soft = 0.08;
        double vr_position_hard = 0.25;
        double vr_orientation_soft = 0.35;
        double vr_orientation_hard = 1.2;
        double body_quat_soft = 0.35;
        double body_quat_hard = 1.2;
        double smpl_joints_soft = 0.10;
        double smpl_joints_hard = 0.35;
        double smpl_pose_soft = 0.35;
        double smpl_pose_hard = 1.2;
        double joint_pos_soft = 0.25;
        double joint_pos_hard = 0.8;
        double wuji_qpos_soft = 0.10;
        double wuji_qpos_hard = 0.5;
    };

    struct Result {
        bool accepted = true;
        bool clamped = false;
        bool reset_requested = false;
        int consecutive_rejects = 0;
        std::string reason;
    };

    TeleopSafetyFilter()
        : TeleopSafetyFilter(Limits{}, 5) {}

    explicit TeleopSafetyFilter(Limits limits, int max_consecutive_hard_rejects = 5)
        : limits_(limits),
          max_consecutive_hard_rejects_(std::max(1, max_consecutive_hard_rejects)) {}

    static void SetGlobalEnabled(bool enabled) {
        GlobalEnabledFlag().store(enabled, std::memory_order_release);
    }

    static bool IsGlobalEnabled() {
        return GlobalEnabledFlag().load(std::memory_order_acquire);
    }

    void Reset() {
        state_ = State();
        consecutive_hard_rejects_ = 0;
    }

    int ConsecutiveRejects() const {
        return consecutive_hard_rejects_;
    }

    int MaxConsecutiveHardRejects() const {
        return max_consecutive_hard_rejects_;
    }

    Result FilterDecodedPose(
        std::vector<std::vector<double>>& joint_pos,
        bool has_joint_data,
        std::vector<std::vector<std::array<double, 4>>>& body_quat,
        std::vector<std::vector<std::array<double, 3>>>& smpl_joints,
        bool has_smpl_joints,
        std::vector<std::vector<std::array<double, 3>>>& smpl_pose,
        bool has_smpl_pose,
        bool has_left_hand,
        WujiHandJointArray& left_hand,
        bool has_right_hand,
        WujiHandJointArray& right_hand,
        bool has_vr_position,
        std::array<double, 9>& vr_position,
        bool has_vr_orientation,
        std::array<double, 12>& vr_orientation) {
        Result result;
        if (!IsGlobalEnabled()) {
            consecutive_hard_rejects_ = 0;
            return result;
        }

        State candidate = state_;
        if (!FilterScalarFrames(joint_pos, has_joint_data, limits_.joint_pos_soft, limits_.joint_pos_hard,
                                candidate.has_joint_pos, candidate.joint_pos, "joint_pos", result) ||
            !FilterQuatFrames(body_quat, !body_quat.empty(), limits_.body_quat_soft, limits_.body_quat_hard,
                              candidate.has_body_quat, candidate.body_quat, "body_quat_w", result) ||
            !FilterTripletFrames(smpl_joints, has_smpl_joints, limits_.smpl_joints_soft, limits_.smpl_joints_hard,
                                 candidate.has_smpl_joints, candidate.smpl_joints, "smpl_joints", result) ||
            !FilterTripletFrames(smpl_pose, has_smpl_pose, limits_.smpl_pose_soft, limits_.smpl_pose_hard,
                                 candidate.has_smpl_pose, candidate.smpl_pose, "smpl_pose", result) ||
            !FilterScalarArray(left_hand, has_left_hand, limits_.wuji_qpos_soft, limits_.wuji_qpos_hard,
                               candidate.has_left_hand, candidate.left_hand, "left_wuji_qpos", result) ||
            !FilterScalarArray(right_hand, has_right_hand, limits_.wuji_qpos_soft, limits_.wuji_qpos_hard,
                               candidate.has_right_hand, candidate.right_hand, "right_wuji_qpos", result) ||
            !FilterTripletArray(vr_position, has_vr_position, limits_.vr_position_soft, limits_.vr_position_hard,
                                candidate.has_vr_position, candidate.vr_position, "vr_position", result) ||
            !FilterQuatArray(vr_orientation, has_vr_orientation, limits_.vr_orientation_soft, limits_.vr_orientation_hard,
                             candidate.has_vr_orientation, candidate.vr_orientation, "vr_orientation", result)) {
            consecutive_hard_rejects_ += 1;
            result.accepted = false;
            result.consecutive_rejects = consecutive_hard_rejects_;
            result.reset_requested = consecutive_hard_rejects_ >= max_consecutive_hard_rejects_;
            return result;
        }

        state_ = std::move(candidate);
        consecutive_hard_rejects_ = 0;
        result.consecutive_rejects = 0;
        return result;
    }

private:
    struct State {
        bool has_joint_pos = false;
        std::vector<double> joint_pos;

        bool has_body_quat = false;
        std::vector<std::array<double, 4>> body_quat;

        bool has_smpl_joints = false;
        std::vector<std::array<double, 3>> smpl_joints;

        bool has_smpl_pose = false;
        std::vector<std::array<double, 3>> smpl_pose;

        bool has_left_hand = false;
        WujiHandJointArray left_hand = kDefaultWujiHandPose;

        bool has_right_hand = false;
        WujiHandJointArray right_hand = kDefaultWujiHandPose;

        bool has_vr_position = false;
        std::array<double, 9> vr_position{};

        bool has_vr_orientation = false;
        std::array<double, 12> vr_orientation{};
    };

    static std::atomic<bool>& GlobalEnabledFlag() {
        static std::atomic<bool> enabled{true};
        return enabled;
    }

    static bool IsFinite(double value) {
        // The deploy target is built with -ffast-math, which can make
        // std::isfinite unreliable. Check IEEE-754 exponent bits directly.
        static_assert(sizeof(double) == sizeof(std::uint64_t));
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    }

    static double Norm3(const std::array<double, 3>& value) {
        return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    }

    static bool NormalizeQuat(std::array<double, 4>& quat) {
        double norm = 0.0;
        for (double value : quat) {
            if (!IsFinite(value)) {
                return false;
            }
            norm += value * value;
        }
        norm = std::sqrt(norm);
        if (norm <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        for (double& value : quat) {
            value /= norm;
        }
        return true;
    }

    static double QuatAngle(std::array<double, 4> from, std::array<double, 4> to) {
        if (!NormalizeQuat(from) || !NormalizeQuat(to)) {
            return std::numeric_limits<double>::infinity();
        }
        double dot = 0.0;
        for (int i = 0; i < 4; ++i) {
            dot += from[i] * to[i];
        }
        dot = std::clamp(std::abs(dot), 0.0, 1.0);
        return 2.0 * std::acos(dot);
    }

    static std::array<double, 4> SlerpQuat(std::array<double, 4> from, std::array<double, 4> to, double t) {
        NormalizeQuat(from);
        NormalizeQuat(to);
        double dot = 0.0;
        for (int i = 0; i < 4; ++i) {
            dot += from[i] * to[i];
        }
        if (dot < 0.0) {
            dot = -dot;
            for (double& value : to) {
                value = -value;
            }
        }
        dot = std::clamp(dot, 0.0, 1.0);

        std::array<double, 4> output{};
        if (dot > 0.9995) {
            for (int i = 0; i < 4; ++i) {
                output[i] = from[i] + t * (to[i] - from[i]);
            }
            NormalizeQuat(output);
            return output;
        }

        const double theta = std::acos(dot);
        const double sin_theta = std::sin(theta);
        const double weight_from = std::sin((1.0 - t) * theta) / sin_theta;
        const double weight_to = std::sin(t * theta) / sin_theta;
        for (int i = 0; i < 4; ++i) {
            output[i] = weight_from * from[i] + weight_to * to[i];
        }
        NormalizeQuat(output);
        return output;
    }

    static std::string BuildReason(const std::string& field, std::size_t index, double delta, double hard_limit) {
        std::ostringstream oss;
        oss << field << "[" << index << "] jump " << delta << " exceeds hard limit " << hard_limit;
        return oss.str();
    }

    static bool FilterScalarSample(std::vector<double>& sample,
                                   double soft_limit,
                                   double hard_limit,
                                   bool& has_previous,
                                   std::vector<double>& previous,
                                   const std::string& field,
                                   Result& result) {
        if (!has_previous || previous.size() != sample.size()) {
            for (double value : sample) {
                if (!IsFinite(value)) {
                    result.reason = field + " contains non-finite value";
                    return false;
                }
            }
            previous = sample;
            has_previous = true;
            return true;
        }

        for (std::size_t i = 0; i < sample.size(); ++i) {
            if (!IsFinite(sample[i])) {
                result.reason = field + " contains non-finite value";
                return false;
            }
            const double delta = sample[i] - previous[i];
            const double abs_delta = std::abs(delta);
            if (abs_delta > hard_limit) {
                result.reason = BuildReason(field, i, abs_delta, hard_limit);
                return false;
            }
            if (abs_delta > soft_limit) {
                sample[i] = previous[i] + std::copysign(soft_limit, delta);
                result.clamped = true;
            }
        }
        previous = sample;
        return true;
    }

    static bool FilterScalarFrames(std::vector<std::vector<double>>& frames,
                                   bool has_field,
                                   double soft_limit,
                                   double hard_limit,
                                   bool& has_previous,
                                   std::vector<double>& previous,
                                   const std::string& field,
                                   Result& result) {
        if (!has_field) {
            return true;
        }
        for (auto& sample : frames) {
            if (!FilterScalarSample(sample, soft_limit, hard_limit, has_previous, previous, field, result)) {
                return false;
            }
        }
        return true;
    }

    static bool FilterScalarArray(WujiHandJointArray& sample,
                                  bool has_field,
                                  double soft_limit,
                                  double hard_limit,
                                  bool& has_previous,
                                  WujiHandJointArray& previous,
                                  const std::string& field,
                                  Result& result) {
        if (!has_field) {
            return true;
        }
        std::vector<double> sample_vec(sample.begin(), sample.end());
        std::vector<double> previous_vec(previous.begin(), previous.end());
        bool local_has_previous = has_previous;
        if (!FilterScalarSample(sample_vec, soft_limit, hard_limit, local_has_previous, previous_vec, field, result)) {
            return false;
        }
        std::copy(sample_vec.begin(), sample_vec.end(), sample.begin());
        std::copy(previous_vec.begin(), previous_vec.end(), previous.begin());
        has_previous = local_has_previous;
        return true;
    }

    static bool FilterTripletSample(std::vector<std::array<double, 3>>& sample,
                                    double soft_limit,
                                    double hard_limit,
                                    bool& has_previous,
                                    std::vector<std::array<double, 3>>& previous,
                                    const std::string& field,
                                    Result& result) {
        if (!has_previous || previous.size() != sample.size()) {
            for (const auto& point : sample) {
                for (double value : point) {
                    if (!IsFinite(value)) {
                        result.reason = field + " contains non-finite value";
                        return false;
                    }
                }
            }
            previous = sample;
            has_previous = true;
            return true;
        }

        for (std::size_t i = 0; i < sample.size(); ++i) {
            std::array<double, 3> delta{};
            for (int j = 0; j < 3; ++j) {
                if (!IsFinite(sample[i][j])) {
                    result.reason = field + " contains non-finite value";
                    return false;
                }
                delta[j] = sample[i][j] - previous[i][j];
            }
            const double norm = Norm3(delta);
            if (norm > hard_limit) {
                result.reason = BuildReason(field, i, norm, hard_limit);
                return false;
            }
            if (norm > soft_limit && norm > std::numeric_limits<double>::epsilon()) {
                const double scale = soft_limit / norm;
                for (int j = 0; j < 3; ++j) {
                    sample[i][j] = previous[i][j] + delta[j] * scale;
                }
                result.clamped = true;
            }
        }
        previous = sample;
        return true;
    }

    static bool FilterTripletFrames(std::vector<std::vector<std::array<double, 3>>>& frames,
                                    bool has_field,
                                    double soft_limit,
                                    double hard_limit,
                                    bool& has_previous,
                                    std::vector<std::array<double, 3>>& previous,
                                    const std::string& field,
                                    Result& result) {
        if (!has_field) {
            return true;
        }
        for (auto& sample : frames) {
            if (!FilterTripletSample(sample, soft_limit, hard_limit, has_previous, previous, field, result)) {
                return false;
            }
        }
        return true;
    }

    static bool FilterTripletArray(std::array<double, 9>& sample,
                                   bool has_field,
                                   double soft_limit,
                                   double hard_limit,
                                   bool& has_previous,
                                   std::array<double, 9>& previous,
                                   const std::string& field,
                                   Result& result) {
        if (!has_field) {
            return true;
        }
        std::vector<std::array<double, 3>> sample_vec(3);
        std::vector<std::array<double, 3>> previous_vec(3);
        for (int tracker = 0; tracker < 3; ++tracker) {
            for (int axis = 0; axis < 3; ++axis) {
                sample_vec[tracker][axis] = sample[tracker * 3 + axis];
                previous_vec[tracker][axis] = previous[tracker * 3 + axis];
            }
        }

        bool local_has_previous = has_previous;
        if (!FilterTripletSample(sample_vec, soft_limit, hard_limit, local_has_previous, previous_vec, field, result)) {
            return false;
        }
        for (int tracker = 0; tracker < 3; ++tracker) {
            for (int axis = 0; axis < 3; ++axis) {
                sample[tracker * 3 + axis] = sample_vec[tracker][axis];
                previous[tracker * 3 + axis] = previous_vec[tracker][axis];
            }
        }
        has_previous = local_has_previous;
        return true;
    }

    static bool FilterQuatSample(std::vector<std::array<double, 4>>& sample,
                                 double soft_limit,
                                 double hard_limit,
                                 bool& has_previous,
                                 std::vector<std::array<double, 4>>& previous,
                                 const std::string& field,
                                 Result& result) {
        for (auto& quat : sample) {
            if (!NormalizeQuat(quat)) {
                result.reason = field + " contains invalid quaternion";
                return false;
            }
        }

        if (!has_previous || previous.size() != sample.size()) {
            previous = sample;
            has_previous = true;
            return true;
        }

        for (std::size_t i = 0; i < sample.size(); ++i) {
            const double angle = QuatAngle(previous[i], sample[i]);
            if (!IsFinite(angle) || angle > hard_limit) {
                result.reason = BuildReason(field, i, angle, hard_limit);
                return false;
            }
            if (angle > soft_limit && angle > std::numeric_limits<double>::epsilon()) {
                sample[i] = SlerpQuat(previous[i], sample[i], soft_limit / angle);
                result.clamped = true;
            }
        }
        previous = sample;
        return true;
    }

    static bool FilterQuatFrames(std::vector<std::vector<std::array<double, 4>>>& frames,
                                 bool has_field,
                                 double soft_limit,
                                 double hard_limit,
                                 bool& has_previous,
                                 std::vector<std::array<double, 4>>& previous,
                                 const std::string& field,
                                 Result& result) {
        if (!has_field) {
            return true;
        }
        for (auto& sample : frames) {
            if (!FilterQuatSample(sample, soft_limit, hard_limit, has_previous, previous, field, result)) {
                return false;
            }
        }
        return true;
    }

    static bool FilterQuatArray(std::array<double, 12>& sample,
                                bool has_field,
                                double soft_limit,
                                double hard_limit,
                                bool& has_previous,
                                std::array<double, 12>& previous,
                                const std::string& field,
                                Result& result) {
        if (!has_field) {
            return true;
        }
        std::vector<std::array<double, 4>> sample_vec(3);
        std::vector<std::array<double, 4>> previous_vec(3);
        for (int tracker = 0; tracker < 3; ++tracker) {
            for (int axis = 0; axis < 4; ++axis) {
                sample_vec[tracker][axis] = sample[tracker * 4 + axis];
                previous_vec[tracker][axis] = previous[tracker * 4 + axis];
            }
        }

        bool local_has_previous = has_previous;
        if (!FilterQuatSample(sample_vec, soft_limit, hard_limit, local_has_previous, previous_vec, field, result)) {
            return false;
        }
        for (int tracker = 0; tracker < 3; ++tracker) {
            for (int axis = 0; axis < 4; ++axis) {
                sample[tracker * 4 + axis] = sample_vec[tracker][axis];
                previous[tracker * 4 + axis] = previous_vec[tracker][axis];
            }
        }
        has_previous = local_has_previous;
        return true;
    }

    Limits limits_;
    int max_consecutive_hard_rejects_;
    int consecutive_hard_rejects_ = 0;
    State state_;
};

#endif // TELEOP_SAFETY_FILTER_HPP
