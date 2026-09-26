/**
 * @file zmq_state_action_output_handler.hpp
 * @brief ZMQ PUB output handler for collector-friendly robot state + action data.
 */

#ifndef ZMQ_STATE_ACTION_OUTPUT_HANDLER_HPP
#define ZMQ_STATE_ACTION_OUTPUT_HANDLER_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <zmq.hpp>

#include "output_interface.hpp"

class ZMQStateActionOutputHandler : public OutputInterface {
public:
    static std::string MakeReplayTopic(const std::string& topic) {
        return "replay_" + topic;
    }

    static void BuildStateActionMessage(
        const StateLogger::Entry& state,
        const WujiHandJointArray& current_left_hand_action,
        const WujiHandJointArray& current_right_hand_action,
        const std::string& topic,
        std::string& message
    ) {
        std::vector<FieldSpec> fields;
        std::string payload;
        fields.reserve(14);
        payload.reserve(4096);

        AppendScalar(fields, payload, "state_index", static_cast<int64_t>(state.index), "i64");
        AppendScalar(
            fields,
            payload,
            "timestamp_monotonic_ns",
            static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    state.timestamp_monotonic.time_since_epoch()
                )
                    .count()
            ),
            "i64"
        );
        AppendArray(fields, payload, "base_quat", state.base_quat);
        AppendArray(fields, payload, "base_ang_vel", state.base_ang_vel);
        AppendArray(fields, payload, "base_accel", state.base_accel);
        AppendVector(fields, payload, "body_q", state.body_q);
        AppendVector(fields, payload, "body_dq", state.body_dq);
        AppendVector(fields, payload, "body_action", state.last_action);
        AppendVector(fields, payload, "left_hand_q", state.left_hand_q);
        AppendVector(fields, payload, "right_hand_q", state.right_hand_q);
        AppendArray(fields, payload, "left_hand_action", current_left_hand_action);
        AppendArray(fields, payload, "right_hand_action", current_right_hand_action);
        AppendVector(fields, payload, "token_state", state.token_state);
        AppendScalar(fields, payload, "encoder_mode", static_cast<int64_t>(state.encoder_mode), "i64");

        const std::string header = BuildHeader(fields);
        message.clear();
        message.reserve(topic.size() + header.size() + payload.size());
        message.append(topic);
        message.append(header);
        message.append(payload);
    }

    static void BuildReplayMessage(
        const StateLogger::Entry& state,
        const std::string& topic,
        std::string& message
    ) {
        std::vector<FieldSpec> replay_fields;
        std::string replay_payload;
        replay_fields.reserve(20);
        replay_payload.reserve(1024);

        AppendScalar(
            replay_fields,
            replay_payload,
            "state_index",
            static_cast<int64_t>(state.index),
            "i64"
        );
        AppendScalar(
            replay_fields,
            replay_payload,
            "timestamp_monotonic_ns",
            static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    state.timestamp_monotonic.time_since_epoch()
                )
                    .count()
            ),
            "i64"
        );
        AppendScalar(
            replay_fields,
            replay_payload,
            "replay_route",
            state.replay_route,
            "i64"
        );
        AppendScalar(
            replay_fields,
            replay_payload,
            "replay_stream_epoch",
            state.replay_stream_epoch,
            "i64"
        );
        AppendScalar(
            replay_fields,
            replay_payload,
            "replay_source_frame",
            state.replay_source_frame,
            "i64"
        );
        const uint8_t encoder_cache_refresh =
            state.encoder_cache_refresh ? 1U : 0U;
        AppendScalar(
            replay_fields,
            replay_payload,
            "encoder_cache_refresh",
            encoder_cache_refresh,
            "bool"
        );
        AppendArray(
            replay_fields,
            replay_payload,
            "encoder_input_indices",
            state.encoder_input_indices
        );
        AppendScalar(
            replay_fields,
            replay_payload,
            "last_pose_rx_age_ms",
            state.last_pose_rx_age_ms,
            "f64"
        );
        AppendScalar(
            replay_fields,
            replay_payload,
            "replay_fault_code",
            state.replay_fault_code,
            "i64"
        );
        AppendScalar(
            replay_fields, replay_payload, "safety_telemetry_version",
            state.safety_telemetry_version, "i64");
        const uint8_t safety_fall_detected =
            state.safety_fall_detected ? 1U : 0U;
        AppendScalar(
            replay_fields, replay_payload, "safety_fall_detected",
            safety_fall_detected, "bool");
        AppendScalar(
            replay_fields, replay_payload, "safety_reset_count",
            state.safety_reset_count, "i64");
        AppendScalar(
            replay_fields, replay_payload,
            "safety_joint_position_limit_violation_count",
            state.safety_joint_position_limit_violation_count, "i64");
        AppendScalar(
            replay_fields, replay_payload,
            "safety_joint_velocity_limit_violation_count",
            state.safety_joint_velocity_limit_violation_count, "i64");
        AppendScalar(
            replay_fields, replay_payload,
            "safety_command_limit_violation_count",
            state.safety_command_limit_violation_count, "i64");
        AppendArray(replay_fields, replay_payload, "q_target", state.q_target);
        AppendArray(replay_fields, replay_payload, "dq_target", state.dq_target);
        AppendArray(replay_fields, replay_payload, "kp", state.kp);
        AppendArray(replay_fields, replay_payload, "kd", state.kd);
        AppendArray(replay_fields, replay_payload, "tau_ff", state.tau_ff);
        // Append-only for wire compatibility: existing replay field offsets
        // remain unchanged. This acknowledges that Planner is actually
        // playing planner_motion, rather than merely reporting replay_route=0
        // while a streamed-motion reset is still in progress.
        const uint8_t planner_control_active =
            state.has_post_state_data && state.play
                && state.motion_name == "planner_motion" ? 1U : 0U;
        AppendScalar(
            replay_fields, replay_payload, "planner_control_active",
            planner_control_active, "bool");
        AppendScalar(
            replay_fields, replay_payload, "hand_frame_index",
            state.hand_frame_index, "i64");


        const std::string replay_header = BuildHeader(replay_fields, true);
        message.clear();
        message.reserve(topic.size() + replay_header.size() +
                        replay_payload.size());
        message.append(topic);
        message.append(replay_header);
        message.append(replay_payload);
    }

    explicit ZMQStateActionOutputHandler(StateLogger& logger, int port, const std::string& topic)
        : OutputInterface(logger), context_(1), topic_(topic), replay_topic_(MakeReplayTopic(topic)) {
        socket_ = std::make_unique<zmq::socket_t>(context_, ZMQ_PUB);
        socket_->set(zmq::sockopt::sndhwm, 10);
        socket_->set(zmq::sockopt::sndbuf, 32768);
        socket_->set(zmq::sockopt::linger, 0);
        socket_->bind("tcp://*:" + std::to_string(port));

        std::cout << "[INFO] Collector state/action socket bound to port: " << port
                  << " topic: " << topic_ << std::endl;
        type_ = OutputType::ZMQ;
    }

    void publish(
        const std::array<double, 9>&,
        const std::array<double, 12>&,
        const std::array<double, 3>&,
        const WujiHandJointArray& current_left_hand_action,
        const WujiHandJointArray& current_right_hand_action,
        const std::array<double, 4>&,
        DataBuffer<HeadingState>&,
        std::shared_ptr<const MotionSequence>,
        int
    ) override {
        const auto entries = state_logger_.GetLatest(1);
        if (entries.empty()) {
            return;
        }

        const auto& state = entries.front();
        BuildStateActionMessage(
            state,
            current_left_hand_action,
            current_right_hand_action,
            topic_,
            message_buffer_
        );
        if (SendDontWait(message_buffer_)) {
            std::lock_guard<std::mutex> lock(last_published_state_mutex_);
            last_published_state_ = state;
        }
        (void)PublishReplayEntry(state);
    }
    bool GetLastPublishedStateEntry(
        StateLogger::Entry& entry) const noexcept override {
        std::lock_guard<std::mutex> lock(last_published_state_mutex_);
        if (!last_published_state_.has_value()) {
            return false;
        }
        entry = *last_published_state_;
        return true;
    }

    bool PublishReplayEntry(
        const StateLogger::Entry& entry) noexcept override {
        try {
            BuildReplayMessage(entry, replay_topic_, replay_message_buffer_);
            return SendDontWait(replay_message_buffer_);
        } catch (const std::exception&) {
            return false;
        }
    }

private:
    struct FieldSpec {
        std::string name;
        std::string dtype;
        std::vector<int> shape;
    };

    static constexpr std::size_t kHeaderSize = 1280;

    template <typename T>
    static void AppendBytes(std::string& payload, const T* data, std::size_t count) {
        payload.append(reinterpret_cast<const char*>(data), sizeof(T) * count);
    }

    template <typename T>
    static void AppendScalar(
        std::vector<FieldSpec>& fields,
        std::string& payload,
        const std::string& name,
        const T& value,
        const std::string& dtype
    ) {
        fields.push_back(FieldSpec{name, dtype, {1}});
        AppendBytes(payload, &value, 1);
    }

    template <typename T, std::size_t N>
    static void AppendArray(
        std::vector<FieldSpec>& fields,
        std::string& payload,
        const std::string& name,
        const std::array<T, N>& values
    ) {
        fields.push_back(FieldSpec{name, SelectDType<T>(), {static_cast<int>(N)}});
        AppendBytes(payload, values.data(), values.size());
    }

    template <typename T>
    static void AppendVector(
        std::vector<FieldSpec>& fields,
        std::string& payload,
        const std::string& name,
        const std::vector<T>& values
    ) {
        fields.push_back(FieldSpec{name, SelectDType<T>(), {static_cast<int>(values.size())}});
        if (!values.empty()) {
            AppendBytes(payload, values.data(), values.size());
        }
    }

    template <typename T>
    static std::string SelectDType() {
        if constexpr (std::is_same_v<T, double>) {
            return "f64";
        } else if constexpr (std::is_same_v<T, float>) {
            return "f32";
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return "i64";
        } else {
            throw std::runtime_error("Unsupported collector ZMQ dtype");
        }
    }

    static std::string BuildHeader(
        const std::vector<FieldSpec>& fields, bool compact = false) {
        std::ostringstream oss;
        if (compact) {
            oss << "{\"v\":4,\"fields\":[";
        } else {
            oss << "{\"v\":4,\"endian\":\"le\",\"count\":1,\"fields\":[";
        }
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i != 0) {
                oss << ",";
            }
            oss << "{\"name\":\"" << fields[i].name << "\",\"dtype\":\"" << fields[i].dtype
                << "\",\"shape\":[";
            for (std::size_t j = 0; j < fields[i].shape.size(); ++j) {
                if (j != 0) {
                    oss << ",";
                }
                oss << fields[i].shape[j];
            }
            oss << "]}";
        }
        oss << "]}";

        std::string header = oss.str();
        if (header.size() > kHeaderSize) {
            throw std::runtime_error("Collector header exceeds fixed size");
        }
        header.resize(kHeaderSize, '\0');
        return header;
    }

    bool SendDontWait(const std::string& message) noexcept {
        try {
            return socket_
                ->send(zmq::buffer(message), zmq::send_flags::dontwait)
                .has_value();
        } catch (const zmq::error_t&) {
            // Telemetry is best-effort and must never stall or unwind.
            return false;
        }
    }

    zmq::context_t context_;
    std::unique_ptr<zmq::socket_t> socket_;
    std::string topic_;
    mutable std::mutex last_published_state_mutex_;
    std::optional<StateLogger::Entry> last_published_state_;
    std::string replay_topic_;
    std::string message_buffer_;
    std::string replay_message_buffer_;
};

#endif  // ZMQ_STATE_ACTION_OUTPUT_HANDLER_HPP
