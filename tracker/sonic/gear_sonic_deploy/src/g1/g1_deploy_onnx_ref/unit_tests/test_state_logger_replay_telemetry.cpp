#include "state_logger.hpp"
#include "output_interface/replay_terminal_telemetry.hpp"
#include "utils.hpp"
#include "output_interface/zmq_state_action_output_handler.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

void LogEmptyState(StateLogger& logger) {
  std::array<double, 4> quat {1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> vector3 {};
  std::array<double, G1_NUM_MOTOR> body {};
  std::array<double, 20> hand {};
  std::span<double> body_span(body);
  std::span<double> hand_span(hand);

  logger.LogFullState(
    quat,
    vector3,
    vector3,
    quat,
    vector3,
    vector3,
    body_span,
    body_span,
    body_span,
    hand_span,
    hand_span,
    hand_span,
    hand_span,
    hand_span,
    hand_span
  );
}

TEST(StateLoggerReplayTelemetry, StoresReplayMetadataAndMotorCommand) {
  StateLogger logger("", 4, G1_NUM_MOTOR, G1_NUM_MOTOR, 0.02, false);
  LogEmptyState(logger);

  const std::array<int64_t, 10> encoder_indices {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  EXPECT_TRUE(logger.LogPostReplayMetadata(
    1, 7, 42, true, encoder_indices, 12.5, 0, true, 2, 3, 4, 5, 99));

  MotorCommand command;
  for (std::size_t i = 0; i < command.q_target.size(); ++i) {
    command.q_target[i] = static_cast<float>(i) + 0.1F;
    command.dq_target[i] = static_cast<float>(i) + 0.2F;
    command.kp[i] = static_cast<float>(i) + 0.3F;
    command.kd[i] = static_cast<float>(i) + 0.4F;
    command.tau_ff[i] = static_cast<float>(i) + 0.5F;
  }
  EXPECT_TRUE(logger.LogPostMotorCommand(command));

  const StateLogger::Entry entry = logger.GetLatest(1).front();
  EXPECT_TRUE(entry.has_replay_metadata);
  EXPECT_EQ(entry.replay_route, 1);
  EXPECT_EQ(entry.replay_stream_epoch, 7);
  EXPECT_EQ(entry.replay_source_frame, 42);
  EXPECT_TRUE(entry.encoder_cache_refresh);
  EXPECT_EQ(entry.encoder_input_indices, encoder_indices);
  EXPECT_DOUBLE_EQ(entry.last_pose_rx_age_ms, 12.5);
  EXPECT_EQ(entry.replay_fault_code, 0);
  EXPECT_EQ(entry.safety_telemetry_version, 1);
  EXPECT_EQ(entry.hand_frame_index, 99);
  EXPECT_TRUE(entry.safety_fall_detected);
  EXPECT_EQ(entry.safety_reset_count, 2);
  EXPECT_EQ(entry.safety_joint_position_limit_violation_count, 3);
  EXPECT_EQ(entry.safety_joint_velocity_limit_violation_count, 4);
  EXPECT_EQ(entry.safety_command_limit_violation_count, 5);

  EXPECT_TRUE(entry.has_motor_command);
  EXPECT_EQ(entry.q_target, command.q_target);
  EXPECT_EQ(entry.dq_target, command.dq_target);
  EXPECT_EQ(entry.kp, command.kp);
  EXPECT_EQ(entry.kd, command.kd);
  EXPECT_EQ(entry.tau_ff, command.tau_ff);
}

TEST(StateLoggerReplayTelemetry, DefaultEntryRepresentsNoReplay) {
  StateLogger logger("", 4, G1_NUM_MOTOR, G1_NUM_MOTOR, 0.02, false);
  LogEmptyState(logger);

  const StateLogger::Entry entry = logger.GetLatest(1).front();
  EXPECT_FALSE(entry.has_replay_metadata);
  EXPECT_EQ(entry.replay_route, 0);
  EXPECT_EQ(entry.replay_stream_epoch, -1);
  EXPECT_EQ(entry.replay_source_frame, -1);
  EXPECT_FALSE(entry.encoder_cache_refresh);
  EXPECT_EQ(entry.last_pose_rx_age_ms, -1.0);
  EXPECT_EQ(entry.replay_fault_code, 0);
  EXPECT_EQ(entry.hand_frame_index, -1);
  EXPECT_FALSE(entry.has_motor_command);
}

TEST(StateLoggerReplayTelemetry, ReplayTopicDoesNotMatchLegacySubscriptionPrefix) {
  const std::string legacy_topic = "robot_state_action";
  const std::string replay_topic = ZMQStateActionOutputHandler::MakeReplayTopic(legacy_topic);

  EXPECT_EQ(replay_topic, "replay_robot_state_action");
  EXPECT_NE(replay_topic.compare(0, legacy_topic.size(), legacy_topic), 0);
}

TEST(StateLoggerReplayTelemetry, WireHandActionsUseCurrentPublishArguments) {
  StateLogger logger("", 4, G1_NUM_MOTOR, G1_NUM_MOTOR, 0.02, false);
  LogEmptyState(logger);
  const StateLogger::Entry entry = logger.GetLatest(1).front();

  WujiHandJointArray current_left {};
  WujiHandJointArray current_right {};
  for (std::size_t i = 0; i < current_left.size(); ++i) {
    current_left[i] = 100.0 + static_cast<double>(i);
    current_right[i] = 200.0 + static_cast<double>(i);
  }
  ASSERT_NE(entry.last_left_hand_action, std::vector<double>(current_left.begin(), current_left.end()));
  ASSERT_NE(entry.last_right_hand_action, std::vector<double>(current_right.begin(), current_right.end()));

  const std::string topic = "robot_state_action";
  std::string message;
  ZMQStateActionOutputHandler::BuildStateActionMessage(
    entry,
    current_left,
    current_right,
    topic,
    message
  );
  ASSERT_EQ(message.compare(0, topic.size(), topic), 0);

  constexpr std::size_t header_size = 1280;
  const std::string header = message.substr(topic.size(), header_size);
  EXPECT_NE(
    header.find("\"name\":\"left_hand_action\",\"dtype\":\"f64\",\"shape\":[20]"),
    std::string::npos
  );
  EXPECT_NE(
    header.find("\"name\":\"right_hand_action\",\"dtype\":\"f64\",\"shape\":[20]"),
    std::string::npos
  );

  const std::size_t values_before_hand_actions =
    entry.base_quat.size() +
    entry.base_ang_vel.size() +
    entry.base_accel.size() +
    entry.body_q.size() +
    entry.body_dq.size() +
    entry.last_action.size() +
    entry.left_hand_q.size() +
    entry.right_hand_q.size();
  std::size_t payload_offset = 2 * sizeof(int64_t) + values_before_hand_actions * sizeof(double);
  const char* payload = message.data() + topic.size() + header_size;

  WujiHandJointArray decoded_left {};
  WujiHandJointArray decoded_right {};
  std::memcpy(decoded_left.data(), payload + payload_offset, decoded_left.size() * sizeof(double));
  payload_offset += decoded_left.size() * sizeof(double);
  std::memcpy(decoded_right.data(), payload + payload_offset, decoded_right.size() * sizeof(double));

  EXPECT_EQ(decoded_left, current_left);
  EXPECT_EQ(decoded_right, current_right);
}

TEST(StateLoggerReplayTelemetry, TerminalFaultMaterializerPreservesPublishedState) {
  StateLogger::Entry entry;
  entry.index = 77;
  entry.has_motor_command = true;
  entry.q_target[0] = 1.25F;
  entry.kd[0] = 8.0F;

  sonic_replay::ReplayStreamInfo info;
  info.faulted = true;
  info.has_packet = true;
  info.source_hz = 20.0;
  info.stream_epoch = 9;
  info.source_frame = 123;

  EXPECT_TRUE(sonic_replay::MaterializeTerminalReplayFault(
    entry, info, 275.5
  ));
  EXPECT_EQ(entry.index, 77U);
  EXPECT_TRUE(entry.has_motor_command);
  EXPECT_FLOAT_EQ(entry.q_target[0], 1.25F);
  EXPECT_FLOAT_EQ(entry.kd[0], 8.0F);
  EXPECT_TRUE(entry.has_replay_metadata);
  EXPECT_EQ(entry.replay_route, 1);
  EXPECT_EQ(entry.replay_stream_epoch, 9);
  EXPECT_EQ(entry.replay_source_frame, 123);
  EXPECT_FALSE(entry.encoder_cache_refresh);
  EXPECT_EQ(
    entry.encoder_input_indices,
    (std::array<int64_t, 10> {0, 2, 4, 6, 8, 10, 12, 14, 16, 18})
  );
  EXPECT_DOUBLE_EQ(entry.last_pose_rx_age_ms, 275.5);
  EXPECT_EQ(entry.replay_fault_code, 1);
}

TEST(StateLoggerReplayTelemetry, TerminalFaultMaterializerUsesFiftyHertzIndices) {
  StateLogger::Entry entry;
  sonic_replay::ReplayStreamInfo info;
  info.faulted = true;
  info.has_packet = true;
  info.source_hz = 50.0;
  info.stream_epoch = 10;
  info.source_frame = 124;

  EXPECT_TRUE(sonic_replay::MaterializeTerminalReplayFault(
    entry, info, 12.0
  ));
  EXPECT_EQ(entry.replay_route, 1);
  EXPECT_EQ(
    entry.encoder_input_indices,
    (std::array<int64_t, 10> {0, 5, 10, 15, 20, 25, 30, 35, 40, 45})
  );
}

TEST(StateLoggerReplayTelemetry, TerminalFaultMaterializerHandlesSavedToken) {
  StateLogger::Entry entry;
  sonic_replay::ReplayStreamInfo info;
  info.faulted = true;
  info.source_hz = 0.0;
  info.stream_epoch = 3;
  info.source_frame = 41;

  EXPECT_TRUE(sonic_replay::MaterializeTerminalReplayFault(
    entry, info, 251.0
  ));
  EXPECT_EQ(entry.replay_route, 2);
  for (const int64_t index : entry.encoder_input_indices) {
    EXPECT_EQ(index, -1);
  }

  info.faulted = false;
  EXPECT_FALSE(sonic_replay::MaterializeTerminalReplayFault(
    entry, info, 0.0
  ));
}

TEST(StateLoggerReplayTelemetry, TerminalPublishGateAllowsOnlyOneAttempt) {
  sonic_replay::TerminalReplayPublishGate gate;
  EXPECT_FALSE(gate.Published());
  EXPECT_TRUE(gate.TryBegin());
  EXPECT_FALSE(gate.TryBegin());

  gate.Retry();
  EXPECT_TRUE(gate.TryBegin());
  gate.Complete();

  EXPECT_TRUE(gate.Published());
  EXPECT_FALSE(gate.TryBegin());
}

TEST(StateLoggerReplayTelemetry, TerminalReplayWireCarriesFaultOnSameIndex) {
  StateLogger::Entry entry;
  entry.index = 88;
  entry.timestamp_monotonic =
    std::chrono::steady_clock::time_point(std::chrono::nanoseconds(1234));
  entry.replay_route = 1;
  entry.replay_stream_epoch = 5;
  entry.replay_source_frame = 144;
  entry.encoder_input_indices =
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  entry.last_pose_rx_age_ms = 260.0;
  entry.replay_fault_code = 1;
  entry.safety_telemetry_version = 1;
  entry.hand_frame_index = 145;
  entry.safety_fall_detected = true;
  entry.safety_reset_count = 2;
  entry.safety_joint_position_limit_violation_count = 3;
  entry.safety_joint_velocity_limit_violation_count = 4;
  entry.safety_command_limit_violation_count = 5;
  entry.q_target[0] = 2.5F;
  entry.has_post_state_data = true;
  entry.motion_name = "planner_motion";
  entry.play = true;

  const std::string topic = "replay_robot_state_action";
  std::string message;
  ZMQStateActionOutputHandler::BuildReplayMessage(
    entry, topic, message
  );

  ASSERT_EQ(message.compare(0, topic.size(), topic), 0);
  constexpr std::size_t header_size = 1280;
  const std::string header = message.substr(topic.size(), header_size);
  EXPECT_NE(header.find("replay_source_frame"), std::string::npos);
  EXPECT_NE(header.find("replay_fault_code"), std::string::npos);
  EXPECT_NE(header.find("safety_telemetry_version"), std::string::npos);
  EXPECT_NE(header.find("safety_fall_detected"), std::string::npos);
  EXPECT_NE(header.find("safety_reset_count"), std::string::npos);
  EXPECT_NE(header.find("safety_command_limit_violation_count"),
            std::string::npos);
  EXPECT_NE(header.find("planner_control_active"), std::string::npos);
  EXPECT_NE(header.find("hand_frame_index"), std::string::npos);

  const char* payload = message.data() + topic.size() + header_size;
  int64_t state_index = -1;
  int64_t source_frame = -1;
  int64_t fault_code = 0;
  int64_t safety_version = 0;
  uint8_t fall_detected = 0;
  int64_t reset_count = 0;
  int64_t command_violation_count = 0;
  float q_target0 = 0.0F;
  uint8_t planner_control_active = 0;
  int64_t hand_frame_index = -1;
  std::memcpy(&state_index, payload, sizeof(state_index));
  std::memcpy(&source_frame, payload + 32, sizeof(source_frame));
  std::memcpy(&fault_code, payload + 129, sizeof(fault_code));
  std::memcpy(&safety_version, payload + 137, sizeof(safety_version));
  std::memcpy(&fall_detected, payload + 145, sizeof(fall_detected));
  std::memcpy(&reset_count, payload + 146, sizeof(reset_count));
  std::memcpy(
    &command_violation_count, payload + 170,
    sizeof(command_violation_count));
  std::memcpy(&q_target0, payload + 178, sizeof(q_target0));
  std::memcpy(
    &planner_control_active, payload + 758, sizeof(planner_control_active));
  std::memcpy(
    &hand_frame_index, payload + 759, sizeof(hand_frame_index));

  EXPECT_EQ(state_index, 88);
  EXPECT_EQ(source_frame, 144);
  EXPECT_EQ(fault_code, 1);
  EXPECT_EQ(safety_version, 1);
  EXPECT_EQ(fall_detected, 1);
  EXPECT_EQ(reset_count, 2);
  EXPECT_EQ(command_violation_count, 5);
  EXPECT_FLOAT_EQ(q_target0, 2.5F);
  EXPECT_EQ(planner_control_active, 1);
  EXPECT_EQ(hand_frame_index, 145);
}

}  // namespace
