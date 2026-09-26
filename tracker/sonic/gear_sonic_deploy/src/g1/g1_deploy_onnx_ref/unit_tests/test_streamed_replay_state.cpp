#include <array>
#include <cmath>

#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include <gtest/gtest.h>

#include "input_interface/input_interface.hpp"
#include "input_interface/replay_payload_safety.hpp"
#include "input_interface/streamed_motion_merger.hpp"
#include "input_interface/streamed_replay_state.hpp"

namespace {

using sonic_replay::PacketDecision;
using sonic_replay::StreamedReplayState;

TEST(StreamedReplayStateTest, LegacyOffsetsRemainControlRateOffsets) {
    std::vector<int> offsets;
    for (int i = 0; i < 10; ++i) {
        offsets.push_back(
            StreamedReplayState::ResolveObservationFrameOffset(i, 5, 0.0, 50.0));
    }
    EXPECT_EQ(offsets, (std::vector<int>{0, 5, 10, 15, 20, 25, 30, 35, 40, 45}));
    EXPECT_EQ(StreamedReplayState::RequiredRows(10, 5, 0.0, 50.0), 46);
}

TEST(StreamedReplayStateTest, TwentyHertzOffsetsPreservePhysicalLookahead) {
    std::vector<int> offsets;
    for (int i = 0; i < 10; ++i) {
        offsets.push_back(
            StreamedReplayState::ResolveObservationFrameOffset(i, 5, 20.0, 50.0));
    }
    EXPECT_EQ(offsets, (std::vector<int>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}));
    EXPECT_EQ(StreamedReplayState::RequiredRows(10, 5, 20.0, 50.0), 19);
}

TEST(StreamedReplayStateTest, FiftyHertzOffsetsMatchEncoderTelemetryWindow) {
    std::vector<int> offsets;
    for (int i = 0; i < 10; ++i) {
        offsets.push_back(
            StreamedReplayState::ResolveObservationFrameOffset(i, 5, 50.0, 50.0));
    }
    EXPECT_EQ(offsets, (std::vector<int>{0, 5, 10, 15, 20, 25, 30, 35, 40, 45}));
    EXPECT_EQ(StreamedReplayState::RequiredRows(10, 5, 50.0, 50.0), 46);
}

TEST(StreamedReplayStateTest, SourceRateMustPreserveExactTenthSecondRows) {
    EXPECT_TRUE(StreamedReplayState::IsSupportedSourceRate(20.0));
    EXPECT_TRUE(StreamedReplayState::IsSupportedSourceRate(50.0));
    EXPECT_EQ(StreamedReplayState::ExpectedSnapshotRows(20.0), 20U);
    EXPECT_EQ(StreamedReplayState::ExpectedSnapshotRows(50.0), 46U);
    for (double rate : {-1.0, 0.0, 10.0, 17.0, 20.5, 30.0, 40.0, 60.0}) {
        EXPECT_FALSE(StreamedReplayState::IsSupportedSourceRate(rate));
        EXPECT_EQ(StreamedReplayState::ExpectedSnapshotRows(rate), 0U);
    }
}

TEST(StreamedReplayStateTest, SavedTokenMustBeFinite64DAndOnFsqGrid) {
    std::vector<double> token(64, 0.125);
    std::string reason;
    EXPECT_TRUE(StreamedReplayState::ValidateFsqTokenState(token, reason));

    token[7] = 0.13;
    EXPECT_FALSE(StreamedReplayState::ValidateFsqTokenState(token, reason));
    token[7] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(StreamedReplayState::ValidateFsqTokenState(token, reason));
    token.assign(63, 0.0);
    EXPECT_FALSE(StreamedReplayState::ValidateFsqTokenState(token, reason));
}

TEST(StreamedReplayStateTest, EncoderCacheRefreshes20TimesAcross50DecoderTicks) {
    sonic_replay::EncoderCacheKey cached;
    bool valid = false;
    int refreshes = 0;
    int decoder_ticks = 0;
    for (int tick = 0; tick < 50; ++tick) {
        const std::int64_t source_frame = tick * 20 / 50;
        const sonic_replay::EncoderCacheKey key{
            1, static_cast<std::uint64_t>(source_frame + 1), source_frame,
            0, true, 7};
        if (!valid || !(cached == key)) {
            cached = key;

            valid = true;
            ++refreshes;
        }
        ++decoder_ticks;
    }
    EXPECT_EQ(refreshes, 20);
    EXPECT_EQ(decoder_ticks, 50);
}

TEST(StreamedReplayStateTest, EncoderCacheKeyInvalidatesRequiredStateChanges) {
    const sonic_replay::EncoderCacheKey base{1, 2, 3, 0, true, 4};
    EXPECT_FALSE((base == sonic_replay::EncoderCacheKey{2, 2, 3, 0, true, 4}));
    EXPECT_FALSE((base == sonic_replay::EncoderCacheKey{1, 3, 3, 0, true, 4}));
    EXPECT_FALSE((base == sonic_replay::EncoderCacheKey{1, 2, 4, 0, true, 4}));
    EXPECT_FALSE((base == sonic_replay::EncoderCacheKey{1, 2, 3, 1, true, 4}));
    EXPECT_FALSE((base == sonic_replay::EncoderCacheKey{1, 2, 3, 0, false, 4}));
    EXPECT_FALSE((base == sonic_replay::EncoderCacheKey{1, 2, 3, 0, true, 5}));
}

TEST(StreamedReplayStateTest, RequiresExactlyContiguousSnapshotRows) {
    std::vector<std::int64_t> valid(20);
    for (std::size_t i = 0; i < valid.size(); ++i) {
        valid[i] = 100 + static_cast<std::int64_t>(i);
    }
    EXPECT_TRUE(StreamedReplayState::ValidateContiguousFrameIndices(valid, 20));

    auto gap = valid;
    gap[7] += 1;
    EXPECT_FALSE(StreamedReplayState::ValidateContiguousFrameIndices(gap, 20));
    EXPECT_FALSE(StreamedReplayState::ValidateContiguousFrameIndices(valid, 19));

    std::vector<std::int64_t> valid_50_hz(46);
    for (std::size_t i = 0; i < valid_50_hz.size(); ++i) {
        valid_50_hz[i] = 200 + static_cast<std::int64_t>(i);
    }
    EXPECT_TRUE(StreamedReplayState::ValidateContiguousFrameIndices(
        valid_50_hz, StreamedReplayState::ExpectedSnapshotRows(50.0)));
    EXPECT_FALSE(StreamedReplayState::ValidateContiguousFrameIndices(
        valid_50_hz, StreamedReplayState::ExpectedSnapshotRows(20.0)));
}

TEST(StreamedReplayStateTest, SourceFrameIsTheOnlyPacketClock) {
    StreamedReplayState state(20.0);

    EXPECT_EQ(state.Classify(10, false), PacketDecision::kNewEpoch);
    state.Commit(10, PacketDecision::kNewEpoch);
    auto info = state.Snapshot();
    EXPECT_TRUE(info.enabled);
    EXPECT_EQ(info.stream_epoch, 1U);
    EXPECT_EQ(info.generation, 1U);
    EXPECT_EQ(info.source_frame, 10);

    EXPECT_EQ(state.Classify(10, false), PacketDecision::kDuplicate);
    EXPECT_EQ(state.Classify(11, false), PacketDecision::kNext);
    EXPECT_EQ(state.Classify(9, false), PacketDecision::kStale);
    EXPECT_EQ(state.Classify(12, false), PacketDecision::kGap);

    state.Commit(11, PacketDecision::kNext);
    info = state.Snapshot();
    EXPECT_EQ(info.stream_epoch, 1U);
    EXPECT_EQ(info.generation, 2U);
    EXPECT_EQ(info.source_frame, 11);

    EXPECT_EQ(state.Classify(200, true), PacketDecision::kNewEpoch);
    state.Commit(200, PacketDecision::kNewEpoch);
    info = state.Snapshot();
    EXPECT_EQ(info.stream_epoch, 2U);
    EXPECT_EQ(info.generation, 3U);
    EXPECT_EQ(info.source_frame, 200);
}

TEST(StreamedReplayStateTest, ExplicitEpochMetadataPersistsUntilNextEpoch) {
    StreamedReplayState state(20.0);

    state.Commit(0, PacketDecision::kNewEpoch, false);
    EXPECT_FALSE(state.Snapshot().explicit_epoch);

    state.Commit(100, PacketDecision::kNewEpoch, true);
    EXPECT_TRUE(state.Snapshot().explicit_epoch);
    state.Commit(101, PacketDecision::kNext);
    EXPECT_TRUE(state.Snapshot().explicit_epoch);

    state.ResetForNewStream();
    auto info = state.Snapshot();
    EXPECT_FALSE(info.has_packet);
    EXPECT_FALSE(info.explicit_epoch);
    EXPECT_EQ(info.stream_epoch, 2U);

    state.Commit(200, PacketDecision::kNewEpoch, false);
    EXPECT_EQ(state.Snapshot().stream_epoch, 3U);
    state.Reset();
    EXPECT_EQ(state.Snapshot().stream_epoch, 0U);
}

TEST(StreamedReplayStateTest, SafetyRearmRequiresExplicitEpochAndSafeState) {
    sonic_replay::ReplaySafetyEpochGate gate;
    sonic_replay::ReplayStreamInfo info;
    info.has_packet = true;
    info.source_hz = 20.0;
    info.stream_epoch = 1;

    auto update = gate.Observe(info, true);
    EXPECT_FALSE(update.rearm_accepted);
    EXPECT_FALSE(update.reset_count_blocks_run);

    info.explicit_epoch = true;
    info.safety_reset_count = 1;
    update = gate.Observe(info, true);
    EXPECT_FALSE(update.rearm_accepted);
    EXPECT_TRUE(update.reset_count_blocks_run);

    info.safety_reset_count = 0;
    update = gate.Observe(info, false);
    EXPECT_FALSE(update.rearm_accepted);
    EXPECT_FALSE(update.reset_count_blocks_run);

    update = gate.Observe(info, true);
    EXPECT_FALSE(update.rearm_accepted);
    update = gate.Observe(info, true);
    EXPECT_TRUE(update.rearm_accepted);
    update = gate.Observe(info, true);
    EXPECT_FALSE(update.rearm_accepted);

    info.stream_epoch = 2;
    update = gate.Observe(info, true);
    EXPECT_FALSE(update.rearm_accepted);
    update = gate.Observe(info, true);
    EXPECT_TRUE(update.rearm_accepted);
}

TEST(StreamedReplayStateTest, SafetyResetCountIsDormantWithoutReplayPacket) {
    sonic_replay::ReplaySafetyEpochGate gate;
    sonic_replay::ReplayStreamInfo info;
    info.safety_reset_count = 3;

    const auto update = gate.Observe(info, true);
    EXPECT_FALSE(update.rearm_accepted);
    EXPECT_FALSE(update.reset_count_blocks_run);
}

TEST(StreamedReplayStateTest, DuplicateRequiresIdenticalPayloadBeforeAcknowledge) {
    EXPECT_EQ(
        StreamedReplayState::ApplyDuplicateContentCheck(
            PacketDecision::kDuplicate, true),
        PacketDecision::kDuplicate);
    EXPECT_EQ(
        StreamedReplayState::ApplyDuplicateContentCheck(
            PacketDecision::kDuplicate, false),
        PacketDecision::kInvalid);
    EXPECT_EQ(
        StreamedReplayState::ApplyDuplicateContentCheck(
            PacketDecision::kNext, false),
        PacketDecision::kNext);
}

TEST(StreamedReplayStateTest, WatchdogExpiresAfterConfiguredTimeout) {
    using namespace std::chrono_literals;
    StreamedReplayState state(20.0, 50.0, 250ms);
    const auto start = StreamedReplayState::Clock::time_point{1s};
    state.Commit(0, PacketDecision::kNewEpoch, start);

    EXPECT_FALSE(state.WatchdogExpired(start + 250ms));
    EXPECT_TRUE(state.WatchdogExpired(start + 251ms));

    state.AcknowledgeDuplicate(start + 200ms);
    EXPECT_FALSE(state.WatchdogExpired(start + 400ms));
    EXPECT_TRUE(state.WatchdogExpired(start + 451ms));
}

TEST(ReplayPayloadSafetyTest, ReplayTargetRequiresStrictBoolAndExactMatch) {
    using sonic_replay::ReplayArraySpec;
    using sonic_replay::ReplayPayloadSafety;

    const ReplayArraySpec valid{{1}, "bool", 1};
    std::string reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateReplayTarget(
        valid, 0, false, reason));
    EXPECT_TRUE(ReplayPayloadSafety::ValidateReplayTarget(
        valid, 1, true, reason));
    EXPECT_FALSE(ReplayPayloadSafety::ValidateReplayTarget(
        valid, 1, false, reason));
    EXPECT_NE(reason.find("mismatch"), std::string::npos);
    EXPECT_FALSE(ReplayPayloadSafety::ValidateReplayTarget(
        valid, 2, true, reason));

    for (const ReplayArraySpec malformed : {
             ReplayArraySpec{{}, "bool", 1},
             ReplayArraySpec{{1}, "u8", 1},
             ReplayArraySpec{{1}, "bool", 2}}) {
        EXPECT_FALSE(ReplayPayloadSafety::ValidateReplayTarget(
            malformed, 0, false, reason));
    }
}

TEST(ReplayPayloadSafetyTest, StrictV1SchemaRejectsShapeDtypeAndByteMismatches) {
    using sonic_replay::ReplayArraySpec;
    using sonic_replay::ReplayPayloadSafety;

    ReplayArraySpec q{{20, 29}, "f32", 20 * 29 * sizeof(float)};
    ReplayArraySpec dq = q;
    ReplayArraySpec quat{{20, 1, 4}, "f32", 20 * 4 * sizeof(float)};
    ReplayArraySpec frames{{20}, "i64", 20 * sizeof(std::int64_t)};
    std::string reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateV1CoreSchema(
        q, dq, quat, frames, 20, reason));

    auto malformed = q;
    malformed.shape = {20, 28};
    EXPECT_FALSE(ReplayPayloadSafety::ValidateV1CoreSchema(
        malformed, dq, quat, frames, 20, reason));

    malformed = q;
    malformed.dtype = "i32";
    EXPECT_FALSE(ReplayPayloadSafety::ValidateV1CoreSchema(
        malformed, dq, quat, frames, 20, reason));

    malformed = q;
    malformed.payload_bytes -= sizeof(float);
    EXPECT_FALSE(ReplayPayloadSafety::ValidateV1CoreSchema(
        malformed, dq, quat, frames, 20, reason));

    auto f64_dq = dq;
    f64_dq.dtype = "f64";
    f64_dq.payload_bytes = 20 * 29 * sizeof(double);
    EXPECT_FALSE(ReplayPayloadSafety::ValidateV1CoreSchema(
        q, f64_dq, quat, frames, 20, reason));

    auto malformed_quat = quat;
    malformed_quat.shape = {20, 4};
    EXPECT_FALSE(ReplayPayloadSafety::ValidateV1CoreSchema(
        q, dq, malformed_quat, frames, 20, reason));
}

TEST(ReplayPayloadSafetyTest, FiftyHertzExpectedRowsDriveEverySnapshotValidator) {
    using sonic_replay::ReplayArraySpec;
    using sonic_replay::ReplayPayloadSafety;

    const std::size_t rows = StreamedReplayState::ExpectedSnapshotRows(50.0);
    ASSERT_EQ(rows, 46U);
    const ReplayArraySpec q_spec{
        {rows, 29}, "f32", rows * 29 * sizeof(float)};
    const ReplayArraySpec quat_spec{
        {rows, 1, 4}, "f32", rows * 4 * sizeof(float)};
    const ReplayArraySpec frame_spec{
        {rows}, "i64", rows * sizeof(std::int64_t)};
    std::string reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateV1CoreSchema(
        q_spec, q_spec, quat_spec, frame_spec, rows, reason)) << reason;
    EXPECT_FALSE(ReplayPayloadSafety::ValidateV1CoreSchema(
        q_spec, q_spec, quat_spec, frame_spec, 20, reason));

    std::vector<std::int64_t> frames(rows);
    std::vector<std::vector<double>> q(
        rows, std::vector<double>(29, 0.0));
    std::vector<std::vector<double>> dq(
        rows, std::vector<double>(29, 0.0));
    std::vector<std::vector<std::array<double, 4>>> quat(
        rows, std::vector<std::array<double, 4>>(
                  1, {1.0, 0.0, 0.0, 0.0}));
    for (std::size_t row = 0; row < rows; ++row) {
        frames[row] = 1000 + static_cast<std::int64_t>(row);
    }

    EXPECT_TRUE(ReplayPayloadSafety::ValidateJointSnapshot(
        q, dq, rows, reason)) << reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateRawSnapshotTransitions(
        frames, q, quat, rows, reason)) << reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateRawOverlap(
        frames, q, dq, quat, frames, q, dq, quat, rows, reason)) << reason;

    EXPECT_FALSE(ReplayPayloadSafety::ValidateJointSnapshot(
        q, dq, 20, reason));
    EXPECT_FALSE(ReplayPayloadSafety::ValidateRawSnapshotTransitions(
        frames, q, quat, 20, reason));
    EXPECT_FALSE(ReplayPayloadSafety::ValidateRawOverlap(
        frames, q, dq, quat, frames, q, dq, quat, 20, reason));
}

TEST(ReplayPayloadSafetyTest, G1BoundsAcceptEpisodesAndRejectAbsoluteViolations) {
    using sonic_replay::ReplayPayloadSafety;
    std::vector<std::vector<double>> q(20, std::vector<double>(29, 0.0));
    std::vector<std::vector<double>> dq(20, std::vector<double>(29, 0.0));
    std::string reason;

    // Covers the observed episode-2 peak while remaining below every mapped
    // hardware velocity limit (the smallest is 20 rad/s).
    dq[3][10] = 4.72;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateJointSnapshot(q, dq, 20, reason));

    q[0][17] = sonic_replay::kG1ReplayJointPositionUpperLimits[17] + 0.01;
    EXPECT_FALSE(ReplayPayloadSafety::ValidateJointSnapshot(q, dq, 20, reason));
    q[0][17] = 0.0;

    dq[0][9] = sonic_replay::kG1ReplayJointVelocityLimits[9] + 0.01;
    EXPECT_FALSE(ReplayPayloadSafety::ValidateJointSnapshot(q, dq, 20, reason));
}

TEST(ReplayPayloadSafetyTest, MeasuredStateDetectsFallAndAbsoluteLimits) {
    std::array<double, sonic_replay::kG1ReplayJointCount> q{};
    std::array<double, sonic_replay::kG1ReplayJointCount> dq{};
    for (std::size_t joint = 0; joint < q.size(); ++joint) {
        q[joint] = 0.5 * (
            sonic_replay::kG1ReplayJointPositionLowerLimits[joint] +
            sonic_replay::kG1ReplayJointPositionUpperLimits[joint]);
    }
    std::array<double, 4> quat{1.0, 0.0, 0.0, 0.0};

    auto result = sonic_replay::ReplayPayloadSafety::EvaluateMeasuredState(
        q, dq, quat);
    EXPECT_TRUE(result.finite);
    EXPECT_FALSE(result.joint_position_violation);
    EXPECT_FALSE(result.joint_velocity_violation);
    EXPECT_FALSE(result.fall_detected);

    q[7] = sonic_replay::kG1ReplayJointPositionUpperLimits[7] + 0.01;
    result = sonic_replay::ReplayPayloadSafety::EvaluateMeasuredState(
        q, dq, quat);
    EXPECT_TRUE(result.joint_position_violation);
    q[7] = 0.0;

    dq[9] = -(sonic_replay::kG1ReplayJointVelocityLimits[9] + 0.01);
    result = sonic_replay::ReplayPayloadSafety::EvaluateMeasuredState(
        q, dq, quat);
    EXPECT_TRUE(result.joint_velocity_violation);
    dq[9] = 0.0;

    constexpr double roll = 0.8726646259971648;  // 50 degrees.
    quat = {std::cos(roll / 2.0), std::sin(roll / 2.0), 0.0, 0.0};
    result = sonic_replay::ReplayPayloadSafety::EvaluateMeasuredState(
        q, dq, quat);
    EXPECT_TRUE(result.fall_detected);

    q[0] = std::numeric_limits<double>::quiet_NaN();
    result = sonic_replay::ReplayPayloadSafety::EvaluateMeasuredState(
        q, dq, quat);
    EXPECT_FALSE(result.finite);
    EXPECT_TRUE(result.joint_position_violation);
}

TEST(ReplayPayloadSafetyTest, PolicyCommandUsesTrainingActionEnvelope) {
    using sonic_replay::ReplayPayloadSafety;
    EXPECT_TRUE(ReplayPayloadSafety::PolicyActionWithinEnvelope(0.0));
    EXPECT_TRUE(ReplayPayloadSafety::PolicyActionWithinEnvelope(20.0));
    EXPECT_TRUE(ReplayPayloadSafety::PolicyActionWithinEnvelope(-20.0));
    EXPECT_FALSE(ReplayPayloadSafety::PolicyActionWithinEnvelope(20.01));
    EXPECT_FALSE(ReplayPayloadSafety::PolicyActionWithinEnvelope(-20.01));
    EXPECT_FALSE(ReplayPayloadSafety::PolicyActionWithinEnvelope(
        std::numeric_limits<double>::infinity()));
}

TEST(StreamedReplayStateTest, DecoderOnlyV4RouteUsesPacketClock) {
    sonic_replay::ReplayStreamInfo info;
    EXPECT_FALSE(sonic_replay::IsDecoderOnlyV4Replay(info));

    info.enabled = true;
    info.has_packet = true;
    info.source_hz = 0.0;
    EXPECT_TRUE(sonic_replay::IsDecoderOnlyV4Replay(info));

    info.source_hz = 20.0;
    EXPECT_FALSE(sonic_replay::IsDecoderOnlyV4Replay(info));

    info.source_hz = 0.0;
    info.faulted = true;
    EXPECT_FALSE(sonic_replay::IsDecoderOnlyV4Replay(info));
}

TEST(ReplayPayloadSafetyTest, WujiBoundsMatchUrdfLimits) {
    WujiHandJointArray command = kWujiReplayCommandLowerLimits;
    std::string reason;
    EXPECT_TRUE(sonic_replay::ReplayPayloadSafety::ValidateWujiCommand(
        command, "hand", reason));

    command[1] = -0.1387;
    command[5] = -0.37;
    command[19] = 1.5735;
    EXPECT_TRUE(sonic_replay::ReplayPayloadSafety::ValidateWujiCommand(
        command, "hand", reason));

    command[0] = kWujiReplayCommandLowerLimits[0] - 0.001;
    EXPECT_FALSE(sonic_replay::ReplayPayloadSafety::ValidateWujiCommand(
        command, "hand", reason));
    command[0] = kWujiReplayCommandLowerLimits[0];
    command[19] = 1.58;
    EXPECT_FALSE(sonic_replay::ReplayPayloadSafety::ValidateWujiCommand(
        command, "hand", reason));
}

TEST(ReplayPayloadSafetyTest, AcceptedPacketsReplacePerSideHandValidity) {
    sonic_replay::ReplayHandValidity validity;
    EXPECT_FALSE(validity.Active());
    EXPECT_FALSE(validity.Valid(true));
    EXPECT_FALSE(validity.Valid(false));

    validity.ApplyAcceptedPacket(true, false);
    EXPECT_TRUE(validity.Active());
    EXPECT_TRUE(validity.Valid(true));
    EXPECT_FALSE(validity.Valid(false));

    validity.ApplyAcceptedPacket(false, true);
    EXPECT_FALSE(validity.Valid(true));
    EXPECT_TRUE(validity.Valid(false));

    validity.Reset();
    EXPECT_FALSE(validity.Active());
    EXPECT_FALSE(validity.Valid(true));
    EXPECT_FALSE(validity.Valid(false));
}

TEST(ReplayPayloadSafetyTest, RawTransitionsPreserveSoftRangeWithoutClampOrSlerp) {
    using sonic_replay::ReplayPayloadSafety;
    std::vector<std::int64_t> frames(20);
    std::vector<std::vector<double>> q(20, std::vector<double>(29, 0.0));
    std::vector<std::vector<std::array<double, 4>>> quat(
        20, std::vector<std::array<double, 4>>(1, {1.0, 0.0, 0.0, 0.0}));
    for (std::size_t row = 0; row < frames.size(); ++row) {
        frames[row] = 100 + static_cast<std::int64_t>(row);
    }

    // Episode 2 reaches about 0.266 rad between source rows: above the legacy
    // soft threshold (0.25), but below the hard replay threshold (0.8).
    q[19][0] = 0.266;
    quat[19][0] = {std::cos(0.25), 0.0, 0.0, std::sin(0.25)};
    const auto q_before = q;
    const auto quat_before = quat;
    std::string reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateRawSnapshotTransitions(
        frames, q, quat, 20, reason));
    EXPECT_EQ(q, q_before);
    EXPECT_EQ(quat, quat_before);

    q[19][0] = sonic_replay::kReplayJointPositionHardDelta + 0.01;
    EXPECT_FALSE(ReplayPayloadSafety::ValidateRawSnapshotTransitions(
        frames, q, quat, 20, reason));

    q = q_before;
    quat[19][0] = {std::cos(0.65), 0.0, 0.0, std::sin(0.65)};
    EXPECT_FALSE(ReplayPayloadSafety::ValidateRawSnapshotTransitions(
        frames, q, quat, 20, reason));
}

TEST(ReplayPayloadSafetyTest, GlobalOverlapChecksRowsWithoutTailToHeadComparison) {
    using sonic_replay::ReplayPayloadSafety;
    std::vector<std::int64_t> previous_frames(20);
    std::vector<std::int64_t> current_frames(20);
    std::vector<std::vector<double>> previous_q(
        20, std::vector<double>(29, 0.0));
    std::vector<std::vector<double>> current_q(
        20, std::vector<double>(29, 0.0));
    std::vector<std::vector<double>> previous_dq(
        20, std::vector<double>(29, 0.0));
    std::vector<std::vector<double>> current_dq(
        20, std::vector<double>(29, 0.0));
    std::vector<std::vector<std::array<double, 4>>> previous_quat(
        20, std::vector<std::array<double, 4>>(1, {1.0, 0.0, 0.0, 0.0}));
    auto current_quat = previous_quat;

    for (std::size_t row = 0; row < 20; ++row) {
        previous_frames[row] = 100 + static_cast<std::int64_t>(row);
        current_frames[row] = 101 + static_cast<std::int64_t>(row);
        previous_q[row][0] = 0.05 * static_cast<double>(row);
        if (row < 19) {
            current_q[row] = previous_q[row + 1];
            current_dq[row] = previous_dq[row + 1];
            current_quat[row] = previous_quat[row + 1];
        }
    }
    for (std::size_t row = 0; row < 19; ++row) {
        current_q[row] = previous_q[row + 1];
        current_dq[row] = previous_dq[row + 1];
        current_quat[row] = previous_quat[row + 1];
    }
    current_q[19] = previous_q[19];
    current_q[19][0] += 0.266;

    // previous tail (0.95) versus current head (0.05) exceeds 0.8, but those
    // are different, reverse-ordered global frames and must not be compared.
    std::string reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateRawSnapshotTransitions(
        current_frames, current_q, current_quat, 20, reason)) << reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateRawOverlap(
        current_frames, current_q, current_dq, current_quat,
        previous_frames, previous_q, previous_dq, previous_quat, 20, reason)) << reason;

    current_q[5][0] += 0.01;
    EXPECT_FALSE(ReplayPayloadSafety::ValidateRawOverlap(
        current_frames, current_q, current_dq, current_quat,
        previous_frames, previous_q, previous_dq, previous_quat, 20, reason));
}

TEST(ReplayPayloadSafetyTest, RejectsNonFiniteJointHandAndRawDelta) {
    using sonic_replay::ReplayPayloadSafety;
    std::vector<std::vector<double>> q(20, std::vector<double>(29, 0.0));
    std::vector<std::vector<double>> dq(20, std::vector<double>(29, 0.0));
    std::string reason;

    q[0][0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(ReplayPayloadSafety::ValidateJointSnapshot(q, dq, 20, reason));
    q[0][0] = 0.0;
    dq[0][0] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(ReplayPayloadSafety::ValidateJointSnapshot(q, dq, 20, reason));

    WujiHandJointArray hand = kDefaultWujiHandPose;
    hand[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(ReplayPayloadSafety::ValidateWujiCommand(
        hand, "hand", reason));
    hand[0] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(ReplayPayloadSafety::ValidateWujiCommand(
        hand, "hand", reason));

    std::vector<std::int64_t> frames(20);
    std::vector<std::vector<std::array<double, 4>>> quat(
        20, std::vector<std::array<double, 4>>(1, {1.0, 0.0, 0.0, 0.0}));
    for (std::size_t row = 0; row < frames.size(); ++row) {
        frames[row] = static_cast<std::int64_t>(row);
    }
    q[1][0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(ReplayPayloadSafety::ValidateRawSnapshotTransitions(
        frames, q, quat, 20, reason));
}


TEST(ReplayPayloadSafetyTest, ReplayHandFrameMustBeExactI64ScalarAndMatchSource) {
    using sonic_replay::ReplayArraySpec;
    using sonic_replay::ReplayPayloadSafety;

    const ReplayArraySpec valid{{1}, "i64", sizeof(std::int64_t)};
    std::string reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateReplayHandFrame(
        valid, 42, 42, reason));

    EXPECT_FALSE(ReplayPayloadSafety::ValidateReplayHandFrame(
        valid, 41, 42, reason));
    EXPECT_NE(reason.find("must equal frame_index"), std::string::npos);

    for (const ReplayArraySpec malformed : {
             ReplayArraySpec{{}, "i64", sizeof(std::int64_t)},
             ReplayArraySpec{{1}, "i32", sizeof(std::int32_t)},
             ReplayArraySpec{{1}, "i64", sizeof(std::int64_t) - 1},
             ReplayArraySpec{{2}, "i64", 2 * sizeof(std::int64_t)}}) {
        EXPECT_FALSE(ReplayPayloadSafety::ValidateReplayHandFrame(
            malformed, 42, 42, reason));
    }
}

TEST(ReplayHandSnapshotTest, SourceTransitionNeverMixesHandsValidityOrFrame) {
    DataBuffer<HandPoseSnapshot> latch;

    HandPoseSnapshot frame10;
    frame10.replay_latched = true;
    frame10.left_valid = true;
    frame10.right_valid = false;
    frame10.left.fill(10.0);
    frame10.right.fill(0.0);
    frame10.hand_frame_index = 10;
    latch.SetData(frame10);

    const auto held = latch.GetDataWithTime().data;
    ASSERT_NE(held, nullptr);

    HandPoseSnapshot frame11;
    frame11.replay_latched = true;
    frame11.left_valid = false;
    frame11.right_valid = true;
    frame11.left.fill(0.0);
    frame11.right.fill(11.0);
    frame11.hand_frame_index = 11;
    latch.SetData(frame11);

    const auto latest = latch.GetDataWithTime().data;
    ASSERT_NE(latest, nullptr);
    EXPECT_EQ(latest->hand_frame_index, 11);
    EXPECT_FALSE(latest->left_valid);
    EXPECT_TRUE(latest->right_valid);
    EXPECT_EQ(latest->left[0], 0.0);
    EXPECT_EQ(latest->right[0], 11.0);

    // A held immutable DataBuffer snapshot remains a complete old packet.
    EXPECT_EQ(held->hand_frame_index, 10);
    EXPECT_TRUE(held->left_valid);
    EXPECT_FALSE(held->right_valid);
    EXPECT_EQ(held->left[0], 10.0);
    EXPECT_EQ(held->right[0], 0.0);
}

TEST(ReplayPayloadSafetyTest, OfficialV1CoreSchemaStillHasNoHandFrameRequirement) {
    using sonic_replay::ReplayArraySpec;
    using sonic_replay::ReplayPayloadSafety;
    const ReplayArraySpec q{{20, 29}, "f32", 20 * 29 * sizeof(float)};
    const ReplayArraySpec quat{{20, 1, 4}, "f32", 20 * 4 * sizeof(float)};
    const ReplayArraySpec frames{{20}, "i64", 20 * sizeof(std::int64_t)};
    std::string reason;
    EXPECT_TRUE(ReplayPayloadSafety::ValidateV1CoreSchema(
        q, q, quat, frames, 20, reason));
}
TEST(StreamedMotionMergerTest, PreservesGlobalSourceIndices) {
    StreamedMotionMerger merger;
    StreamedMotionMerger::IncomingData incoming;
    incoming.protocol_version = 2;
    incoming.num_frames = 10;
    incoming.num_quat_bodies = 1;
    incoming.num_smpl_joints = 24;
    incoming.num_smpl_poses = 21;
    incoming.body_quat.resize(
        10, std::vector<std::array<double, 4>>(1, {1.0, 0.0, 0.0, 0.0}));
    incoming.smpl_joints.resize(
        10, std::vector<std::array<double, 3>>(24, {1.0, 2.0, 3.0}));
    incoming.smpl_pose.resize(
        10, std::vector<std::array<double, 3>>(21, {4.0, 5.0, 6.0}));
    for (std::int64_t i = 0; i < 10; ++i) {
        incoming.frame_indices.push_back(100 + 2 * i);
    }

    const auto result = merger.MergeIncomingData(incoming, 0);
    ASSERT_TRUE(result.motion);
    EXPECT_EQ(result.frame_step, 2);
    EXPECT_TRUE(result.motion->HasSourceProvenance());
    EXPECT_EQ(result.motion->SourceWindowStart(), 100);
    EXPECT_EQ(result.motion->SourceFrameStep(), 2);
    EXPECT_EQ(result.motion->SourceGenerationIndex(), 118);
    EXPECT_EQ(result.motion->SourceFrameIndex(0), 100);
    EXPECT_EQ(result.motion->SourceFrameIndex(9), 118);
}
}  // namespace
