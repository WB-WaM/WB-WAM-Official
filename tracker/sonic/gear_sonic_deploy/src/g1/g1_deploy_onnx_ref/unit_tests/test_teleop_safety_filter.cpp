#include <gtest/gtest.h>

#include "../include/input_interface/teleop_safety_filter.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

struct PosePacket {
    std::vector<std::vector<double>> joint_pos;
    std::vector<std::vector<std::array<double, 4>>> body_quat;
    std::vector<std::vector<std::array<double, 3>>> smpl_joints;
    std::vector<std::vector<std::array<double, 3>>> smpl_pose;
    WujiHandJointArray left_hand = kDefaultWujiHandPose;
    WujiHandJointArray right_hand = kDefaultWujiHandPose;
    std::array<double, 9> vr_position{};
    std::array<double, 12> vr_orientation{};
};

PosePacket MakePacket() {
    PosePacket packet;
    packet.joint_pos = {{0.0, 0.0}};
    packet.body_quat = {{{{1.0, 0.0, 0.0, 0.0}}}};
    packet.smpl_joints = {{{{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}}}};
    packet.smpl_pose = {{{{0.0, 0.0, 0.0}, {0.0, 0.1, 0.0}}}};
    packet.vr_orientation = {
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    };
    return packet;
}

TeleopSafetyFilter::Result ApplyFilter(TeleopSafetyFilter& filter, PosePacket& packet) {
    return filter.FilterDecodedPose(
        packet.joint_pos,
        true,
        packet.body_quat,
        packet.smpl_joints,
        true,
        packet.smpl_pose,
        true,
        true,
        packet.left_hand,
        true,
        packet.right_hand,
        true,
        packet.vr_position,
        true,
        packet.vr_orientation);
}

std::array<double, 4> RotationAroundX(double radians) {
    return {std::cos(radians / 2.0), std::sin(radians / 2.0), 0.0, 0.0};
}

class TeleopSafetyFilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        TeleopSafetyFilter::SetGlobalEnabled(true);
    }
};

}  // namespace

TEST_F(TeleopSafetyFilterTest, AcceptsNormalContinuousInput) {
    TeleopSafetyFilter filter;
    auto first = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first).accepted);

    auto second = MakePacket();
    second.joint_pos[0][0] = 0.05;
    second.smpl_joints[0][0][0] = 0.02;
    second.smpl_pose[0][0][1] = 0.02;
    second.left_hand[0] = 0.02;
    second.vr_position[0] = 0.01;

    const auto result = ApplyFilter(filter, second);
    EXPECT_TRUE(result.accepted);
    EXPECT_FALSE(result.clamped);
    EXPECT_FALSE(result.reset_requested);
}

TEST_F(TeleopSafetyFilterTest, RejectsNonFiniteInput) {
    TeleopSafetyFilter filter;
    auto first = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first).accepted);

    auto bad = MakePacket();
    bad.joint_pos[0][0] = std::numeric_limits<double>::quiet_NaN();

    const auto result = ApplyFilter(filter, bad);
    EXPECT_FALSE(result.accepted);
    EXPECT_FALSE(result.reset_requested);
    EXPECT_NE(result.reason.find("non-finite"), std::string::npos);
}

TEST_F(TeleopSafetyFilterTest, RejectsHardJointJump) {
    TeleopSafetyFilter filter;
    auto first = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first).accepted);

    auto jump = MakePacket();
    jump.joint_pos[0][0] = 0.81;

    const auto result = ApplyFilter(filter, jump);
    EXPECT_FALSE(result.accepted);
    EXPECT_NE(result.reason.find("joint_pos"), std::string::npos);
}

TEST_F(TeleopSafetyFilterTest, ClampsSoftJointJump) {
    TeleopSafetyFilter filter;
    auto first = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first).accepted);

    auto jump = MakePacket();
    jump.joint_pos[0][0] = 0.30;

    const auto result = ApplyFilter(filter, jump);
    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.clamped);
    EXPECT_NEAR(jump.joint_pos[0][0], 0.25, 1e-9);
}

TEST_F(TeleopSafetyFilterTest, ConsecutiveHardRejectsRequestReset) {
    TeleopSafetyFilter filter;
    auto first = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first).accepted);

    TeleopSafetyFilter::Result result;
    for (int i = 0; i < 5; ++i) {
        auto jump = MakePacket();
        jump.joint_pos[0][0] = 1.0;
        result = ApplyFilter(filter, jump);
        EXPECT_FALSE(result.accepted);
    }

    EXPECT_TRUE(result.reset_requested);
    EXPECT_EQ(result.consecutive_rejects, 5);
}

TEST_F(TeleopSafetyFilterTest, RejectsHandAndVRPositionJumps) {
    TeleopSafetyFilter filter;
    auto first = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first).accepted);

    auto hand_jump = MakePacket();
    hand_jump.left_hand[0] = 0.51;
    const auto hand_result = ApplyFilter(filter, hand_jump);
    EXPECT_FALSE(hand_result.accepted);
    EXPECT_NE(hand_result.reason.find("left_wuji_qpos"), std::string::npos);

    filter.Reset();
    auto first_again = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first_again).accepted);

    auto vr_jump = MakePacket();
    vr_jump.vr_position[0] = 0.26;
    const auto vr_result = ApplyFilter(filter, vr_jump);
    EXPECT_FALSE(vr_result.accepted);
    EXPECT_NE(vr_result.reason.find("vr_position"), std::string::npos);
}

TEST_F(TeleopSafetyFilterTest, RejectsQuaternionJumps) {
    TeleopSafetyFilter filter;
    auto first = MakePacket();
    EXPECT_TRUE(ApplyFilter(filter, first).accepted);

    auto jump = MakePacket();
    jump.vr_orientation[0] = RotationAroundX(1.21)[0];
    jump.vr_orientation[1] = RotationAroundX(1.21)[1];

    const auto result = ApplyFilter(filter, jump);
    EXPECT_FALSE(result.accepted);
    EXPECT_NE(result.reason.find("vr_orientation"), std::string::npos);
}
