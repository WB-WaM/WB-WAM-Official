#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "wuji_hand_constants.hpp"
#include <wujihandcpp/data/joint.hpp>
#include <wujihandcpp/device/controller.hpp>
#include <wujihandcpp/device/hand.hpp>
#include <wujihandcpp/filter/low_pass.hpp>

class WujiHands {
public:
    bool initialize(const std::string& left_serial, const std::string& right_serial) {
        const bool left_ok = initializeHand(left_hand_, left_serial, "left");
        const bool right_ok = initializeHand(right_hand_, right_serial, "right");
        initialized_ = left_ok || right_ok;
        if (!initialized_) {
            std::cerr << "[WujiHands] No Wuji hands available, continuing without hand hardware"
                      << std::endl;
        }
        return initialized_;
    }

    void setAllJointsCommand(bool is_left, const WujiHandJointArray& command) {
        HandContext& hand = is_left ? left_hand_ : right_hand_;
        if (!hand.initialized || !hand.controller) {
            return;
        }
        std::lock_guard<std::mutex> lock(hand.mutex);

        hand.target = command;
        double positions[kWujiFingerCount][kWujiJointsPerFinger];
        toMatrix(command, positions);
        hand.controller->set_joint_target_position(positions);
    }

    void setOpenPose(bool is_left) {
        setAllJointsCommand(is_left, kDefaultWujiHandPose);
    }

    WujiHandJointArray getJointActualPosition(bool is_left) const {
        const HandContext& hand = is_left ? left_hand_ : right_hand_;
        if (!hand.initialized || !hand.device) {
            return kDefaultWujiHandPose;
        }
        std::lock_guard<std::mutex> lock(hand.mutex);

        return fromAtomicMatrix(hand.device->realtime_get_joint_actual_position());
    }

    void writeOnce() const {
        // Wuji realtime controllers stream from the SDK background thread.
    }

    bool isInitialized() const {
        return initialized_;
    }

    void shutdown() {
        shutdownHand(left_hand_, "left");
        shutdownHand(right_hand_, "right");
        initialized_ = false;
    }

private:
    struct HandContext {
        std::unique_ptr<wujihandcpp::device::Hand> device;
        std::unique_ptr<wujihandcpp::device::IController> controller;
        WujiHandJointArray target = kDefaultWujiHandPose;
        bool initialized = false;
        mutable std::mutex mutex;
    };

    static constexpr double kCutoffFreqHz = 2.0;

    static bool initializeHand(
        HandContext& hand, const std::string& serial, const char* side_label) {
        if (serial.empty()) {
            std::cerr << "[WujiHands] Missing serial for " << side_label
                      << " hand, skipping initialization" << std::endl;
            hand.initialized = false;
            return false;
        }

        try {
            hand.device = std::make_unique<wujihandcpp::device::Hand>(serial.c_str());
            hand.device->disable_thread_safe_check();
            hand.device->write<wujihandcpp::data::joint::Enabled>(true);
            hand.controller = hand.device->realtime_controller<false>(
                wujihandcpp::filter::LowPass(kCutoffFreqHz));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            hand.initialized = true;

            std::cout << "[WujiHands] Initialized " << side_label
                      << " hand with serial " << serial << std::endl;
            return true;
        } catch (const std::exception& exc) {
            hand.controller.reset();
            hand.device.reset();
            hand.target = kDefaultWujiHandPose;
            hand.initialized = false;
            std::cerr << "[WujiHands] Failed to initialize " << side_label
                      << " hand (" << serial << "), skipping it: " << exc.what() << std::endl;
            return false;
        }
    }

    static void shutdownHand(
        HandContext& hand, const char* side_label) {
        std::lock_guard<std::mutex> lock(hand.mutex);
        if (!hand.device) {
            hand.initialized = false;
            return;
        }

        hand.controller.reset();
        try {
            hand.device->write<wujihandcpp::data::joint::Enabled>(false);
            std::cout << "[WujiHands] Disabled " << side_label << " hand" << std::endl;
        } catch (const std::exception& exc) {
            std::cerr << "[WujiHands] Failed to disable " << side_label
                      << " hand: " << exc.what() << std::endl;
        }
        hand.device.reset();
        hand.target = kDefaultWujiHandPose;
        hand.initialized = false;
    }

    static void toMatrix(
        const WujiHandJointArray& flat,
        double (&matrix)[kWujiFingerCount][kWujiJointsPerFinger]) {
        for (std::size_t finger = 0; finger < kWujiFingerCount; ++finger) {
            for (std::size_t joint = 0; joint < kWujiJointsPerFinger; ++joint) {
                matrix[finger][joint] = flat[finger * kWujiJointsPerFinger + joint];
            }
        }
    }

    static WujiHandJointArray fromAtomicMatrix(
        const std::atomic<double> (&matrix)[kWujiFingerCount][kWujiJointsPerFinger]) {
        WujiHandJointArray flat = kDefaultWujiHandPose;
        for (std::size_t finger = 0; finger < kWujiFingerCount; ++finger) {
            for (std::size_t joint = 0; joint < kWujiJointsPerFinger; ++joint) {
                flat[finger * kWujiJointsPerFinger + joint] =
                    matrix[finger][joint].load(std::memory_order_relaxed);
            }
        }
        return flat;
    }

    HandContext left_hand_;
    HandContext right_hand_;
    bool initialized_ = false;
};
