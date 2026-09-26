#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "../input_interface/streamed_replay_state.hpp"
#include "../state_logger.hpp"

namespace sonic_replay {

class TerminalReplayPublishGate {
public:
    bool TryBegin() {
        int expected = kIdle;
        return state_.compare_exchange_strong(
            expected, kPublishing, std::memory_order_acq_rel);
    }

    void Complete() {
        state_.store(kPublished, std::memory_order_release);
    }

    void Retry() {
        int expected = kPublishing;
        (void)state_.compare_exchange_strong(
            expected, kIdle, std::memory_order_acq_rel);
    }

    bool Published() const {
        return state_.load(std::memory_order_acquire) == kPublished;
    }

private:
    static constexpr int kIdle = 0;
    static constexpr int kPublishing = 1;
    static constexpr int kPublished = 2;
    std::atomic<int> state_{kIdle};
};

inline bool MaterializeTerminalReplayFault(
    StateLogger::Entry& entry,
    const ReplayStreamInfo& replay_info,
    double last_pose_rx_age_ms) {
    if (!replay_info.faulted) {
        return false;
    }

    entry.has_replay_metadata = true;
    entry.replay_route = replay_info.source_hz > 0.0 ? 1 : 2;
    entry.replay_stream_epoch =
        static_cast<std::int64_t>(replay_info.stream_epoch);
    entry.replay_source_frame = replay_info.source_frame;
    entry.encoder_cache_refresh = false;
    entry.encoder_input_indices.fill(-1);
    if (entry.replay_route == 1) {
        for (std::size_t index = 0;
             index < entry.encoder_input_indices.size(); ++index) {
            entry.encoder_input_indices[index] = static_cast<std::int64_t>(
                StreamedReplayState::ResolveObservationFrameOffset(
                    static_cast<int>(index), 5, replay_info.source_hz));
        }
    }
    entry.last_pose_rx_age_ms = last_pose_rx_age_ms;
    entry.replay_fault_code = 1;
    entry.safety_telemetry_version = 1;
    entry.safety_reset_count = std::max(
        entry.safety_reset_count,
        static_cast<std::int64_t>(replay_info.safety_reset_count));
    return true;
}

}  // namespace sonic_replay
