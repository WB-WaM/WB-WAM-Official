/**
 * @file streamed_replay_state.hpp
 * @brief Packet-clock state for explicitly enabled low-rate v1 replay.
 */

#ifndef STREAMED_REPLAY_STATE_HPP
#define STREAMED_REPLAY_STATE_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <string>

namespace sonic_replay {

enum class PacketDecision {
    kDisabled,
    kNewEpoch,
    kNext,
    kDuplicate,
    kStale,
    kGap,
    kInvalid,
};

struct ReplayStreamInfo {
    bool enabled = false;
    bool has_packet = false;
    bool faulted = false;
    // True only when the current epoch was opened by an explicit catch_up
    // packet. Implicit first-packet epochs must not acknowledge old faults.
    bool explicit_epoch = false;
    double source_hz = 0.0;
    std::uint64_t stream_epoch = 0;
    std::uint64_t generation = 0;
    std::int64_t source_frame = -1;
    // Automatic input safety resets survive stream-buffer resets until a
    // validated explicit catch_up epoch acknowledges them.
    std::uint64_t safety_reset_count = 0;
    std::string fault_reason;
};

inline bool IsDecoderOnlyV4Replay(const ReplayStreamInfo& info) {
    return info.enabled && info.has_packet && !info.faulted &&
           info.source_hz == 0.0;
}

struct ReplaySafetyEpochUpdate {
    bool rearm_accepted = false;
    bool reset_count_blocks_run = false;
};

/**
 * Admit a new replay safety epoch only after its first measured state is safe.
 *
 * Endpoint resets may leave a historical reset count while planner/reference
 * control is active. That count is deliberately ignored until replay has an
 * accepted packet. A validated explicit catch_up epoch may then re-arm only if
 * the endpoint has acknowledged the reset (count == 0) and the robot's current
 * measured state passes every hard safety check.
 */
class ReplaySafetyEpochGate {
public:
    ReplaySafetyEpochUpdate Observe(
        const ReplayStreamInfo& info, bool measured_state_safe) {
        ReplaySafetyEpochUpdate update;
        if (!info.has_packet) {
            candidate_pending_ = false;
            return update;
        }

        update.reset_count_blocks_run = info.safety_reset_count > 0;
        const bool v1_route = info.source_hz > 0.0;
        const bool unseen_epoch =
            !has_epoch_ || stream_epoch_ != info.stream_epoch ||
            v1_route_ != v1_route;
        if (!info.explicit_epoch || !unseen_epoch || !measured_state_safe ||
            update.reset_count_blocks_run) {
            candidate_pending_ = false;
            return update;
        }

        // Require two consecutive safe control samples for the same identity.
        // Besides rejecting one-tick transients, this closes the tiny window
        // between the endpoint's atomic reset acknowledgement and state commit.
        if (candidate_pending_ && candidate_epoch_ == info.stream_epoch &&
            candidate_v1_route_ == v1_route) {
            has_epoch_ = true;
            stream_epoch_ = info.stream_epoch;
            v1_route_ = v1_route;
            candidate_pending_ = false;
            update.rearm_accepted = true;
        } else {
            candidate_pending_ = true;
            candidate_epoch_ = info.stream_epoch;
            candidate_v1_route_ = v1_route;
        }
        return update;
    }

private:
    bool has_epoch_ = false;
    bool v1_route_ = false;
    std::uint64_t stream_epoch_ = 0;
    bool candidate_pending_ = false;
    bool candidate_v1_route_ = false;
    std::uint64_t candidate_epoch_ = 0;
};

struct EncoderCacheKey {
    std::uint64_t stream_epoch = 0;
    std::uint64_t source_generation = 0;
    std::int64_t source_frame = -1;
    int encoder_mode = -1;
    bool play_state = false;
    std::int64_t heading_generation = 0;

    bool operator==(const EncoderCacheKey&) const = default;
};

/**
 * Replay mode is deliberately packet-driven: a newly accepted source frame
 * advances the source clock once; control-loop ticks between packets do not.
 */
class StreamedReplayState {
public:
    using Clock = std::chrono::steady_clock;

    explicit StreamedReplayState(
        double source_hz = 0.0,
        double control_hz = 50.0,
        std::chrono::milliseconds watchdog_timeout = std::chrono::milliseconds(250))
        : source_hz_(source_hz),
          control_hz_(control_hz),
          watchdog_timeout_(watchdog_timeout) {}

    bool Configured() const {
        return source_hz_ > 0.0 && control_hz_ > 0.0;
    }

    double SourceHz() const { return source_hz_; }
    double ControlHz() const { return control_hz_; }

    static bool IsSupportedSourceRate(double source_hz) {
        return std::abs(source_hz - 20.0) <= 1e-9 ||
               std::abs(source_hz - 50.0) <= 1e-9;
    }

    static bool ValidateFsqTokenState(
        std::span<const double> values,
        std::string& reason,
        std::size_t expected_dimension = 64,
        double tolerance = 1e-5) {
        if (values.size() != expected_dimension) {
            reason = "token_state dimension must be " +
                     std::to_string(expected_dimension);
            return false;
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &values[i], sizeof(bits));
            if ((bits & UINT64_C(0x7ff0000000000000)) ==
                UINT64_C(0x7ff0000000000000)) {
                reason = "token_state contains non-finite value at index " +
                         std::to_string(i);
                return false;
            }
            const double nearest_fsq = std::round(values[i] * 16.0) / 16.0;
            if (std::abs(values[i] - nearest_fsq) > tolerance) {
                reason = "token_state is off the 1/16 FSQ grid at index " +
                         std::to_string(i);
                return false;
            }
        }
        return true;
    }

    /** Convert a legacy control-rate observation offset to source rows. */
    static int ResolveObservationFrameOffset(
        int observation_index,
        int legacy_step,
        double source_hz,
        double control_hz = 50.0) {
        if (observation_index <= 0) {
            return 0;
        }
        if (source_hz <= 0.0 || control_hz <= 0.0) {
            return observation_index * legacy_step;
        }
        const double source_rows = static_cast<double>(observation_index * legacy_step) *
                                   source_hz / control_hz;
        return std::max(0, static_cast<int>(std::llround(source_rows)));
    }

    static int RequiredRows(
        int observation_count,
        int legacy_step,
        double source_hz,
        double control_hz = 50.0) {
        if (observation_count <= 0) {
            return 0;
        }
        return ResolveObservationFrameOffset(
                   observation_count - 1, legacy_step, source_hz, control_hz) +
               1;
    }

    /**
     * Number of rows carried by one explicit v1 replay snapshot.
     *
     * The established 20 Hz wire schema deliberately keeps its historical
     * 20 rows even though the 10-frame encoder look-ahead only consumes rows
     * through offset 18. A 50 Hz source needs the full legacy encoder window
     * through offset 45, hence 46 rows. Unsupported rates return zero so
     * callers can fail closed before decoding payload buffers.
     */
    static std::size_t ExpectedSnapshotRows(
        double source_hz,
        double control_hz = 50.0) {
        if (std::abs(source_hz - 20.0) <= 1e-9) {
            return 20;
        }
        if (std::abs(source_hz - 50.0) <= 1e-9) {
            return static_cast<std::size_t>(
                RequiredRows(10, 5, source_hz, control_hz));
        }
        return 0;
    }

    static bool ValidateContiguousFrameIndices(
        std::span<const std::int64_t> frame_indices,
        std::size_t expected_rows) {
        if (frame_indices.size() != expected_rows || frame_indices.empty()) {
            return false;
        }
        for (std::size_t i = 1; i < frame_indices.size(); ++i) {
            if (frame_indices[i] != frame_indices[i - 1] + 1) {
                return false;
            }
        }
        return true;
    }

    /** Convert a duplicate into a fatal invalid decision when its payload changed. */
    static PacketDecision ApplyDuplicateContentCheck(
        PacketDecision decision, bool payload_matches) {
        if (decision == PacketDecision::kDuplicate && !payload_matches) {
            return PacketDecision::kInvalid;
        }
        return decision;
    }

    PacketDecision Classify(std::int64_t head_frame, bool catch_up) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!Configured()) {
            return PacketDecision::kDisabled;
        }
        if (head_frame < 0) {
            return PacketDecision::kInvalid;
        }
        if (catch_up || !has_packet_) {
            return PacketDecision::kNewEpoch;
        }
        if (head_frame == source_frame_) {
            return PacketDecision::kDuplicate;
        }
        if (head_frame < source_frame_) {
            return PacketDecision::kStale;
        }
        if (head_frame == source_frame_ + 1) {
            return PacketDecision::kNext;
        }
        return PacketDecision::kGap;
    }

    void Commit(
        std::int64_t head_frame,
        PacketDecision decision,
        Clock::time_point now = Clock::now()) {
        Commit(head_frame, decision, false, now);
    }

    void Commit(
        std::int64_t head_frame,
        PacketDecision decision,
        bool explicit_catch_up,
        Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (decision != PacketDecision::kNewEpoch && decision != PacketDecision::kNext) {
            return;
        }
        if (decision == PacketDecision::kNewEpoch) {
            ++stream_epoch_;
            explicit_epoch_ = explicit_catch_up;
        }
        source_frame_ = head_frame;
        has_packet_ = true;
        faulted_ = false;
        fault_reason_.clear();
        ++generation_;
        last_accepted_time_ = now;
    }

    /** A duplicate is ignored as source data but still proves the sender is alive. */
    void AcknowledgeDuplicate(Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_packet_ && !faulted_) {
            last_accepted_time_ = now;
        }
    }

    void MarkFault(std::string reason) {
        std::lock_guard<std::mutex> lock(mutex_);
        faulted_ = true;
        fault_reason_ = std::move(reason);
    }

    bool WatchdogExpired(Clock::time_point now = Clock::now()) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return Configured() && has_packet_ && !faulted_ &&
               now - last_accepted_time_ > watchdog_timeout_;
    }

    ReplayStreamInfo Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ReplayStreamInfo info;
        info.enabled = Configured() && has_packet_ && !faulted_;
        info.has_packet = has_packet_;
        info.faulted = faulted_;
        info.explicit_epoch = explicit_epoch_;
        info.source_hz = source_hz_;
        info.stream_epoch = stream_epoch_;
        info.generation = generation_;
        info.source_frame = source_frame_;
        info.fault_reason = fault_reason_;
        return info;
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        ResetForNewStreamUnlocked();
        stream_epoch_ = 0;
    }

    /** Clear packet-local state while keeping a process-lifetime epoch serial. */
    void ResetForNewStream() {
        std::lock_guard<std::mutex> lock(mutex_);
        ResetForNewStreamUnlocked();
    }

private:
    void ResetForNewStreamUnlocked() {
        has_packet_ = false;
        faulted_ = false;
        explicit_epoch_ = false;
        source_frame_ = -1;
        generation_ = 0;
        fault_reason_.clear();
        last_accepted_time_ = Clock::time_point{};
    }
    double source_hz_ = 0.0;
    double control_hz_ = 50.0;
    std::chrono::milliseconds watchdog_timeout_{250};

    mutable std::mutex mutex_;
    bool has_packet_ = false;
    bool faulted_ = false;
    bool explicit_epoch_ = false;
    std::uint64_t stream_epoch_ = 0;
    std::uint64_t generation_ = 0;
    std::int64_t source_frame_ = -1;
    Clock::time_point last_accepted_time_{};
    std::string fault_reason_;
};

inline const char* PacketDecisionName(PacketDecision decision) {
    switch (decision) {
        case PacketDecision::kDisabled: return "disabled";
        case PacketDecision::kNewEpoch: return "new_epoch";
        case PacketDecision::kNext: return "next";
        case PacketDecision::kDuplicate: return "duplicate";
        case PacketDecision::kStale: return "stale";
        case PacketDecision::kGap: return "gap";
        case PacketDecision::kInvalid: return "invalid";
    }
    return "unknown";
}

}  // namespace sonic_replay

#endif  // STREAMED_REPLAY_STATE_HPP
