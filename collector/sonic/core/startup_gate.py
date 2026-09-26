from __future__ import annotations

from dataclasses import dataclass

from .constants import DEFAULT_CAPTURE_FPS, half_delay_ns_for_fps, normalize_capture_fps
from .pose_pairing import IncrementalPosePairer, PairedPoseSample
from .schema import DrainedSamples, RobotStateActionSample


@dataclass
class StartupGateStatus:
    pose_pair: PairedPoseSample | None = None
    robot_sample: RobotStateActionSample | None = None


class StartupGate:
    def __init__(self, *, capture_fps: int = DEFAULT_CAPTURE_FPS) -> None:
        self.capture_fps = normalize_capture_fps(capture_fps)
        self.half_delay_ns = half_delay_ns_for_fps(self.capture_fps)
        self._pose_pairer = IncrementalPosePairer()
        self._status = StartupGateStatus()

    def reset(self) -> None:
        self._pose_pairer.reset()
        self._status = StartupGateStatus()

    def ingest(self, drained: DrainedSamples) -> None:
        pairs = self._pose_pairer.push_samples(drained.pose_controls, drained.pose_raws)
        for pair in pairs:
            if pair.sample.human_raw and pair.sample.human_derived:
                self._status.pose_pair = pair

        for sample in drained.state_actions:
            if sample.obs:
                self._status.robot_sample = sample

    def ready(
        self,
        *,
        camera_snapshot_count: int,
        expected_camera_count: int,
        reference_time_ns: int,
    ) -> bool:
        if camera_snapshot_count != expected_camera_count:
            return False
        if self._status.pose_pair is None or self._status.robot_sample is None:
            return False
        if abs(reference_time_ns - self._status.pose_pair.sample.timestamp_ns) > self.half_delay_ns:
            return False
        if abs(reference_time_ns - self._status.robot_sample.timestamp_ns) > self.half_delay_ns:
            return False
        return True

    def build_start_samples(self) -> DrainedSamples:
        if self._status.pose_pair is None or self._status.robot_sample is None:
            return DrainedSamples(pose_controls=[], pose_raws=[], state_actions=[])

        return DrainedSamples(
            pose_controls=[self._status.pose_pair.control],
            pose_raws=[self._status.pose_pair.raw],
            state_actions=[self._status.robot_sample],
        )
