from __future__ import annotations

from .schema import CameraFrame, EpisodeFrame, PoseSample, RobotStateActionSample


class FrameAligner:
    def __init__(self, max_sync_slop_ms: int = 100):
        self.max_sync_slop_ns = int(max_sync_slop_ms * 1e6)

    def align(
        self,
        *,
        primary_camera: str,
        expected_camera_keys: list[str],
        camera_frames: dict[str, CameraFrame],
        pose_sample: PoseSample | None,
        robot_sample: RobotStateActionSample | None,
    ) -> EpisodeFrame | None:
        primary_frame = camera_frames.get(primary_camera)
        if primary_frame is None or pose_sample is None or robot_sample is None:
            return None
        if not pose_sample.human_raw or not pose_sample.human_derived:
            return None

        reference_ts = primary_frame.timestamp_ns
        if abs(reference_ts - pose_sample.timestamp_ns) > self.max_sync_slop_ns:
            return None
        if abs(reference_ts - robot_sample.timestamp_ns) > self.max_sync_slop_ns:
            return None

        aligned_frames: dict[str, CameraFrame] = {}
        for key in expected_camera_keys:
            frame = camera_frames.get(key)
            if frame is None:
                return None
            if abs(frame.timestamp_ns - reference_ts) > self.max_sync_slop_ns:
                return None
            aligned_frames[key] = frame

        return EpisodeFrame(
            time_ns=reference_ts,
            primary_camera=primary_camera,
            cameras=aligned_frames,
            human_raw=pose_sample.human_raw,
            human_derived=pose_sample.human_derived,
            obs=robot_sample.obs,
            action=robot_sample.action,
        )
