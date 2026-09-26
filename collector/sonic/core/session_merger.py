from __future__ import annotations

from bisect import bisect_left
import json
from pathlib import Path
from typing import TypeVar

from .constants import DEFAULT_CAPTURE_FPS, frame_interval_ns_for_fps, half_delay_ns_for_fps, normalize_capture_fps
from .pose_pairing import pair_pose_streams
from .schema import HandStatusSample, PoseControlSample, PoseRawSample, PoseSample, RobotStateActionSample

T = TypeVar("T", PoseSample, RobotStateActionSample, HandStatusSample)

SAMPLE_MODE_NEAREST_HOLD = "nearest_hold"
SAMPLE_MODE_TICK_LATEST = "tick_latest"
SUPPORTED_SAMPLE_MODES = (SAMPLE_MODE_NEAREST_HOLD, SAMPLE_MODE_TICK_LATEST)


class SessionMerger:
    def __init__(
        self,
        episode_dir: Path,
        temp_dir: Path,
        *,
        capture_fps: int = DEFAULT_CAPTURE_FPS,
        sample_mode: str = SAMPLE_MODE_NEAREST_HOLD,
    ):
        self.episode_dir = episode_dir
        self.temp_dir = temp_dir
        self.capture_fps = normalize_capture_fps(capture_fps)
        if sample_mode not in SUPPORTED_SAMPLE_MODES:
            raise ValueError(f"unsupported sample mode: {sample_mode}")
        self.sample_mode = sample_mode
        self.delay = 1.0 / self.capture_fps
        self.frame_interval_ns = frame_interval_ns_for_fps(self.capture_fps)
        self.half_delay_ns = half_delay_ns_for_fps(self.capture_fps)

    def write(self, primary_camera: str) -> Path:
        data_path = self.episode_dir / "data.json"
        frames = self.merge(primary_camera=primary_camera)
        with data_path.open("w", encoding="utf-8") as handle:
            json.dump(frames, handle, indent=2, ensure_ascii=True)
        return data_path

    def merge(self, primary_camera: str) -> list[dict]:
        camera_ticks = self._read_jsonl("camera_ticks.jsonl")
        if self.sample_mode == SAMPLE_MODE_TICK_LATEST:
            return [self._tick_latest_frame(tick, primary_camera) for tick in camera_ticks]
        pose_controls = [self._pose_control_from_record(record) for record in self._read_jsonl("pose.jsonl")]
        pose_raws = [self._pose_raw_from_record(record) for record in self._read_jsonl("teleop_raw.jsonl")]
        robot_samples = [
            self._robot_state_action_from_record(record) for record in self._read_jsonl("robot_state_action.jsonl")
        ]
        hand_status_samples = [
            self._hand_status_from_record(record) for record in self._read_jsonl("hand_status.jsonl")
        ]

        pose_samples = pair_pose_streams(pose_controls, pose_raws)
        pose_timestamps = [sample.timestamp_ns for sample in pose_samples]
        robot_timestamps = [sample.timestamp_ns for sample in robot_samples]
        hand_timestamps = [sample.timestamp_ns for sample in hand_status_samples]
        hand_by_frame = {
            sample.source_frame_index: sample for sample in hand_status_samples if sample.source_frame_index >= 0
        }

        last_pose: PoseSample | None = None
        last_robot: RobotStateActionSample | None = None
        last_hand_status: HandStatusSample | None = None
        frames: list[dict] = []

        for tick in camera_ticks:
            frame_index = int(tick["frame_index"])
            tick_time_ns = int(tick["tick_time_ns"])

            pose_candidate = self._nearest_sample(pose_samples, pose_timestamps, tick_time_ns)
            if (
                pose_candidate is not None
                and abs(pose_candidate.timestamp_ns - tick_time_ns) <= self.half_delay_ns
            ):
                last_pose = pose_candidate

            robot_candidate = self._nearest_sample(robot_samples, robot_timestamps, tick_time_ns)
            if (
                robot_candidate is not None
                and abs(robot_candidate.timestamp_ns - tick_time_ns) <= self.half_delay_ns
            ):
                last_robot = robot_candidate

            hand_target_time_ns = last_pose.timestamp_ns if last_pose is not None else tick_time_ns
            hand_candidate = None
            if last_pose is not None:
                hand_candidate = hand_by_frame.get(last_pose.frame_index)
            if (
                hand_candidate is None
                or hand_candidate.timestamp_ns < 0
                or abs(hand_candidate.timestamp_ns - hand_target_time_ns) > self.half_delay_ns
            ):
                hand_candidate = self._nearest_sample(hand_status_samples, hand_timestamps, hand_target_time_ns)
            if (
                hand_candidate is not None
                and hand_candidate.timestamp_ns >= 0
                and abs(hand_candidate.timestamp_ns - hand_target_time_ns) <= self.half_delay_ns
            ):
                last_hand_status = hand_candidate

            frames.append(
                {
                    "frame_index": frame_index,
                    "time_ns": frame_index * self.frame_interval_ns,
                    "primary_camera": tick.get("primary_camera", primary_camera),
                    "cameras": self._finalize_camera_records(tick.get("cameras", {})),
                    "human_raw": last_pose.human_raw if last_pose is not None else {},
                    "human_derived": last_pose.human_derived if last_pose is not None else {},
                    "obs": last_robot.obs if last_robot is not None else {},
                    "action": last_robot.action if last_robot is not None else {},
                    "hand_feedback": last_hand_status.feedback if last_hand_status is not None else {},
                }
            )

        return frames

    def _tick_latest_frame(self, tick: dict, primary_camera: str) -> dict:
        latest = tick.get("latest_samples", {})
        if not isinstance(latest, dict):
            latest = {}
        frame_index = int(tick["frame_index"])
        return {
            "frame_index": frame_index,
            "time_ns": frame_index * self.frame_interval_ns,
            "tick_time_ns": int(tick["tick_time_ns"]),
            "state_index": latest.get("state_index", -1),
            "pose_frame_index": latest.get("pose_frame_index", -1),
            "pose_timestamp_ns": latest.get("pose_timestamp_ns", -1),
            "robot_timestamp_ns": latest.get("robot_timestamp_ns", -1),
            "smpl_age_ns": latest.get("smpl_age_ns", -1),
            "smpl_valid": bool(latest.get("smpl_valid", False)),
            "primary_camera": tick.get("primary_camera", primary_camera),
            "cameras": self._finalize_camera_records(tick.get("cameras", {})),
            "human_raw": latest.get("human_raw", {}),
            "human_derived": latest.get("human_derived", {}),
            "obs": latest.get("obs", {}),
            "action": latest.get("action", {}),
            "hand_feedback": latest.get("hand_feedback", {}),
        }

    def _read_jsonl(self, name: str) -> list[dict]:
        path = self.temp_dir / name
        if not path.exists():
            return []

        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _finalize_camera_records(self, cameras: dict[str, dict]) -> dict[str, dict]:
        finalized: dict[str, dict] = {}
        for key, record in cameras.items():
            finalized_record = {
                "color": record["color"],
                "time_ns": int(record["time_ns"]),
                "model": record["model"],
                "serial": record["serial"],
            }
            if "depth" in record:
                finalized_record["depth"] = record["depth"]
            if "sequence" in record:
                finalized_record["sequence"] = int(record["sequence"])
            if "camera_age_ns" in record:
                finalized_record["camera_age_ns"] = int(record["camera_age_ns"])
            if "reused" in record:
                finalized_record["reused"] = bool(record["reused"])
            if "stale" in record:
                finalized_record["stale"] = bool(record["stale"])
            if "source_time_ns" in record:
                finalized_record["source_time_ns"] = int(record["source_time_ns"])
            if "source_realtime_ns" in record:
                finalized_record["source_realtime_ns"] = int(record["source_realtime_ns"])
            if "transport" in record:
                finalized_record["transport"] = record["transport"]
            finalized[key] = finalized_record
        return finalized

    def _pose_control_from_record(self, record: dict) -> PoseControlSample:
        return PoseControlSample(
            timestamp_ns=int(record["timestamp_ns"]),
            frame_index=int(record["frame_index"]),
            human_derived=dict(record.get("human_derived", {})),
            payload=dict(record.get("payload", {})),
        )

    def _pose_raw_from_record(self, record: dict) -> PoseRawSample:
        return PoseRawSample(
            timestamp_ns=int(record["timestamp_ns"]),
            frame_index=int(record["frame_index"]),
            human_raw=dict(record.get("human_raw", {})),
            payload=dict(record.get("payload", {})),
        )

    def _robot_state_action_from_record(self, record: dict) -> RobotStateActionSample:
        return RobotStateActionSample(
            timestamp_ns=int(record["timestamp_ns"]),
            obs=dict(record.get("obs", {})),
            action=dict(record.get("action", {})),
            payload=dict(record.get("payload", {})),
        )

    def _hand_status_from_record(self, record: dict) -> HandStatusSample:
        return HandStatusSample(
            timestamp_ns=int(record.get("timestamp_ns", -1)),
            source_frame_index=int(record.get("source_frame_index", -1)),
            feedback=dict(record.get("feedback", {})),
            payload=dict(record.get("payload", {})),
        )

    def _nearest_sample(
        self,
        samples: list[T],
        timestamps: list[int],
        target_time_ns: int,
    ) -> T | None:
        if not samples:
            return None

        index = bisect_left(timestamps, target_time_ns)
        candidates: list[T] = []
        if index < len(samples):
            candidates.append(samples[index])
        if index > 0:
            candidates.append(samples[index - 1])
        if not candidates:
            return None
        return min(candidates, key=lambda sample: abs(sample.timestamp_ns - target_time_ns))
