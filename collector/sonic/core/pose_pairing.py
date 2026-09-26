from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .schema import PoseControlSample, PoseRawSample, PoseSample


@dataclass(frozen=True)
class PairedPoseSample:
    control: PoseControlSample
    raw: PoseRawSample
    sample: PoseSample


def build_paired_pose_sample(control: PoseControlSample, raw: PoseRawSample) -> PairedPoseSample:
    return PairedPoseSample(
        control=control,
        raw=raw,
        sample=PoseSample(
            timestamp_ns=max(control.timestamp_ns, raw.timestamp_ns),
            frame_index=control.frame_index,
            human_raw=raw.human_raw,
            human_derived=control.human_derived,
            payload={"pose": control.payload, "teleop_raw": raw.payload},
        ),
    )


def pair_pose_streams(
    controls: list[PoseControlSample],
    raws: list[PoseRawSample],
) -> list[PoseSample]:
    controls_by_frame: dict[int, deque[PoseControlSample]] = defaultdict(deque)
    raws_by_frame: dict[int, deque[PoseRawSample]] = defaultdict(deque)

    for sample in controls:
        controls_by_frame[sample.frame_index].append(sample)
    for sample in raws:
        raws_by_frame[sample.frame_index].append(sample)

    pairs: list[PoseSample] = []
    for frame_index in sorted(set(controls_by_frame.keys()) & set(raws_by_frame.keys())):
        control_queue = controls_by_frame[frame_index]
        raw_queue = raws_by_frame[frame_index]
        while control_queue and raw_queue:
            pairs.append(build_paired_pose_sample(control_queue.popleft(), raw_queue.popleft()).sample)

    pairs.sort(key=lambda sample: sample.timestamp_ns)
    return pairs


class IncrementalPosePairer:
    def __init__(self) -> None:
        self._controls_by_frame: dict[int, deque[PoseControlSample]] = defaultdict(deque)
        self._raws_by_frame: dict[int, deque[PoseRawSample]] = defaultdict(deque)

    def reset(self) -> None:
        self._controls_by_frame.clear()
        self._raws_by_frame.clear()

    def push_samples(
        self,
        controls: list[PoseControlSample],
        raws: list[PoseRawSample],
    ) -> list[PairedPoseSample]:
        touched_frames: set[int] = set()
        for sample in controls:
            self._controls_by_frame[sample.frame_index].append(sample)
            touched_frames.add(sample.frame_index)
        for sample in raws:
            self._raws_by_frame[sample.frame_index].append(sample)
            touched_frames.add(sample.frame_index)

        pairs: list[PairedPoseSample] = []
        for frame_index in sorted(touched_frames):
            control_queue = self._controls_by_frame.get(frame_index)
            raw_queue = self._raws_by_frame.get(frame_index)
            if not control_queue or not raw_queue:
                continue
            while control_queue and raw_queue:
                pairs.append(build_paired_pose_sample(control_queue.popleft(), raw_queue.popleft()))
            if not control_queue:
                self._controls_by_frame.pop(frame_index, None)
            if not raw_queue:
                self._raws_by_frame.pop(frame_index, None)
        return pairs
