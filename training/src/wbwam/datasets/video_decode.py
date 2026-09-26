#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video decoding utilities used by WB datasets."""

import importlib
import logging
from pathlib import Path
import warnings

import torch
import torchvision


class VideoIntegrityError(RuntimeError):
    """Raised when video bytes and their source-frame metadata disagree."""


def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def _is_out_of_range_decode_error(err: Exception) -> bool:
    message = str(err).lower()
    return any(
        marker in message
        for marker in (
            "invalid frame index",
            "invalid pts in seconds",
            "out of bounds",
        )
    )


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to
            "torchcodec" when available; otherwise, defaults to "pyav".

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        try:
            return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
        except Exception as err:
            # Seeking beyond EOF with torchvision/PyAV can block a dataloader worker.
            # Let the dataset's missing-video policy handle known range errors instead.
            if _is_out_of_range_decode_error(err):
                raise

            warnings.warn(
                f"torchcodec video decode failed ({type(err).__name__}: {err}) on {video_path}; falling back to torchvision/pyav."  # noqa: E501
            )
            return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend="pyav")
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    elif backend == "decord":
        return decode_video_frames_decord(video_path, timestamps, tolerance_s)
    elif backend == "decord2":
        return decode_video_frames_decord2(video_path, timestamps, tolerance_s)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_by_indices_torchcodec(
    video_path: Path | str,
    frame_indices: list[int],
    device: str = "cpu",
    source_episode_id: str | None = None,
) -> torch.Tensor:
    """Decode exact sequential frame indices with TorchCodec."""
    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    indices = [int(index) for index in frame_indices]
    if not indices:
        raise ValueError("frame_indices must not be empty.")
    if min(indices) < 0:
        raise IndexError(f"Frame indices must be non-negative, got {indices}")

    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    num_frames = decoder.metadata.num_frames
    source_label = source_episode_id or str(video_path)
    if num_frames is None:
        raise VideoIntegrityError(f"Video frame count is unavailable for source={source_label}: {video_path}")
    num_frames = int(num_frames)
    if max(indices) >= num_frames:
        raise VideoIntegrityError(
            f"Frame index {max(indices)} is out of range for video with {int(num_frames)} frames: {video_path}"
        )

    frames = decoder.get_frames_at(indices=indices).data.type(torch.float32) / 255
    if len(frames) != len(indices):
        raise RuntimeError(
            f"TorchCodec returned {len(frames)} frames for {len(indices)} requested indices: {video_path}"
        )
    return frames


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # The closest key frame timestamp is usually smaller than `first_ts`
    # (for example, the key frame can be the first frame of the video).
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    # Use float32 for timestamp distance computation (torch.cdist doesn't support bfloat16).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."  # noqa: E501
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, such as in data
    loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    # initialize video decoder
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")

    # Header FPS can differ from the actual rate for variable-rate videos.
    # Decode by presentation timestamp instead of converting back to indices.
    frames_batch = decoder.get_frames_played_at(seconds=timestamps)

    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.as_tensor(frames_batch.pts_seconds, dtype=torch.float32, device="cpu").reshape(-1)
    durations = torch.as_tensor(
        frames_batch.duration_seconds,
        dtype=torch.float32,
        device="cpu",
    ).reshape(-1)
    if len(query_ts) != len(loaded_ts) or len(query_ts) != len(durations):
        raise RuntimeError(
            "TorchCodec returned mismatched frame metadata lengths: "
            f"queries={len(query_ts)} pts={len(loaded_ts)} durations={len(durations)}"
        )

    # get_frames_played_at returns the frame visible at each requested time.
    # Validate against the frame interval, not distance from its starting PTS.
    is_visible = (query_ts >= loaded_ts - tolerance_s) & (query_ts < loaded_ts + durations + tolerance_s)
    assert is_visible.all(), (
        "One or several query timestamps are outside their decoded frame intervals."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nframe durations: {durations}"
        f"\nvideo: {video_path}"
    )

    if log_loaded_timestamps:
        logging.info("Frame timestamps loaded at %s", loaded_ts)

    frames = frames_batch.data.type(torch.float32) / 255
    assert len(timestamps) == len(frames)
    return frames


def decode_video_frames_decord(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using decord.

    Requires the `decord` package to be installed.
    """
    try:
        import decord
    except ImportError:
        raise ImportError("decord is required but not installed. Install it with: pip install decord")

    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(str(video_path))
    avg_fps = vr.get_avg_fps()

    # Convert timestamps to frame indices
    frame_indices = [min(round(ts * avg_fps), len(vr) - 1) for ts in timestamps]

    # Retrieve frames as a batch (returns NHWC torch tensor via torch bridge)
    frames_batch = vr.get_batch(frame_indices)  # shape: (N, H, W, C)

    # Compute actual loaded timestamps
    loaded_ts = [idx / avg_fps for idx in frame_indices]

    if log_loaded_timestamps:
        for ts in loaded_ts:
            logging.info(f"Frame loaded at timestamp={ts:.4f}")

    # Tolerance check
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."  # noqa: E501
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # Convert from NHWC to NCHW, float32 in [0, 1]
    closest_frames = frames_batch.permute(0, 3, 1, 2).to(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_decord2(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using decord2.

    Requires the `decord2` package to be installed.
    """
    try:
        import decord2
    except ImportError:
        raise ImportError("decord2 is required but not installed. Install it with: pip install decord2")

    decord2.bridge.set_bridge("torch")

    vr = decord2.VideoReader(str(video_path))
    avg_fps = vr.get_avg_fps()

    # Convert timestamps to frame indices
    frame_indices = [min(round(ts * avg_fps), len(vr) - 1) for ts in timestamps]

    # Retrieve frames as a batch (returns NHWC torch tensor via torch bridge)
    frames_batch = vr.get_batch(frame_indices)  # shape: (N, H, W, C)

    # Compute actual loaded timestamps
    loaded_ts = [idx / avg_fps for idx in frame_indices]

    if log_loaded_timestamps:
        for ts in loaded_ts:
            logging.info(f"Frame loaded at timestamp={ts:.4f}")

    # Tolerance check
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."  # noqa: E501
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # Convert from NHWC to NCHW, float32 in [0, 1]
    closest_frames = frames_batch.permute(0, 3, 1, 2).to(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames
