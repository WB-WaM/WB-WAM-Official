import argparse, json, os, subprocess, sys, tempfile


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def ffprobe_json(path: str) -> dict:
    code, out, err = run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-print_format", "json", path
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{err}")
    return json.loads(out)


def get_wh(path: str) -> tuple[int, int]:
    info = ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s["width"]), int(s["height"])
    raise RuntimeError(f"No video stream in {path}")


def has_audio(path: str) -> bool:
    info = ffprobe_json(path)
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def get_duration_seconds(path: str) -> float:
    info = ffprobe_json(path)
    # Prefer container duration
    fmt = info.get("format", {})
    dur = fmt.get("duration")
    if dur is not None:
        try:
            return float(dur)
        except Exception:
            pass
    # Fallback: max stream duration
    max_stream_dur = 0.0
    for s in info.get("streams", []):
        sd = s.get("duration")
        if sd is not None:
            try:
                max_stream_dur = max(max_stream_dur, float(sd))
            except Exception:
                continue
    return max_stream_dur


def get_fps(path: str) -> float:
    """Get video frame rate (fps)."""
    info = ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            fps_str = s.get("r_frame_rate") or s.get("avg_frame_rate")
            if fps_str:
                try:
                    num, den = map(int, fps_str.split("/"))
                    if den > 0:
                        return float(num) / den
                except Exception:
                    pass
    # Default to 30 if not found
    return 30.0


def merge_videos(video_paths: list[str], out: str, speed: float = 1.0):
    """
    Merge multiple videos based on count:
    - 2 videos: left-right (hstack)
    - 3 videos: left-center-right (hstack)
    - 4 videos: 2x2 grid (xstack)
    - 6 videos: 3x2 grid (xstack)
    
    Args:
        video_paths: List of input video file paths
        out: Output video file path
        speed: Playback speed multiplier (e.g., 2.0 for 2x speed, 0.5 for 0.5x speed)
    """
    num_videos = len(video_paths)
    
    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    
    if num_videos not in [2, 3, 4, 6]:
        raise ValueError(f"Unsupported number of videos: {num_videos}. Supported: 2, 3, 4, 6")
    
    # Default parameters
    height = None
    audio = "left"
    vcodec = "libx264"
    crf = 20
    preset = "veryfast"
    shortest = False
    output_fps = 30.0
    
    # Calculate target height: use minimum height of all videos
    heights = [get_wh(p)[1] for p in video_paths]
    height = min(heights)
    
    # Get filenames for text overlay
    video_names = [os.path.basename(p) for p in video_paths]
    
    # Build video filters: scale, speed adjustment, and draw text for each video
    vf_filters = []
    for i, (path, name) in enumerate(zip(video_paths, video_names)):
        # Apply speed adjustment using setpts
        # setpts=PTS/speed: speed>1 makes faster (less time per frame), speed<1 makes slower (more time per frame)
        speed_filter = f"setpts=PTS/{speed:.6f}"
        vf = f"[{i}:v]scale=-2:{height}:flags=lanczos,setsar=1,{speed_filter}," \
             f"drawtext=text='{name}':x=10:y=10:fontsize=24:" \
             f"fontcolor=white:box=1:boxcolor=black@0.5[s{i}]"
        vf_filters.append(vf)
    
    # Build layout based on number of videos
    if num_videos == 2:
        # Left-right: hstack
        layout_filter = "[s0][s1]hstack=inputs=2[vout]"
    elif num_videos == 3:
        # Left-center-right: hstack with 3 inputs
        layout_filter = "[s0][s1][s2]hstack=inputs=3[vout]"
    elif num_videos == 4:
        # 2x2 grid: xstack
        # Layout: 0_0 (top-left), w0_0 (top-right), 0_h0 (bottom-left), w0_h0 (bottom-right)
        layout_filter = "[s0][s1][s2][s3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[vout]"
    elif num_videos == 6:
        # 3x2 grid: xstack
        # Layout: 0_0, w0_0, w0+w1_0 (top row)
        #        0_h0, w0_h0, w0+w1_h0 (bottom row)
        # For 3x2, we need to calculate positions: each video has width w0, height h0
        # Top row: 0_0, w0_0, w0+w1_0 (since all videos have same width after scaling)
        # Bottom row: 0_h0, w0_h0, w0+w1_h0
        # Use w0+w1 instead of w0*2 for better compatibility
        layout_filter = "[s0][s1][s2][s3][s4][s5]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0:fill=black[vout]"
    
    # Audio strategy: use first video's audio if available
    # Apply speed adjustment to audio using atempo (range: 0.5 to 2.0)
    map_args = []
    map_audio = False
    audio_filters = []
    if audio == "left" and has_audio(video_paths[0]):
        # atempo can only handle 0.5-2.0 range, so chain multiple if needed
        if speed != 1.0:
            remaining_speed = speed
            atempo_chain = []
            while remaining_speed > 2.0:
                atempo_chain.append("atempo=2.0")
                remaining_speed /= 2.0
            while remaining_speed < 0.5:
                atempo_chain.append("atempo=0.5")
                remaining_speed /= 0.5
            if abs(remaining_speed - 1.0) > 1e-6:
                atempo_chain.append(f"atempo={remaining_speed:.6f}")
            
            if atempo_chain:
                audio_filter_str = ",".join(atempo_chain)
                audio_filters.append(f"[0:a]{audio_filter_str}[aout]")
                map_args += ["-map", "[aout]"]
            else:
                map_args += ["-map", "0:a"]
        else:
            map_args += ["-map", "0:a"]
        map_audio = True
    
    # Calculate duration alignment: use minimum duration and center crop
    durations = [get_duration_seconds(p) for p in video_paths]
    target_duration = None
    seek_times = [0.0] * num_videos
    
    if all(d > 0 for d in durations):
        target_duration = min(durations)
        for i, d in enumerate(durations):
            if d > target_duration + 1e-3:
                seek_times[i] = max(0.0, (d - target_duration) / 2.0)
    
    def build_cmd():
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        ]
        
        # Add inputs with seek and duration if needed
        for i, path in enumerate(video_paths):
            if seek_times[i] > 0:
                cmd += ["-ss", f"{seek_times[i]:.3f}"]
            if target_duration is not None:
                # Adjust duration for speed: setpts=PTS/speed changes timestamps
                adjusted_duration = target_duration / speed
                cmd += ["-t", f"{adjusted_duration:.3f}"]
            cmd += ["-i", path]
        
        # Combine video and audio filters
        all_filters = vf_filters + audio_filters + [layout_filter]
        filter_complex_full = ";".join(all_filters)
        
        # Processing
        cmd += [
            "-filter_complex", filter_complex_full,
            "-map", "[vout]", *map_args,
            "-c:v", vcodec, "-crf", str(crf), "-preset", preset,
            "-r", str(output_fps),  # Set output frame rate to 30fps
            "-movflags", "+faststart",
        ]
        
        if map_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        
        if shortest:
            cmd += ["-shortest"]
        
        cmd += [out]
        return cmd
    
    # Try hstack first for 2 and 3 videos
    if num_videos in [2, 3]:
        code, outlog, errlog = run(build_cmd())
        if code == 0:
            return
        
        # Fallback to xstack if hstack fails
        if num_videos == 2:
            layout_filter = "[s0][s1]xstack=inputs=2:layout=0_0|w0_0:fill=black[vout]"
        else:  # num_videos == 3
            layout_filter = "[s0][s1][s2]xstack=inputs=3:layout=0_0|w0_0|w0+w1_0:fill=black[vout]"
        code, outlog, errlog = run(build_cmd())
        if code == 0:
            return
    else:
        # For 4 and 6 videos, use xstack directly
        code, outlog, errlog = run(build_cmd())
        if code == 0:
            return
    
    # If all methods failed, raise error
    raise RuntimeError(
        f"ffmpeg failed to merge {num_videos} videos.\n"
        f"Error:\n{errlog}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Merge multiple MP4 videos. Supports 2 (left-right), 3 (left-center-right), "
                    "4 (2x2 grid), or 6 (3x2 grid) videos. Supports speed adjustment."
    )
    ap.add_argument("videos", nargs="+", help="Input video files (2, 3, 4, or 6 videos)")
    ap.add_argument("--output", default="merge.mp4", help="Output video file")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (e.g., 2.0 for 2x speed, 0.5 for 0.5x speed). Default: 1.0")
    args = ap.parse_args()
    
    num_videos = len(args.videos)
    if num_videos not in [2, 3, 4, 6]:
        print(f"Error: Expected 2, 3, 4, or 6 videos, got {num_videos}", file=sys.stderr)
        sys.exit(1)
    
    if args.speed <= 0:
        print(f"Error: Speed must be positive, got {args.speed}", file=sys.stderr)
        sys.exit(1)
    
    for p in args.videos:
        if not os.path.isfile(p):
            print(f"Not found: {p}", file=sys.stderr)
            sys.exit(1)
    
    merge_videos(args.videos, args.output, speed=args.speed)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
