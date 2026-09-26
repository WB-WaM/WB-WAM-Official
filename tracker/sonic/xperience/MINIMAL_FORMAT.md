# Minimal Xperience Processing Format

This folder adds a minimal, non-invasive pipeline for testing Xperience-10M
episodes against the G1 + WUJI + SONIC data contract. It does not modify any
existing repository files.

## Raw Episode

The downloader keeps only these files per episode:

```text
annotation.hdf5
stereo_left.mp4
stereo_right.mp4
```

For the full dataset, pass a GPFS destination with enough free space and a
free-space floor. The script checks disk space before each file and stops before
crossing the threshold.

## Processed Episode

Each converted episode writes:

```text
annotation_minimal.hdf5
stereo_left.mp4
stereo_right.mp4
wuji_hand_action_40d.npy
g1_motion_minimal.npz
smpl_motion_minimal.npz
gmr_input_smpl24_joints.npz
sonic_encoder_obs_1762d.npy
sonic_action_token_64d.npy
state_action_107_104.parquet
processing_report.json
```

`annotation_minimal.hdf5` copies all available HDF5 groups except
`calibration` and `slam`.

Videos are resized to `224x224` with ffmpeg when available. If ffmpeg is not
available, they are hardlinked or copied and the report records that they were
not resized.

## Hand Validity

For each hand, a frame is invalid when any coordinate in that frame is NaN or
Inf. Invalid frames are repaired as follows:

```text
before first valid frame: fill zeros
after first valid frame: copy previous valid frame
```

The report includes invalid runs for left/right hands, so it is easy to see
whether a hand is missing from the start or drops intermittently.

The included WUJI retarget is a finite geometric proxy from 21 hand joints to
20 values per hand. It is not a calibrated WUJI hand map; use it to validate the
pipeline shape and replace it with a calibrated map when available.

## Body Retarget And SONIC Tokens

The script prepares SMPL-like arrays from `full_body_mocap/keypoints` and
creates SONIC encoder observations in the deploy encoder's 1762-D layout.
For the generated `sonic_action_token_64d.npy`, it uses SONIC `g1` mode
(`mode_id=0`): G1 joint positions, G1 joint velocities, and G1/root anchor
orientation over the future window. SMPL observations are still exported for
inspection and possible ablations, but they are not used for the default token.

GMR is treated as an optional external dependency. This minimal file does not
vendor GMR code. If SMPL/SMPL-X params are present, they are exported for GMR.
If only SMPL-24-like full-body joints are present, the script writes
`gmr_input_smpl24_joints.npz` and builds the per-frame GMR body dictionary
directly:

```text
full_body_mocap joints -> SMPL-24 names -> GMR smplx_to_g1 IK -> G1 qpos
```

When neither path can run, G1 output uses a default standing-pose fallback and
marks `g1_retarget.ok=false` in `processing_report.json`.

SONIC token encoding uses `onnxruntime` if it is installed and
`--encoder-model` is provided. Otherwise the token array is zero-filled and
marked invalid in the report. The state/action parquet still keeps the final
training contract:

```text
state:  107 = base_gravity(3) + base_ang_vel(3) + base_accel(3)
              + body_q(29) + body_dq(29) + WUJI actual/proxy(40)

action: 104 = SONIC token(64) + WUJI target(40)
```

## Example

```bash
python minimal_xperience_pipeline.py list-remote --sample --only-kept

python minimal_xperience_pipeline.py download \
  --sample \
  --raw-root /gpfs/$USER/xperience_raw_sample \
  --output-root /gpfs/$USER/xperience_processed_sample \
  --parse-after-download \
  --encoder-model <repo-root>/checkpoints/tracker/sonic/policy/release/model_encoder.onnx \
  --min-free-gb 50 \
  --overwrite
```

For a local smoke test that downloads nothing:

```bash
python minimal_xperience_pipeline.py self-test
```
