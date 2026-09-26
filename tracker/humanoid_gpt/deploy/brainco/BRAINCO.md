# BrainCo Dexterous-Hand Deployment

Drop-in BrainCo variant of the standard `deploy/` tracking pipeline.  Body
locomotion / tracking logic is **identical** to `deploy.play_track`; the
BrainCo path adds 24-DoF finger retargeting and a 12-D DDS command stream to
`BraincoController`.

## Pipeline

```
 Noitom PNLink ──► GMR (body)         ──► qpos_full  ──► LiveRefConverter ──► G1TrackInferFn ──► G1 motors
  (mocap suit)     GMR-galbot                                                                        │
                   ──► finger retarget ──► hand_qpos (24-D) ──► BraincoHandSmoother ──► hand_cmd (12-D)
                       (brainco / brainco2 / brainco3)                                               │
                                                                                                     ▼
                                                                                              BraincoController
                                                                                              (rt/brainco/* DDS)
```

The retarget subprocess produces **two** mocap streams in shared memory:

- `qpos_full[36]` — body (root pos / quat / 29-DoF joints)
- `hand_qpos[24]` — left finger qpos (12) ‖ right finger qpos (12)

`BraincoHandSmoother` converts `hand_qpos` to a 12-D actuator command
`[right(6) | left(6)]` with EMA smoothing, which is published at
`brainco_hand_fps` (default 100 Hz) by `BraincoController`.

## Files

```
deploy/brainco/
├── play_track_brainco.py        # Real-robot tracking + BrainCo hands
├── noitom_hand_retarget.py      # Noitom-format finger retargeting helpers
├── brainco_controller.py        # DDS publisher / subscriber for rt/brainco/*
└── __init__.py
```

`play_track_brainco.py` is **non-invasive**: it imports (and never modifies)
`deploy.play_track`, `deploy.retarget`, `deploy.real_robot`, and the local
`brainco_controller` / `noitom_hand_retarget`.

## Installation

The base setup follows `deploy/DEPLOY.md` (steps 1 – 5).  Additionally,
`deploy/brainco/brainco_controller.py` requires `unitree_sdk2py` — the same
dependency used by Dex3 control, so no extra package is needed.

## Quick Start

### 0. Setup the brainco_hand_service in Unitree G1
Refer to the following link to setup the brainco_hand_service in Unitree G1:
[https://github.com/unitreerobotics/brainco_hand_service](https://github.com/unitreerobotics/brainco_hand_service)

### 1. Real-robot tracking with BrainCo hands

```bash
conda activate h-gpt

python -m deploy.brainco.play_track_brainco --real \
    --net enx6c1ff76e8ef5 \
    --hand-target brainco2
```

Simulation mode delegates to `deploy.play_track.run_sim` (BrainCo hands
are a hardware-only feature):

```bash
python -m deploy.brainco.play_track_brainco --track_dir storage/test
```

## BrainCo-Specific Arguments

| Argument                       | Default     | Meaning                                              |
| ------------------------------ | ----------- | ---------------------------------------------------- |
| `--enable-brainco-hand`        | `True`      | Toggle BrainCo DDS command publishing                |
| `--hand-target`                | `brainco2`  | `brainco` / `brainco2` / `brainco3` (retarget map)   |
| `--brainco-hand-fps`           | `100`       | DDS publish rate for `BraincoController`             |
| `--brainco-hand-smooth-alpha`  | `0.45`      | EMA blend factor for the 12-D command (1.0 disables) |
| `--brainco-hand-scale`         | `1.0`       | Multiplicative gain on the final actuator command    |
| `--rest-hand-in-walk`          | `True`      | Send open-hand command while in walk mode            |

All other flags inherit from `deploy.play_track.DeployArgs`.

## Hand Retargeting Variants

| Target      | Description                                                              |
| ----------- | ------------------------------------------------------------------------ |
| `brainco`   | Default 24-DoF retargeting                                               |
| `brainco2`  | Improved thumb roll mapping (recommended for most motions)               |
| `brainco3`  | Aggressive index/middle splay mapping for fine manipulation              |

Implementation details live in `_retarget_noitom_hand_qpos`
(`deploy/brainco/noitom_hand_retarget.py`) and `_brainco_hand12_to_ctrl6`
(`deploy/brainco/play_track_brainco.py`).

## Relation to Existing Modules

```
deploy/retarget.py                       Body-only retarget (3-buf shared memory)
deploy/play_track.py                     Unified sim / real tracking (Dex3 hand)
deploy/brainco/play_track_brainco.py     ↑ + BrainCo hand retarget + BraincoController
deploy/brainco/brainco_controller.py     DDS publisher / subscriber for rt/brainco/*
```

`play_track_brainco` exposes `start_realtime_retarget_with_brainco_hands`
(returns a 4-buffer tuple: `buf_qpos`, `ts`, `buf_hand_state`,
`buf_hand_qpos`), `BraincoHandSmoother`, and
`load_offline_motions_with_brainco_hands`.
