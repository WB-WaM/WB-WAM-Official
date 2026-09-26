# WB-WAM environments

Run from the repository root. `scripts/env/setup_envs.sh` installs `teleop` and
`wam`; `all` also installs the MuJoCo `sim` environment. Individual targets can
be selected, e.g. `scripts/env/setup_envs.sh wam`. `--force` recreates selected
environments and removes their installed packages.

| Environment | Directory | Installation |
| --- | --- | --- |
| SONIC collection | `.venv_teleop` | `scripts/env/setup_envs.sh teleop` |
| SONIC simulation | `.venv_sim` | `scripts/env/setup_envs.sh sim` |
| WB-WAM inference | `bridge/.venv-wam` | `bridge/scripts/setup_env.sh` |

Bridge inference uses PyTorch 2.7.1, torchvision 0.22.1 and transformers 4.49.0,
matching the training model runtime. Linux uses CUDA 12.8 wheels; macOS uses CPU
wheels for tests. Additional dependencies are listed in `bridge/requirements.txt`.
The setup installs `training/` with `--no-deps` to reuse the model package without
installing DeepSpeed or training data/video pipelines.

Use `scripts/env/check_envs.sh` to inspect imports and versions. The actual GPU,
camera, hand server and native SONIC deployment require the target Linux machine.
See [bridge instructions](../../bridge/README.md) for deployment details.

The teleop environment uses NumPy 2.2.6 and Pinocchio 3.8.0 for Wuji
retargeting, with cmeel-urdfdom 4.0.1 matching the Pinocchio wheel's native ABI.
Bridge and simulation environments retain NumPy 1.26.4.

For the default PICO GStreamer video backend, install the system dependencies
listed in the [collection guide](../../collector/sonic/README.md). With its
`libgirepository1.0-dev` package, install the compatible Python bindings:

```bash
uv pip install --python .venv_teleop/bin/python pycairo 'PyGObject==3.50.0'
```
