# Working in WB-WAM

These instructions apply throughout this repository. Follow the user's selected task and read the relevant component guide before changing or running that component. A code or documentation edit does not require environment provisioning or robot access.

## Conversation

- Reply in the user's language. Ask only for missing information; reuse answers and authorization already supplied in the session.
- Ask at most three short questions per turn, preferably one or two. Inspect discoverable PC and robot facts yourself; ask only for missing information or physical actions that require the user.
- When an answer is required, ask in the final response and end the turn. Resume on the user's next message. Do not keep the turn active with sleep loops, status polling, or repeated reminders. Follow any host-required question-tool contract without adding an idle waiting loop.
- Complete short, independent checks before asking. Ending a turn for user input does not mean the overall task is complete. Silence is not an answer or authorization.

## Repository map

| Area | Purpose / reference |
| --- | --- |
| `training/` | WB-WAM training and policy code; [training guide](training/README.md) |
| `bridge/` | Real-robot policy deployment; [deployment guide](bridge/README.md) |
| `collector/sonic/` | SONIC/PICO collection, device probes and robot-side services; [collection guide](collector/sonic/README.md) |
| `collector/humanoid_gpt/` | Humanoid-GPT collection; [collection guide](collector/humanoid_gpt/README.md) |
| `benchmark/humanoidarena/` | Simulation evaluation; [evaluation guide](benchmark/humanoidarena/README.md) |
| `tracker/` | SONIC and Humanoid-GPT controllers |
| `scripts/env/` | Scoped environment helpers; [environment guide](scripts/env/README.md) |
| `third_party/` | Vendored dependencies and submodules; avoid unrelated changes |

## Environment setup requests

Read and follow [docs/agent_setup.md](docs/agent_setup.md) before configuring training, deployment, collection or HumanoidArena evaluation. It contains the detailed commands, configuration templates, staged questions and completion criteria. Select only the requested workflow.

- **The agent may execute real-robot deployment and collection on the PC and robot.** This includes SSH/SCP, dependency installation, approved downloads, source copying/building, local configuration, hardware checks and service startup/shutdown. Complete executable steps directly instead of requiring the user to run commands. An environment request authorizes ordinary reversible setup; live hand/body control and policy execution additionally require the intended motion to be authorized and the on-site operator ready. Reuse existing authorization. Keep processes observable, honor stop requests and verify cleanup. Repository code/documentation edits alone do not require robot access.
- **Training / deployment:** ask whether to download pretrain, midtrain, both, or neither/reuse existing checkpoints. Confirm the selected downloads before starting them, then execute the approved downloads.
- **HumanoidArena evaluation:** ask about the evaluation task and its task-specific checkpoint. Do not offer or download pretrain/midtrain or training datasets for evaluation alone.
- **Collection:** real-robot deployment setup must pass before collection setup. For a direct collection request, first complete and verify the shared robot/PC deployment in section 4B of `docs/agent_setup.md`, then continue to section 4C in the same conversation. Reuse verified results; failed or unverified deployment checks block collection-specific installation/configuration and PICO/MANUS probes. Follow the motion authorization and readiness requirements below. For collection alone, WB-WAM policy installation/weights are not required; offline conversion and simulation-only requests do not require robot deployment.
- For Chinese requests, default Hugging Face downloads to `https://hf-mirror.com` with proxy variables cleared for the download process only. For other languages, ask for the preferred endpoint. Do not silently switch endpoints or enable a proxy after failure.
- Create and populate the applicable local configuration from its example on each host, including `training/.env`, deployment and collection files. Preserve existing unrelated values and check effective configuration through the actual loader. Unchanged examples and placeholders do not constitute completed setup.
- Keep environments separate: training and the Arena policy server use `wbwam`; the Arena simulator uses its own environment; deployment uses `bridge/.venv-wam`; SONIC/PICO collection uses `.venv_teleop`; robot services use a robot-side environment. Read the component guide for other workflows. The root `pyproject.toml` configures tooling and is not an installable root package.

## Real hardware

- When asking to start live body and both-hand control, offer both agent execution after readiness confirmation and “控制终端帮我打开，我来控制” (translate for English conversations). Follow the terminal-handoff section in `docs/agent_setup.md`: open user-visible PC/robot terminals and prepare commands when selected, but leave motor-enabling commands and motion/start inputs to the user. Explain the controller's actual start/stop procedure and readiness requirements. If visible terminals are unavailable, provide host-specific commands instead. Reuse the user's choice; end the turn when a reply is needed.
- Have the user connect the Unitree robot to the PC by Ethernet. Ask only for missing addresses, SSH username, authentication and hardware information; reuse working SSH access and obtain missing authentication through the available secure interaction without storing credentials. Deploy, configure and verify both robot-side and PC-side software directly.
- The supported default hardware is a Unitree G1, G1 2-DOF camera head with RealSense D455, and two Wuji hands. The camera and hands connect by USB to the **robot**, where their services run. USB discovery must run on that host; PC discovery cannot enumerate USB devices across Ethernet.
- Wuji left/right calibration is mandatory. Install/copy `collector/sonic/scripts/discover_wuji_hands.py` and its SDK on the robot as needed. With the operator ready and calibration motion authorized, discover serials, reset both hands, move thumb joint 1 sequentially and read the validated result. Obtain a separate observation for each hand through chat; never infer the second answer. Use `--write-env` on the robot to save the mapping, preserving other settings, and verify it through the actual loader.
- For calibration, retain a supported SSH/PTY session across turns, relay only explicit observations to the same running program, and end the assistant turn after each observation question. Check the process/stage before forwarding a reply. Preserve the script timeout and motor-disable cleanup; stop it on user request. Calibration does not include head motion; handle head servos separately as below. Keep competing hand controllers stopped during calibration. Resolve missing software prerequisites directly; start live hand/body control or policies only as a separate authorized operation after readiness checks.
- Discover/select the camera, save its verified configuration, acquire frames, start/reuse/stop the configured camera service and verify PC reception. Coordinate head-servo operation as described below.
- For deployment AND real-robot collection, follow [the head bundle guide](collector/sonic/head/README.md) and run `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP`. The driver sources must be copied and built on the robot, never use a PC-built kernel module. Require CH340 enumeration, matching kernel module, real replies from both servos, bounded hold/cleanup and real camera frames. Report each failure explicitly.
- The agent may operate the head-module servos through the installed, verified control program/service: start/stop control, set the agreed pose or tracking mode, and check feedback. Confirm the actual interface, axis conventions, limits, requested target/mode and on-site readiness before motion; reuse authorization already given. If these details are unknown, ask rather than inventing a command. Default to `HEAD_SERVO_MODE=raw`, `HEAD_JOINT0_ENCODER=3027`, `HEAD_JOINT1_ENCODER=1849` (raw encoder counts, not degrees). Create the head configuration from its example and fill these defaults where unset, preserving explicit user overrides. Head calibration and initial teaching are not setup prerequisites; do not ask for zero/limit poses or calibration images. Only if the user explicitly requests changing the pose, use powered `HEAD_SERVO_MODE=teach`: after placement confirmation, send `hold` to the same process and save the captured counts without disabling torque. End the turn while waiting for placement; preserve the 300-second timeout and fault/stop cleanup. Inference and collection must use the supervised camera + head launcher (`run_camera_server.sh --head-motion-approved`). Head-only authorization does not by itself include body control or live policy execution; the agent may also perform those operations when their scope is authorized and the operator is ready.
- The agent may run `--check-config` and bounded bridge `--dry-run` checks, including mock or real-model inference using existing environments/assets. Keep `--dry-run` set; default to `--fake-camera --fake-state`. Do not start SONIC, live hand services or the live policy to satisfy a dry-run dependency. Resolve missing installations, approved assets and configuration directly before the check.
- For SONIC/PICO collection, PICO connects to the PC by Ethernet; its wired IP differs from the PC service address. The MANUS wireless USB receiver connects to the PC. Follow the guide's PICO tracking checks and mandatory MANUS left/right live-data checks; detecting a receiver alone does not establish paired gloves or usable data.
- Report camera or Wuji discovery/test failures explicitly and leave the relevant setup pending. Report actual commands, verified feedback and cleanup results. After authorization and on-site readiness, the agent may run deployment services, hand/body controllers, live policies and collection processes, keeping control of their sessions and stopping on request. Manual commands are a fallback only when access/tools are unavailable or the user chooses to run them.

## Changes and validation

- Inspect the working tree first and preserve unrelated user changes. Keep edits scoped; avoid bulk formatting, dependency upgrades or changes to vendored code unrelated to the task.
- Follow the surrounding code and component tooling. Root formatter/linter settings are in `pyproject.toml`; do not impose a new repository-wide style.
- Keep corresponding English and Chinese component documentation consistent when behavior changes. Put detailed agent dialogue in `docs/agent_setup.md`; keep user-facing READMEs focused on usage.
- Use the selected component's environment and relevant tests. For SONIC discovery/probe changes, the existing suite is:

  ```bash
  .venv_teleop/bin/python -m unittest discover -s collector/sonic/tests -p 'test_*.py'
  ```

- Run `git diff --check` for edits. Documentation-only changes normally need link/command review rather than GPU or hardware execution. Root `pytest` targets controller tests and is not a complete repository test suite.
- Inspect output as well as exit codes: `scripts/env/check_envs.sh` can report missing imports while returning zero. Distinguish unit/mock checks, model smokes, simulation and real hardware validation. Report what was actually run and what remains unverified.
- Keep credentials, robot addresses, device identifiers, local `.env` files, logs, datasets and checkpoints out of commits. Never store SSH passwords in commands, scripts or shell history. Review the staged diff before committing.
- Only commit, push, tag or synchronize the official repository when requested. Summarize changed behavior, relevant verification and outstanding requirements without claiming unperformed tests passed.
