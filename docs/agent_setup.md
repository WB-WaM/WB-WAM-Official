# Agent setup, downloads, and deployment guide

Real-robot deployment and operation are subject to the project’s [safety disclaimer](../README.md#safety-disclaimer) ([中文](../README_zh.md#安全免责声明)).

Give this file to an AI agent when you want help setting up WB-WAM. Use the user's requested workflow, clarify only missing choices, and confirm specific downloads before downloading. This is a selectable runbook, not a command to install everything.

Unless a section says otherwise, commands use the repository root as their working directory. Read the linked component guide before executing that component's commands. Never put passwords, tokens, robot addresses, or machine-specific `.env` files in Git or in a progress report. Do not store SSH passwords in scripts, configuration files, command logs, or shell history.

## Agent execution for deployment and collection

The agent may execute real-robot deployment and collection commands on both the PC and robot, including SSH/SCP, dependency installation, selected downloads, source copying/building, local configuration, hardware checks, and service startup/shutdown. Carry the requested workflow through verified results rather than requiring the user to run terminal commands. Apply the same execution model to training, deployment and collection; do not restrict deployment to a small set of exceptions.

An environment-setup request authorizes the ordinary reversible installation and configuration needed for that workflow. Real motion still requires authorization for the intended operation and an on-site operator who is ready to supervise and stop the hardware. Reuse authorization already given; an environment-only request does not itself start body control or a live policy. Once the requested motion scope and readiness are established, the agent may operate the camera, head servos, Wuji hands, body controller, live policy and collection processes. Keep dry-runs non-actuating and preserve each program's limits, timeouts and stop/cleanup behavior.

The user handles physical actions that require being on site: wiring/power, wearing or positioning devices, emergency stop, and observing which Wuji hand moves. The agent handles executable steps, retains the required interactive sessions, checks feedback and saves verified configuration. Ask only for missing information, physical actions or authorization that is actually needed; end the turn when a reply is required.

Label commands by **PC** or **robot** and use the correct working directory, environment and process session. Reuse working SSH authentication; if missing, obtain the minimum required authentication through the available secure interaction, without storing passwords in commands or files. Keep services observable so they can be stopped on request, and report cleanup failures explicitly. If tool access or authentication prevents execution, explain the concrete blocker and provide a manual fallback for that step; manual execution is not the default requirement.

Sections 2–4 contain executable reference commands. Install missing prerequisites, create and populate configuration files, execute the applicable checks and inspect their output. Repository code/documentation edits alone do not require robot access or hardware execution.

### Choose who operates live body and hand control

When asking for readiness and authorization to start body and both-hand control, offer the user a terminal handoff alongside agent execution. For Chinese conversations, use these reply options:

- **由 Agent 控制：** “机器人活动区域无人和杂物，硬件急停随时可用，可以开始身体和双手控制。”
- **由我控制：** “控制终端帮我打开，我来控制。”

For English conversations, offer the equivalent choices: “The robot workspace is clear, the hardware E-stop is available, and you can start body and both-hand control” or “Open the control terminals for me; I will operate them.” Ask in the final response and end the turn. Reuse an already explicit choice and readiness confirmation rather than asking again.

If the user chooses to operate, open user-visible control terminals for the required PC and robot sessions, with the correct working directories and environments, and prepare the verified startup commands. Leave commands that enable motors, publish motion or start a live policy for the user to execute; do not submit those commands or send start/motion keys on their behalf. Explain the actual start, stop and emergency-stop procedure from the selected controller's guide. Opening terminals does not itself confirm a clear workspace: remind the user to check workspace and hardware E-stop readiness before starting control. Continue independent non-actuating setup and checks as needed.

If user-visible terminals cannot be opened, explain that limitation and give the prepared commands grouped by PC/robot and terminal. An agent-only tool PTY is not a terminal handed to the user. This choice applies to the live deployment and collection launch sequences below; when the user selects terminal handoff, prepare those sequences for them instead of executing actuation commands. Agent execution remains available when the user chooses it and authorizes the operation.

## 1. Ask the user first

### Keep the conversation short and staged

- **A question that needs the user's answer ends the assistant turn.** Complete any short, independent checks before asking, then put the question in the final response and stop generating. In Codex, use a final response rather than a commentary/progress message for this handoff. Resume setup on the user's next message; ending this turn does not mean abandoning or completing the setup. The user must be able to answer normally without pressing Esc or interrupting a running turn.
- Do not keep the turn active just to wait for an answer: no sleep loop, repeated status polling, empty tool calls, or repeated reminders. A tool yielding output is not the same as ending the assistant turn. Do not choose an asynchronous question tool merely because it is available; use a native question tool only when the host instructions require it or it provides a proper user-input handoff. If an asynchronous tool is required, follow its host contract without adding an idle polling loop. Silence is never an answer or authorization.
- Ask **at most three short questions per turn**, preferably one or two. Each question should resolve one immediate choice; do not hide a long questionnaire in numbered groups, nested lists, or a reply template. Do not dump the checklist below or the complete setup plan into the first response.
- Treat the workflow named in the user's prompt as selected. Inspect the selected PC and robot hosts' OS, tools, GPU/driver, storage, existing environments and local configuration with read-only checks. Ask only for missing facts that cannot be discovered. Assume the current host is the PC setup host unless the user indicates otherwise, and state that assumption briefly.
- A request to configure an environment authorizes the ordinary reversible setup and local configuration needed for that workflow. Reuse authorization already given; do not ask a generic list of permissions to install packages, create environments, build components and write `.env` files. Resolve missing scope or choices before the dependent action. Confirm specific downloads, privileged system changes when needed, GPU/simulator jobs and hardware actuation at their respective stages; a setup request alone does not authorize robot motion.
- Ask about existing assets and their destination when reaching the download/configuration stage. Check known local paths first and propose a concrete writable default when appropriate. Defer task language instructions, output naming and optional GPU tests until they are needed. Independent authorized work may proceed before the question or in a later turn; do not keep working after asking a question that requires a reply.
- First-turn priorities: **training** — training type and the required pretrain/midtrain download choice; **deployment** — non-secret robot connection information, confirmation of the supported hardware/wiring, and the required pretrain/midtrain choice; **collection** — first establish whether real-robot deployment setup has already passed; if not, enter section 4B before collection setup and ask only for missing robot connection/hardware information; defer SONIC/PICO versus HGPT input details until deployment is ready; **HumanoidArena evaluation** — evaluation task and whether to reuse or download its task checkpoint. Ask for PICO addressing and MANUS details in the collection input-configuration stage, after the initial robot connection discussion.
- State applicable defaults briefly rather than asking the user to reconfirm them, including `hf-mirror.com` without a proxy for Chinese requests. Keep reminders about later mandatory hardware checks to one short sentence. Do not append boilerplate explaining that the guide requires permission.

The following is an **internal checklist to cover over successive turns**, not a first-response questionnaire. Collect each applicable answer before the operation that depends on it. Unrelated unanswered items need not prevent independent setup, but every question requiring a reply follows the turn-ending rule above:

1. Which outcome(s) are wanted: Pico intermediate training, self-collected post-training, HumanoidArena post-training, HumanoidArena evaluation, WB-WAM bridge deployment, SONIC/PICO collection, HGPT collection, SONIC simulation, or offline LeRobot conversion? Which task/checkpoint and language instruction?
2. Which environments should be created or reused? What are the OS, Python/conda/`uv` availability, CUDA driver/GPU count, disk space, network access, and writable storage locations? Is this a PC, cluster node, or robot?
3. Which *specific* downloads are approved for the selected workflow? **For deployment or training, include an explicit question in the initial setup discussion about whether to also download the Hugging Face WB-WAM pretrain and/or midtrain checkpoint bundles**, even if the user only mentioned environment setup. Offer pretrain, midtrain, both, or neither/reuse existing files; do not assume download approval. **HumanoidArena evaluation is not training or real-robot deployment: do not ask about, offer, or download pretrain/midtrain for evaluation.** Its assets are the selected task's post-trained checkpoint/config/stats, pinned base models, SONIC ONNX and simulator assets. Reusing the training `wbwam` Python environment does not require training weights or datasets. If the user separately requests training or pretrain/midtrain, handle that as a separate scope. Collection likewise does not need policy weights. Ask for missing existing paths and the download destination at the relevant stage before downloading duplicates. For requests in Chinese, default Hugging Face downloads to `https://hf-mirror.com` with no proxy unless the user explicitly specifies otherwise; state this default instead of requiring a separate mirror choice. For other requests, ask for the preferred Hugging Face/ModelScope mirror. Check access and estimated size first.
4. Confirm only operations that need additional authorization, such as a privileged system change, bounded GPU/Isaac smoke or hardware motion. Before motion, establish the intended operation, on-site operator readiness, clear workspace and available hardware E-stop. The agent may execute the authorized operation directly, including live hand/body control and policy execution. Reuse authorization already given; do not ask again merely because a command runs on the robot.
5. **When deployment or real-robot collection setup is selected, confirm hardware availability at the start:** ask whether the Unitree (宇树 / Yushu) G1, G1 2-DOF head camera module (RealSense D455), and both left and right Wuji hands are all present and connected using the default wiring. One confirmation may cover this named set; ask device-specific follow-ups only for missing, different or uncertain hardware, including its model and actual USB host. The current bridge supports this hardware combination; missing or different hardware must not be reported as ready for real deployment. When devices are absent, mark hardware checks pending and continue independent authorized software preparation. Walk the user through the IP and serial-number configuration in section 4B; do not merely ask them to supply unexplained identifiers.
6. **During SONIC/PICO input configuration, additionally confirm PICO and MANUS:** default to PICO connected to the PC by **Ethernet**, and MANUS gloves paired to a **USB wireless receiver plugged into the PC**. Collect the PICO model, controllers/ankle trackers, **actual PICO wired IP**, and DHCP status over short successive exchanges as needed. Discover the PC's PICO-facing IP/interface from its route to that address; keep it separate from the robot-facing PC IP. Guide the user to find unknown addresses and configure the headset as in section 4C. Automatically detect the MANUS receiver and glove IDs with the bundled probe; do not ask the user to guess receiver serials or assign it an IP. If another hand-input mode was explicitly selected, follow that choice instead of requiring MANUS.

**Robot connection and robot-side deployment are required for deployment or real-robot collection setup.** Explain: “请将宇树机器人通过网线连接 PC，并将头部相机和左右 Wuji 手通过 USB 接到机器人。我会连接机器人，配置机器人端与 PC 端环境，并实测相机、舵机和 Wuji 双手；需要现场操作或左右手观察时再请你配合。” Reuse known IP/interface, SSH username, authentication and installation paths; ask only for missing information. Execute section 4B on the correct hosts and inspect the results. Without working robot access, report **robot-side deployment and hardware checks pending**. Training, simulation-only and offline conversion do not require this robot connection.

**Wuji left/right identification is the required part of deployment/collection.** Both hands stay connected: after operator readiness and motion authorization, the agent starts the identification script; the user observes the first thumb and replies in chat, then independently observes and confirms the second hand. Relay each answer only to its corresponding active stage. Validate/read the receipt, write the mapping with `--write-env` on the robot and verify the effective configuration. Never infer either side. Package installation and two detected serials do not establish a correct mapping. If verification evidence is missing, report **left/right identification not verified**. Training-only setup does not require this step.

| Hardware path | Deployment | Default SONIC/PICO collection |
| --- | --- | --- |
| Unitree ↔ PC | Ethernet | Same |
| Camera + Wuji hands → robot | USB; discover/configure on robot | Same, including Wuji side confirmation |
| PICO ↔ PC | Not required | Ethernet; actual PICO IP and PC's PICO-facing IP required |
| MANUS receiver → PC | Not required | USB wireless receiver; automatic USB + left/right stream checks |

**Required order: real-robot deployment setup → data-collection setup.** This applies to both SONIC/PICO and real-robot HGPT collection, including a direct request such as “帮我配置数据收集环境”. Start by explaining: “数据收集需要先配好真机部署环境。我们先完成机器人和 PC 的部署配置及硬件检查，通过后再配置采集所需的 PICO、MANUS 等设备。” Then follow section 4B's shared robot/PC deployment steps in the same conversation; do not merely send a deployment link or require the user to start a separate request. Ask only for missing information and end the turn when an answer or physical action is needed.

Before moving to collection setup, require evidence that robot SSH/wired networking, robot-side dependencies and services, PC/robot endpoint configuration and configuration-file readback, real camera frame reception, head-driver/servo tests, and Wuji USB discovery, two independently confirmed sides, motion/cleanup and saved serial mapping have passed. Reuse verified results for unchanged hardware/configuration; do not reinstall or repeat calibration unnecessarily. A claim that deployment is done, a list of commands, or installed packages alone is insufficient. Any failed or unverified item means **deployment prerequisite incomplete; collection setup pending**. Explain the failed item and resolve it before installing/configuring collection-specific inputs or running PICO/MANUS probes. Once deployment passes, explicitly report that result and continue to section 4C without asking the user to repeat the original collection request.

For a collection-only request, this prerequisite covers the shared real-robot deployment environment and hardware checks. WB-WAM policy installation/checkpoint downloads and live policy execution are not needed solely for collection. Follow the authorization and on-site readiness requirements above; complete environment setup before starting authorized live control. Offline conversion and simulation-only requests are exempt from this real-robot prerequisite.

**Mandatory on a data-collection request using MANUS:** the agent must run the full left/right glove-data check in section 4C before starting collection or declaring the inputs ready. This applies to first-time setup and subsequent collection sessions. Tell the user to power/pair both gloves and gently move both hands, then execute the probe; do not leave them a command to run themselves. USB detection, a successful import, or a previous session's result cannot replace this check. A successful full probe already performed for the current session and unchanged hardware satisfies the requirement; repeat after receiver/glove reconnection, power cycling, re-pairing, or a data fault. If another hand-input mode was explicitly selected, validate that input instead.

Briefly summarize the choices and proceed with authorized work; do not ask the user to reconfirm answers they just supplied. Confirm unapproved downloads or GPU/simulator jobs at the relevant stage. Distinguish mock checks, training steps, simulator episodes and actual robot motion. Execute deployment checks and authorized hardware operations directly, retaining feedback and stop/cleanup monitoring. Do not use `--force`, overwrite existing output, or clean existing data without explicit approval.

| Selected workflow | Environment | Minimum assets |
| --- | --- | --- |
| Pico intermediate training | `wbwam` conda, Python 3.10 | Pico archive, full pretrain checkpoint/config/stats, Wan assets, ActionDiT initialization |
| Self-collected post-training | `wbwam` conda | Selected Real task, pretrain checkpoint by default (or user-selected midtrain), task-specific stats, Wan assets |
| HumanoidArena post-training | `wbwam` conda | Official `sonic_8_refpose_v3_1`, pretrain checkpoint, Wan assets |
| HumanoidArena evaluation | `wbwam` policy + separate Isaac environment | Task checkpoint/config/stats, pinned base models, SONIC ONNX, simulator assets |
| WB-WAM bridge deployment | `bridge/.venv-wam` + native SONIC build | Chosen checkpoint/config/stats, Wan inference assets, SONIC encoder/decoder/planner, robot endpoints |
| SONIC/PICO collection | `.venv_teleop` + robot-side environment | PICO/hand input, SONIC native assets, camera and Wuji services |
| HGPT collection | `h-gpt` Python 3.11/3.12 + robot-side services | HGPT tracking and walking ONNX, PICO/GMR/MANUS as selected |
| Offline SONIC conversion | `.venv-process`, Python 3.10–3.12 | Existing **20 Hz** raw SONIC episodes; no GPU or model weights |
| Optional SONIC MuJoCo simulation | `.venv_sim` | Simulator assets and the selected SONIC controller |

## 2. Inspect before installing

For deployment and collection, inspect the PC and robot directly, install the required dependencies and verify the selected environment. Reuse working access and existing installations; ask for assistance only when a specific access or physical prerequisite is missing.

Check the selected paths without exposing credentials. If the repository was cloned without submodules and HumanoidArena/GMR is selected, ask before running `git submodule update --init --recursive`. Check Git LFS files are real assets rather than tiny pointer text. From the repository root, the read-only preflight is:

```bash
git rev-parse HEAD
python --version
command -v conda
command -v uv
command -v ffmpeg
nvidia-smi  # Only on a selected GPU host.
df -h .
```

Record missing tools, available storage, and existing environment/asset paths; do not treat the repository's `checkpoints/` placeholders as model weights. Missing tools are scoped to the selected workflow: the CPU converter does not need CUDA, while training and real WB-WAM inference do.

The default setup helper creates both teleop and bridge environments. **Select one target** unless the user requested both:

```bash
scripts/env/setup_envs.sh teleop   # SONIC/PICO collection only
scripts/env/setup_envs.sh sim      # optional SONIC simulation only
bridge/scripts/setup_env.sh        # WB-WAM real-robot policy only
```

After any of these, run `scripts/env/check_envs.sh`. It prints `ERR` or `missing` for failed imports but can still exit with code 0; inspect the selected environment's lines, not just the command's exit status. Unselected environments may legitimately be reported missing. See the [environment guide](../scripts/env/README.md).

### Create and populate local environment files

**Creating and filling the selected workflow's `.env` files is mandatory setup work, including training, deployment and collection. The agent creates/edits these files on the applicable hosts and verifies effective values through the real loader.** Do not stop at dependencies or an unchanged example. Use the user's answers, verified host/device information, approved download destinations and existing asset paths to write the actual local settings. Ask only for missing required values; never invent an IP, device ID, checkpoint or data path.

| Selected workflow / host | Template → local configuration | Required adaptation |
| --- | --- | --- |
| Training host | `training/.env.example` → `training/.env` | Selected dataset roots, base-model and checkpoint paths, ActionDiT, stats/text caches, run/cache/temp directories and any verified CUDA/FFmpeg overrides |
| WB-WAM deployment PC | `bridge/.env.example` → `bridge/.env` | Selected checkpoint and SONIC encoder, text cache, actual robot camera/hand endpoints and verified camera key |
| SONIC/PICO collection or native SONIC deployment PC | `collector/sonic/scripts/collector_pc.env.example` → `collector/sonic/scripts/collector_pc.env` | Robot-facing interface, service hosts/ports, PC return address; for collection also task/output paths, PICO video IP, hand-input mode and applicable calibration paths |
| Robot camera host | `collector/sonic/scripts/camera_server.env.example` → `collector/sonic/scripts/camera_server.env` | Robot-side Python executable, bind address/port and selected camera settings |
| Robot head host | `collector/sonic/scripts/head_servo.env.example` → `collector/sonic/scripts/head_servo.env` | Verified CH340 serial path, `HEAD_SERVO_MODE=raw`, `HEAD_JOINT0_ENCODER=3027`, `HEAD_JOINT1_ENCODER=1849` (encoder counts; no head calibration or teaching required; preserve explicit user overrides); executable override only if needed. Agent may create/update this head-specific file. |
| Robot hand host | `collector/sonic/scripts/wuji_hand_server.env.example` → `collector/sonic/scripts/wuji_hand_server.env` | Robot-side Python executable, PC return address, feedback port and physically verified left/right Wuji serials |
| HGPT collection host | `collector/humanoid_gpt/scripts/hgpt_collection.env.example` → `collector/humanoid_gpt/scripts/hgpt_collection.env` | Selected workflow's actual paths, interfaces and service endpoints according to the HGPT guide |

Create each missing file from its template with `cp -n` on the host that uses it. For an existing local file, inspect it first, preserve unrelated settings, and update only the selected configuration fields. Do not replace a working file wholesale or edit `.env.example` to contain machine-specific values. Preserve the template syntax: launcher files contain shell assignments/default expressions, while training and bridge use their own loaders. Keep passwords and tokens out of all these files and never commit local environment files.

For **training**, populate `WB_WAM_DATA_ROOT` and the applicable `WB_WAM_PICO_ROOT`, `WB_WAM_REAL_ROOT` or `WBWAM_ARENA_NATIVE50_ROOT`. Set `DIFFSYNTH_MODEL_BASE_PATH`, `WBWAM_ACTION_DIT_PATH` and the selected `WBWAM_PRETRAIN_CHECKPOINT` / `WBWAM_MIDTRAIN_CHECKPOINT` to real downloaded or reused assets; do not assume the example step filenames exist. Configure `WBWAM_STATS_ROOT`, `WBWAM_TEXT_CACHE_ROOT`, `WBWAM_RUNS_ROOT`, `WBWAM_CACHE_ROOT`, `WANDB_DIR` and a short writable `TMPDIR`, including dependent cache paths when needed. Relative training paths are based on `training/`; use absolute paths for storage outside it. Create selected writable output/cache directories as needed, but do not create empty input directories as evidence that datasets or checkpoints exist. Preserve `DIFFSYNTH_SKIP_DOWNLOAD=true` unless the user explicitly selected on-demand downloads. Set CUDA/FFmpeg/module overrides only when inspection shows they are required. Unselected optional fields may retain their defaults; required assets that are not yet available must remain explicitly pending.

After downloads and hardware identification, revisit and finish any pending values. Read back the edited fields without disclosing credentials/device identifiers, check the selected input files and output-directory permissions, and verify effective settings using the workflow's actual loader. For training, run `source scripts/load_env.sh` from `training/` in the active training environment and inspect only the relevant resolved paths; exported variables can override `.env`, so check and resolve conflicting overrides. For shell launcher files, check shell syntax and effective endpoint/path values without launching hardware. Report which local files were created or updated on which host and any unresolved required values. **Setup is not complete while a selected workflow's required `.env` file is missing, still contains placeholders, or resolves to unverified required paths.**

## 3. Download only selected assets

For deployment, execute the selected downloads after confirmation, using the agreed endpoint/proxy settings and destination. Verify the resulting assets before resolving their paths in the local configuration.

Use the user's destination instead of assuming the examples below. The examples use the current `hf` CLI; first check `hf download --help`. If the selected environment only has the older `huggingface-cli`, check `huggingface-cli download --help` and substitute that executable. Do not upgrade the pinned training environment merely to get a downloader. Authenticate through the normal CLI only when the repository requires it, and never print a token. Inspect existing files before any download and verify the resulting files are nonempty and match the selected checkpoint's config/stats. See the [training guide](../training/README.md) for the corresponding model/data layout.

From `training/`, download **only the chosen** WB-WAM bundles:

`scripts/download_models.py` inherits `HF_ENDPOINT` from the environment. For requests in Chinese, default to `HF_ENDPOINT=https://hf-mirror.com` and direct connections without a proxy, unless the user explicitly chose different settings. Apply this policy to all selected Hugging Face downloads, including the Python downloader and `hf` CLI commands below. Clear both uppercase and lowercase proxy variables and set `NO_PROXY`/`no_proxy` to `*` in the download process; do not modify the user's global shell or proxy configuration. For example, from `training/`, after approval to download pretrain:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://hf-mirror.com NO_PROXY='*' no_proxy='*' \
  python ../scripts/download_models.py pretrain --output ./checkpoints/wbwam
```

Use the same `env` prefix for each selected Hugging Face download command. For other requests, use the user's selected endpoint/network settings. If the mirror or direct connection fails, report the failure and ask how to proceed rather than silently enabling a proxy or switching endpoints.

```bash
# Needed for Pico midtraining and default self-collected/Arena post-training.
python ../scripts/download_models.py pretrain --output ./checkpoints/wbwam

# Only if the user chose an existing midtrain checkpoint or a midtrain warm start.
python ../scripts/download_models.py midtrain --output ./checkpoints/wbwam

# Only if the user chose HumanoidArena evaluation for this task.
python ../scripts/download_models.py arena --task hammer --output ./checkpoints/wbwam
```

For a HumanoidArena task-specific checkpoint, use [WB-WAM-HumanoidArena](https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena) and keep its `step_*.pt`, resolved `config.yaml`, and `dataset_stats.json` together under `humanoid_arena_native50/<task>/`. Download into the same `./checkpoints/wbwam` root as pretrain.

For other task checkpoints, ask the user for the release location. Do not substitute midtrain for a post-trained task policy without the user's choice. If a model repository is inaccessible, stop and request credentials or a trusted mirror instead of guessing a replacement.

For the selected training data, from `training/`:

```bash
# Pico intermediate training only.
hf download WB-WAM/Pico --repo-type dataset --local-dir ./data/pico_archive

# Self-collected post-training only.
hf download WB-WAM/Self-Collected --repo-type dataset --local-dir ./data/real_archive

# HumanoidArena post-training only; not needed merely to run the simulator.
hf download WilliamWang16/HumanoidArena_dataset_v3_1 \
  --repo-type dataset --revision a079beddd6b1521f762c991be8f36993f17ebeca \
  --include 'HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1/**' \
  --local-dir ./data/humanoid_arena
```

The Pico and self-collected repositories already contain unpacked recordings; do not unzip them. In `training/.env`, set `WB_WAM_DATA_ROOT`, `WBWAM_ARENA_NATIVE50_ROOT` (if selected), and checkpoint paths to the actual locations. Preserve the official Arena data at 50 Hz; do not shift its actions or resample it offline.

WB-WAM model loading and text precomputation also need the Wan2.2 video DiT/VAE/T5 assets and Wan2.1 tokenizer. The loader's expected paths are determined by `DIFFSYNTH_MODEL_BASE_PATH` and [`loader.py`](../training/src/wbwam/models/wan22/helpers/loader.py): `Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors`, `DiffSynth-Studio/Wan-Series-Converted-Safetensors/{Wan2.2_VAE,models_t5_umt5-xxl-enc-bf16}.safetensors`, and `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/`. Stage these only if the chosen workflow needs them. `training/.env.example` sets `DIFFSYNTH_SKIP_DOWNLOAD=true`; do not silently flip it and start a large download during a smoke. ActionDiT initialization is a separate file: reuse an existing verified copy or, after Wan assets are ready, run the preprocessing command in the [training guide](../training/README.md). A WB-WAM pretrain checkpoint does not by itself prove these base assets are present.

If the user approved these large downloads and chose `training/checkpoints` as `DIFFSYNTH_MODEL_BASE_PATH`, stage the loader's exact layout from `training/`:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' \
  --local-dir ./checkpoints/Wan-AI/Wan2.2-TI2V-5B
hf download DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  Wan2.2_VAE.safetensors models_t5_umt5-xxl-enc-bf16.safetensors \
  --local-dir ./checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include 'google/umt5-xxl/*' \
  --local-dir ./checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

Check the actual `.env` base path and the required files before running text precomputation or training. If ActionDiT initialization is missing and the user approved generating it, from `training/` run:

```bash
source scripts/load_env.sh
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/wbwam_optional_idm.yaml \
  --output "$WBWAM_ACTION_DIT_PATH" --device cuda --dtype bfloat16
```

If any repository is unavailable, ask for a trusted existing copy; do not substitute a different Wan model or silently enable on-demand downloads.

For bridge or Arena inference, the pinned base-model downloader is available after approval:

```bash
python benchmark/humanoidarena/download_base_models.py \
  --base ./checkpoints/base_models --provider modelscope
```

It verifies six file hashes and refuses mismatched existing files. This six-file bundle is the intended base-model layout for HumanoidArena evaluation. Training and bridge deployment have different model requirements; follow their component guides instead of assuming this bundle replaces them. For SONIC, select the files actually used from the [NVIDIA GEAR-SONIC release](https://huggingface.co/nvidia/GEAR-SONIC). For bridge deployment, an example from the repository root is:

```bash
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml planner_sonic.onnx \
  --local-dir ./checkpoints/sonic_release
```

Set the resulting encoder path in `bridge/.env` and the decoder/planner/observation paths for the native SONIC launcher in `collector_pc.env` or its exported overrides; the example destination is **not** the launcher's default path. The Arena simulator uses the SONIC release path in `benchmark/humanoidarena/runtime.local.yaml`. HumanoidArena's Isaac Lab checkout and released `objects/`/`robots/` assets are separate; use the WB-WAM HumanoidArena environment setup guide ([English](../benchmark/humanoidarena/environment.md) · [中文](../benchmark/humanoidarena/environment_zh.md)) as the single source of truth for installation and downloads, then use the [benchmark guide](../benchmark/humanoidarena/README.md) for evaluation commands. Do not substitute or merge installation commands from another HumanoidArena or Isaac Lab release. HGPT's setup requires the tracking and walking ONNX paths described in the [HGPT guide](../tracker/humanoid_gpt/README.md).

## 4. Environment checks, bounded smokes, and example runs

Use only the selected subsections. Execute the applicable deployment/collection checks and authorized hardware operations directly. State the working directory and resolve missing assets or configuration before dependent checks. The smoke commands below are checks, not substitutes for task-quality evaluation or hardware commissioning.

### A. Training and WB-WAM policy environment

From the repository root, create the shared training/Arena-policy environment as in the [root installation guide](../README.md), after installation approval:

```bash
conda create -n wbwam python=3.10
conda activate wbwam
python -m pip install --upgrade pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./training
```

On Ubuntu, install the system FFmpeg package if missing and approved. Then from `training/`:

```bash
cp -n .env.example .env
# Populate .env using the verified paths as required in section 2 before loading it.
source scripts/load_env.sh
python -m pip check
python -c 'import torch, torchcodec, wbwam; print(torch.__version__, torch.cuda.is_available())'
VIDEO=/path/to/one/real/episode.mp4 python -c 'import os; from torchcodec.decoders import VideoDecoder; print(VideoDecoder(os.environ["VIDEO"])[0].shape)'
```

On a GPU training host, CUDA must be available and the video command must decode a real selected-dataset file. Install system FFmpeg libraries if needed, as shown in the root README. A successful import does not validate the data mapping. For a **user-approved** one-step GPU smoke, first precompute the matching text embeddings and use the proper norm stats, then run from `training/` with an agreed `NPROC` (the example uses 8 GPUs):

```bash
python scripts/precompute_text_embeds.py wb_task=midtrain
bash scripts/train_zero2.sh 8 wb_task=midtrain \
  "data.pretrained_norm_stats=./checkpoints/wbwam/pretrain/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT" \
  batch_size=1 max_steps=1 save_every=0 save_final=false eval_every=0 wandb.enabled=false
```

Pass: one optimizer step completes with finite loss and no data/shape/decode error. `save_final=false` intentionally creates no checkpoint. For self-collected or Arena post-training, compute **separate per-task** norm stats and text caches before the same one-step smoke. Examples from `training/`:

```bash
TASK=pillow
python scripts/compute_wb_norm_stats.py --output-dir "$WBWAM_STATS_ROOT/$TASK" "wb_task=posttrain/$TASK"
python scripts/precompute_text_embeds.py "wb_task=posttrain/$TASK"
bash scripts/train_zero2.sh 8 "wb_task=posttrain/$TASK" \
  "data.pretrained_norm_stats=$WBWAM_STATS_ROOT/$TASK/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT" \
  batch_size=1 max_steps=1 save_every=0 save_final=false eval_every=0 wandb.enabled=false

TASK=hammer
python scripts/compute_wb_norm_stats.py \
  --output-dir "$WBWAM_STATS_ROOT/humanoid_arena_native50/$TASK" \
  "wb_task=posttrain/humanoid_arena_native50/$TASK"
python scripts/precompute_text_embeds.py "wb_task=posttrain/humanoid_arena_native50/$TASK"
bash scripts/train_zero2.sh 8 "wb_task=posttrain/humanoid_arena_native50/$TASK" \
  "data.pretrained_norm_stats=$WBWAM_STATS_ROOT/humanoid_arena_native50/$TASK/dataset_stats.json" \
  "resume=$WBWAM_PRETRAIN_CHECKPOINT" \
  batch_size=1 max_steps=1 save_every=0 save_final=false eval_every=0 wandb.enabled=false
```

For a full run, remove the smoke overrides and use the appropriate [training recipe](../training/README.md), after a separate GPU allocation and output-path confirmation. Do not reuse Pico norm stats for post-training.

### B. Robot-side deployment and hardware checks; bridge deployment

Before executing deployment, complete the hardware questions in section 1. The agent may perform every executable step in this section within the requested scope and motion authorization. Use the correct host, environment, paths and device mappings. Keep addresses and serial numbers in local configuration; do not copy them into Git or progress reports.

**Default wiring is PC ↔ Ethernet ↔ Unitree robot, with both Wuji hands and the head camera plugged directly into the robot via USB.** Camera/hand services and USB discovery run on the robot. Copy the discovery program to the robot before running it; executing it on the PC cannot enumerate robot-side USB devices over Ethernet. Use `remote_wuji_proxy` and robot-local hand serial configuration for this topology. Only change these assumptions when the user explicitly specifies different wiring.

#### Required robot-side deployment for both deployment and collection

Perform these steps on the actual robot and PC. Reuse verified results already available in this session for unchanged hardware; do not reinstall or repeat motion without a reason. The user supplies physical connections, readiness and the two independent hand observations; the agent performs installation, configuration, commands and result verification.

1. **Verify SSH and the wired route.** Connect from the PC using `ssh USER@ROBOT_IP` and the available secure authentication. Check the remote hostname, OS/architecture, Python, USB devices and writable installation path to confirm commands are running on the robot. Diagnose routing, connection timeout and authentication failures separately; a PC-side check is not evidence of robot access.
2. **Deploy the robot-side software.** Follow the [bridge robot-side instructions](../bridge/README.md#3-robot-side-camera-and-wuji-services): copy the complete `collector/sonic/` service directory, including discovery scripts, while preserving robot-local `.env` files; create/reuse the robot's `.venv_robot`, install the required camera/hand dependencies, and create missing service configs from their templates. Set `COLLECTOR_PYTHON`, PC return address and matching service ports. Check SDK imports and USB access on the robot. Do not copy the PC environment or model weights to the robot. Installing these services does not mean starting live hand/body control.
3. **Agent-operated camera check.** Run camera discovery below on the robot, identify the selected D455 by model and serial, and verify its camera key. Capture at least one real color frame from that selected device and confirm nonempty image data and dimensions; an import or enumerated USB device alone is insufficient. Start/reuse the configured camera service and verify the PC can obtain a frame for that key through the configured endpoint. Use bounded checks, record the result, and release any temporary direct camera pipeline before the service opens the device. Do not substitute fake-camera output for this test.
4. **Agent-operated Wuji left/right calibration.** Discover both USB serials on the robot and complete the bounded reset/thumb sequence, user-observed side confirmation, saved-result verification and robot-local config readback below. The sequence must succeed for both hands, including position feedback and motor-disable cleanup. Merely listing two serials does not prove the hands move or that sides are configured correctly. Explain that the operator must be ready before running the identification command; keep other hand controllers stopped. This test does not authorize robot-body motion or policy execution.
5. **Install and test head servos automatically, for deployment and collection.** Follow the bundled [head guide](../collector/sonic/head/README.md) ([中文](../collector/sonic/head/README_zh.md)). On the PC, run `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP` with the actual SSH target. It copies sources, compiles `head_servo` and the missing CH341 module against the robot's running kernel, configures the CH340-specific brltty rule and serial permissions, preserves/creates `head_servo.env` and fills its verified serial path if unset, then reconnects and probes both servos without enabling torque. Install missing head-specific dependencies as part of this authorized setup; do not upgrade the kernel. Require matching headers/Module.symvers and module vermagic, a stable CH340 TTY and real replies from servo IDs 0/1. After checking the configured default encoder pose and operator readiness, run `run_head_servo.sh --motion-approved --duration 3` on the robot and require successful position feedback and torque-disable readback, then verify real camera frames with the supervised camera launcher. All failures remain explicit pending items. Continue the other robot deployment steps after this head-specific check.

   **Use the fixed default head pose; no head calibration is required.** Create `collector/sonic/scripts/head_servo.env` from its example with `HEAD_SERVO_MODE=raw`, `HEAD_JOINT0_ENCODER=3027`, and `HEAD_JOINT1_ENCODER=1849`. These are raw encoder counts, not angles or zero/limit references. Fill missing values and preserve explicit user overrides; verify the effective configuration through the launcher loader. Do not request head calibration, initial teaching, lower-limit placement or calibration pictures during environment setup. Continue the driver, serial, two-servo feedback, bounded hold/cleanup and camera checks above; missing calibration is not a blocker for raw mode. Inference and collection use `run_camera_server.sh --head-motion-approved` to reach and hold the configured pose before camera startup.

   Only if the user explicitly requests a different head pose, use `HEAD_SERVO_MODE=teach` with `--motion-approved` in a persistent interactive session. At `TEACH READY`, have the user support/position the powered head and end the turn. After explicit placement confirmation, check that the same process is still teaching, send `hold`, require `CAPTURED` and healthy feedback, and save the reported counts without stopping the controller. Preserve the 300-second timeout and fault/stop cleanup. Wuji left/right confirmation and MANUS calibration remain separate requirements.

6. **Check the final local configuration and report evidence.** Verify robot-side files and PC endpoints agree, including the actual camera key, left/right hand serial mapping and `PC_ZMQ_HOST`. Report robot-side deployment, camera discovery/frame transport and Wuji identification/motion/configuration separately. Collection additionally requires the PICO/MANUS checks in section 4C.

**Explicit failure reporting is mandatory.** If the camera or either Wuji hand fails detection or testing, tell the user immediately and include it in the final report; do not hide it under “environment installed,” “optional hardware check,” or a generic missing dependency. Name the failed device, stage, observed error/result, whether it was tested or blocked, and the next concrete action. For example: “相机检测失败：机器人端未枚举到 D455；请检查机器人上的相机 USB 和供电。相机实测未通过，配置尚未完成。” Or: “Wuji 实测失败：第二只手的拇指动作未通过位置反馈检查；左右手配置尚未确认，部署/数采环境未完成。” Distinguish unknown side from a confirmed left/right side; do not guess which hand failed. If SSH is unavailable, say the hardware is **not tested because robot access failed**, rather than claiming the camera/hands are absent. Follow cleanup results: a failure to disable a hand must be reported explicitly and requires the on-site operator to stop the hardware before further motion. Execute the applicable diagnostic/repair steps within scope, asking for physical assistance when needed; never mark hardware setup verified while a required check is failed or pending.

#### Identify devices and configure IP addresses / serial numbers

1. **Find the PC and robot addresses.** Run `ip -br -4 addr` on each host and identify the interfaces on their shared robot network. Use `ip route get <ROBOT_IP>` on the PC and `ip route get <PC_IP>` on the robot to check the route and source address, replacing the placeholders first. `ROBOT_IP` means the reachable host running the camera/hand services; `PC_IP` means the PC address reachable by those services. The Unitree body-control link can use a separate interface. In PC-local `collector/sonic/scripts/collector_pc.env`, set `ROBOT_INTERFACE` to that interface name (for example, `eno1`), not the remote robot IP. The default `real` auto-detects a local `192.168.123.*` interface; if the network differs or multiple links exist, choose the actual interface explicitly. Explain any required subnet changes before applying them; do not guess fixed PC or robot IPs.
2. **Identify the head camera by serial, then select its camera key.** After preparing the robot-side environment from the [bridge guide](../bridge/README.md#3-robot-side-camera-and-wuji-services), run this read-only discovery from the repository root on the host with the camera connected:

   ```bash
   PYTHONPATH=collector/sonic .venv_robot/bin/python - <<'PY'
   from core.camera_discovery import discover_cameras, format_camera_lines
   for line in format_camera_lines(discover_cameras()):
       print(line)
   PY
   ```

   Match the reported model and serial to the physical head camera. Set PC-local `bridge/.env` variable `WBWAM_BRIDGE_CAMERA_KEY` to its reported **key** (usually `d455`), not its serial. The camera service discovers attached cameras automatically; its launcher has no camera-serial setting. With multiple cameras of the same model, keys such as `d455_2` depend on discovery order, so verify the key-to-serial mapping against the running service before deployment and after reconnecting devices.
3. **Identify left and right Wuji hands with both USB cables connected.** Use [`discover_wuji_hands.py`](../collector/sonic/scripts/discover_wuji_hands.py). Its default mode only lists USB descriptors with system Python; the explicit `--identify` mode uses `numpy` and `wujihandpy` from the robot environment and **enables hand motors**. Prepare that environment using the bridge guide before the motion sequence. The full service-directory copy includes the script. For a standalone copy from the PC repository root:

   ```bash
   ROBOT_SSH=USER@ROBOT_IP
   scp collector/sonic/scripts/discover_wuji_hands.py "$ROBOT_SSH:discover_wuji_hands.py"
   ssh "$ROBOT_SSH" 'python3 ~/discover_wuji_hands.py'
   ```

   Replace the SSH placeholder using the user's connection information. Listing prints USB serials accepted by the hand SDK, which can differ from product serials on labels (see the [official SDK guide](https://docs.wuji.tech/docs/en/wujihandpy/latest/tutorial/)). Require exactly two distinct, readable serials. If devices are missing, check the robot USB host, power, cables, permissions, and firmware IDs (`0483:2000` or legacy `0483:7530`). Listing alone does not establish left/right.

   **The agent prepares and runs Wuji calibration.** Install/copy the script and SDK on the robot and identify competing controllers. Coordinate a controlled stop before calibration; do not interrupt an unrelated robot task without resolving the conflict with the user. Keep both USB cables connected, clear the surroundings and confirm the operator is ready. Reuse motion authorization already given; a general setup request alone does not authorize motion. Left/right is from the robot's own perspective. Reset slowly returns both hands to neutral within limits; reset movements do not identify a side. Then thumb joint 1 repeats 30-degree travel with a 4.2-second cycle on the first hand, followed by a separate second-hand stage. Do not substitute another joint or infer the second answer.

   Use a persistent interactive SSH/PTY session with writable stdin that survives assistant turns. Launch the identification command directly or via shell `exec`, so an exited program cannot leave a shell receiving late confirmations. Do not use closed stdin, `ssh -n`, pre-filled answers or `nohup`. If the host cannot retain a session across turns, give the user the manual procedure instead of maintaining an idle assistant loop. Install a missing script/SDK directly. Stop a competing service only after its role and safe stopping procedure are known and the active workflow permits it.

   **Agent runs on the robot**, from `~/WB-WAM`, after readiness and calibration authorization, with a new result filename:

   ```bash
   .venv_robot/bin/python collector/sonic/scripts/discover_wuji_hands.py \
     --identify --motion-approved --result ~/wuji-identification-01.json
   ```

   At each observation stage, let the tool yield with its session handle, ask in the final response, and end the assistant turn. The script keeps moving within its own timeout; do not poll or sleep while waiting for chat input. On the next reply, read session status/output before writing and require the same active process and expected stage. Confirmation lines are process stdin, never shell commands.

   | Program stage | Agent / user interaction |
   | --- | --- |
   | `HAND 1/2 AWAITING CONFIRMATION` | Ask which hand is moving and end the turn. After an explicit observation and stage check, send `confirm 1 left` or `confirm 1 right` plus newline. |
   | `HAND 2/2 AWAITING CONFIRMATION` | Ask again and end the turn. After a separate observation and stage check, send `confirm 2 left` or `confirm 2 right` plus newline. Never infer the opposite side. |
   | Finish | Require both confirmations, successful motor-disable cleanup and exit code 0, then validate/read the receipt using `--resolve` without `--write-env`. |

   If the user is unsure, leave the current stage looping, ask them to continue observing and end the turn. Each stage times out after 300 seconds by default and aborts with motors disabled. Do not send late answers to an exited process or another attempt. On a stop request, send `stop` plus newline or interrupt this calibration process and check cleanup. Never send EOF or stop merely to end the assistant turn. Retry only when ready, with a new result filename; do not automatically increase amplitude or timeout.

   After successful process exit, the **agent runs on the robot**:

   ```bash
   python3 collector/sonic/scripts/discover_wuji_hands.py \
     --resolve ~/wuji-identification-01.json \
     --write-env ~/WB-WAM/collector/sonic/scripts/wuji_hand_server.env
   ```

   The receipt stays local. Stop, Ctrl+C, EOF/SSH disconnection, timeout, failed motion or failed disable must not produce a completed mapping. Explain these outcomes to the user. If communication or cleanup fails, the on-site operator must establish that motion has stopped before retrying. Do not automatically increase amplitude or timeout. Identification does not require the live hand server, SONIC or the policy.

   Use the script's `--write-env` to update `LEFT_WUJI_SERIAL` and `RIGHT_WUJI_SERIAL` in **robot-local** `collector/sonic/scripts/wuji_hand_server.env`. It creates a missing file from its adjacent `.example`, preserves unrelated settings, checks shell syntax and reads back the update; serials stay out of its success output. Copy the service templates first if needed and fill other host settings separately. Do not replace this with a guessed mapping or an ad hoc editing script. Verify effective values using `load_launcher_config.sh` without launching hand services, and handle any config-path override. For the default `remote_wuji_proxy`, both hands remain attached to the robot. If `local_wuji` was explicitly selected, use the same process on the PC USB host and pass its `collector_pc.env` to `--write-env` instead.
4. **Create and edit the local configuration on each host.** Copy the `.env.example` templates with `cp -n` as shown in the bridge guide, preserving existing files. Set `COLLECTOR_PYTHON="$REPO_ROOT/.venv_robot/bin/python"` in both robot-side service files. Replace placeholders using the mapping below; the example ports must match on both ends.

   | Host / local file | Values to configure |
   | --- | --- |
   | PC: `bridge/.env` | `WBWAM_BRIDGE_CAMERA_ENDPOINT=tcp://<CAMERA_HOST_IP>:5560`, `WBWAM_BRIDGE_CAMERA_KEY=<DISCOVERED_HEAD_CAMERA_KEY>`, `WBWAM_BRIDGE_HAND_STATUS_ENDPOINT=tcp://<HAND_HOST_IP>:5559` |
   | PC: `collector/sonic/scripts/collector_pc.env` | `ROBOT_INTERFACE=<BODY_CONTROL_INTERFACE>`, `CAMERA_HOST=<CAMERA_HOST_IP>`, `CAMERA_PORT=5560`, `ROBOT_HAND_HOST=<HAND_HOST_IP>`, `ROBOT_HAND_STATUS_PORT=5559`, `PC_ZMQ_HOST=<PC_IP>`, `DEPLOY_HAND_BACKEND=remote_wuji_proxy` |
   | Camera host: `collector/sonic/scripts/camera_server.env` | `CAMERA_BIND_PORT=5560`; default `CAMERA_BIND=tcp://0.0.0.0:5560` listens locally on all interfaces |
   | Hand host: `collector/sonic/scripts/wuji_hand_server.env` | `PC_ZMQ_HOST=<PC_IP>`, `WUJI_HAND_STATUS_PORT=5559`, `LEFT_WUJI_SERIAL=<LEFT_SERIAL>`, `RIGHT_WUJI_SERIAL=<RIGHT_SERIAL>` |

   Camera and hand host IPs normally equal `ROBOT_IP`; if the services run on different hosts, use each actual address. `0.0.0.0` is a bind address, not a remote endpoint to put in `bridge/.env`. The hand service connects back to `tcp://<PC_IP>:5556`; the PC connects to camera port 5560 and hand feedback port 5559. Preserve the template's shell syntax when editing. Explicit endpoint overrides and exported variables must agree with these values, since they can override derived defaults.

Verify that placeholders are replaced and the effective camera key and left/right mapping match the discovered hardware and user observations. Require robot access, robot-side software deployment, camera discovery and real frame reception, head-servo setup, two independently observed Wuji sides, motion/cleanup success, receipt validation and configuration readback. Report commands actually executed and verified outputs; instructions alone are not evidence of success. Missing results remain **not tested / not verified**. End the turn only when a user answer or physical action is needed.

#### Environment and model checks

**Agent runs on the PC**, from the repository root, after installing/reusing `bridge/.venv-wam` and creating/filling `bridge/.env`. Resolve missing dependencies, approved assets and configuration directly before running the checks.

```bash
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml \
  --dry-run --mock-policy --fake-camera --fake-state
```

Pass: the mock completes without robot, checkpoint, or CUDA. After setting a real checkpoint, matching config/stats, Wan assets, SONIC encoder, and task prompt, validate metadata and run a bounded **no-hardware** real-model smoke:

```bash
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml --check-config
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml \
  --dry-run --fake-camera --fake-state --smoke-warmup-iters 1 --smoke-iters 1
```

Pass: metadata reports the selected files/dimensions and the real-model call completes on the chosen CUDA device. The second command may prepare a prompt embedding; check model-cache storage and network policy first. Neither command proves that camera timing, hand feedback, SONIC low-level control, or robot safety works. Keep `--dry-run` on every inference check; the dry-run path builds/checks action messages without publishing them. Use `--fake-state` to avoid needing a live body/hand controller. Real camera input may be tested directly, but never start SONIC or live hand control to supply a dry-run dependency.

For real operation, follow the [bridge deployment guide](../bridge/README.md) to configure the robot-side camera/Wuji services and native SONIC build. Confirm `task.prompt`, camera key, robot interface, decoder/planner/encoder paths, operator, clear workspace and hardware E-stop. **After authorization for the intended live operation and on-site readiness, the agent may start, monitor and stop all required services and controllers.** Keep access to each process and follow the guide's shutdown sequence; stop on user request or a failed readiness/feedback check. The PC sequence, in separate process sessions, is:

```bash
bridge/sonic/scripts/run_deploy_v4_pc.sh
# In a separate PC terminal:
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml
```

### C. SONIC/PICO or HGPT collection, and conversion

**Entry requirement for real-robot collection:** first complete and verify the shared deployment setup in section 4B, as required in section 1. If a user arrives here directly and deployment is incomplete or unverified, return to section 4B and complete that setup first. Do not begin collection-specific environment, PICO or MANUS configuration until it passes. This applies to both SONIC/PICO and HGPT; offline conversion does not require robot deployment.

#### Default SONIC/PICO collection: shared robot hardware plus PICO and MANUS

After the deployment prerequisite passes, reuse the verified robot network, service endpoints, camera/head results and Wuji mapping from section 4B. Keep live control stopped during input discovery. For collection, configure PC-local `collector_pc.env` and the robot service files directly; WB-WAM model installation, `bridge/.env`, model smokes and policy execution are not prerequisites.

Use the detailed collection setup ([English](../collector/sonic/README.md#1a-wired-pico-setup) · [中文](../collector/sonic/README_zh.md)) as the executable reference. The user supplies headset-visible information and performs wearable-device actions; the agent performs PC PICO/MANUS checks, collection input configuration and robot deployment. Start live-control processes only within the authorized motion scope after readiness checks. Ask one actionable question at a time in the final response and end the turn when dependent work requires a reply.

| Stage | User interaction | Agent work / completion evidence |
| --- | --- | --- |
| PICO wired IP | “请保持 PICO 通过网线连接 PC，告诉我头显以太网详情或 XRoboToolkit Network 面板里的 IP；如果还没有地址，告诉我目前显示什么。” | Inspect `ip -br -4 addr` and `ip route get <PICO_IP>`. Require the intended Ethernet route; its `src` is the PC's PICO-facing IP. If a direct link has no DHCP, configure the dedicated PC link as described in the collection guide and guide the user to obtain the assigned address. Do not modify the robot link or guess a fixed PICO IP. |
| PICO app and trackers | Guide app installation, tracker pairing/calibration, and ask the user to enter the detected PC address into `PC Service`. “请勾选 Head、Controller、Send，选择 Full body，并告诉我状态是否为 WORKING。” | Install/start the matching XRoboToolkit PC service. Confirm headset controls and ankle trackers, then run the bounded `probe_pico.py` below; advancing timestamps and valid head/controller/body poses are required. A successful ping or SDK import is insufficient. |
| PICO local configuration | Explain which address is the PC's and which is PICO's; ask only for missing headset-side information. | Set `XR_VIDEO_HOST` to the verified PICO wired IP and `XR_LISTEN=0.0.0.0:13579` in PC-local `collector_pc.env`. Keep `PC_ZMQ_HOST` as the PC IP reachable from the robot, and `CAMERA_HOST`/`ROBOT_HAND_HOST` as the robot service hosts. These can be different subnets. Once the collector runs, verify headset Camera Listen / Remote Vision and the logged `OPEN_CAMERA` video destination. |
| MANUS receiver | “请把 MANUS USB 无线接收器插在 PC 上，并给两只手套上电、配对。” | Run `probe_manus.py --usb-only` on the PC to enumerate VID `3325` candidates. If absent, guide USB connection checks; if access fails, inspect the repository's MANUS udev rule. USB presence alone is not glove readiness. |
| MANUS glove streams | “请戴好手套，轻轻活动双手；我会读取手套数据，不会向机器人发送动作。” | Stop competing MANUS consumers and run the full `probe_manus.py`. It identifies distinct left/right glove IDs from the Integrated SDK and requires changing valid skeleton data from both. It publishes no control commands and closes the SDK afterwards. Never use the live hand-only publisher as a receiver probe. |
| Save and validate | Ask for current-operator calibration files if they do not exist; guide SDK Client calibration if needed. | Set `HAND_CONTROL_MODE=manus`, valid per-side `MANUS_*_CALIBRATION_FILE` paths and `MANUS_LOAD_CALIBRATION=1`. Preserve other local settings and read them back. Glove IDs are SDK-discovered; do not put MANUS IDs in Wuji serial fields or invent a receiver IP. |

Run input checks on the **PC**, from the repository root, after the selected environment is installed:

```bash
# This USB-only check also works with system Python before environment setup.
python3 collector/sonic/scripts/probe_manus.py --usb-only
.venv_teleop/bin/python collector/sonic/scripts/probe_pico.py --duration-s 12
.venv_teleop/bin/python collector/sonic/scripts/probe_manus.py --duration-s 12
```

**Required MANUS pass criteria:** the full probe exits with code 0, reports `changing skeleton data received` for **both** `left` and `right` with distinct glove IDs, and prints `PASS: both glove streams detected`. Inspect the output as well as the exit code. Report pairing/connection and live data as verified only after these conditions pass; report operator calibration separately. If the check fails, mark **collection blocked: MANUS glove-data check pending**, guide the user through power, pairing, permissions or movement as appropriate, and end the turn when their action/answer is needed. Rerun the probe after the user replies, before starting collection; do not repeatedly probe to keep the turn alive. This input check publishes no robot actions and must not be replaced with a live hand-control test.

If PICO reports a Wi-Fi address despite the cable, resolve the wired address/route instead of silently switching to Wi-Fi. If MANUS data is absent, static, one-sided, or reports SDK/license errors, retain a pending status and guide the relevant connection/pairing/permission check. Missing calibration remains pending even when discovery passes. Do not mark collection setup complete until the shared robot hardware checks, wired PICO tracking, intended camera return, both MANUS streams (when selected), and configuration/calibration checks pass. After authorization for the live collection smoke and on-site readiness, the agent may start and monitor body/hand control and recording, then verify the output and stop/clean up the processes.

#### Collection launch checks

For SONIC/PICO, install `.venv_teleop`, inspect its `scripts/env/check_envs.sh` section including `ManusServer`, and configure a local copy of `collector_pc.env.example`. If native SONIC deployment is selected, also run `python check_environment.py --deploy` from `tracker/sonic/` in the selected SONIC environment and review each native/TensorRT/LFS result. Before touching hardware, print the startup sequence:

```bash
cp -n collector/sonic/scripts/collector_pc.env.example collector/sonic/scripts/collector_pc.env
collector/sonic/scripts/run_data_collection_flow.sh print
```

Pass: the selected config, interfaces, hand mode, camera/feedback endpoints, and output directory are correct. The default collector records 20 Hz `collector_timer`; do not silently switch to 50 Hz, because the public converter accepts only 20 Hz raw episodes. After completing the hardware checks and configuration in the [SONIC collector guide](../collector/sonic/README.md), confirming on-site readiness and obtaining authorization for live collection, the agent starts these commands in separate process sessions, in this order:

```bash
# Robot terminals, one command in each:
collector/sonic/scripts/run_camera_server.sh --head-motion-approved
collector/sonic/scripts/run_wuji_hand_server.sh
# PC terminals, one command in each:
tracker/sonic/gear_sonic_deploy/scripts/run_deploy_pc.sh
collector/sonic/scripts/run_pico_manager.sh
collector/sonic/scripts/run_collector_pc.sh
```

The Wuji service enables motors; this is not a safe no-hardware smoke.

For HGPT, create/reuse a separate Python 3.11/3.12 `h-gpt` environment, follow the [HGPT controller setup](../tracker/humanoid_gpt/README.md), and supply existing tracking/walking checkpoints. After approval, a default MANUS setup example is:

```bash
conda create -n h-gpt python=3.12
conda activate h-gpt
cd tracker/humanoid_gpt
scripts/setup_pico_env.sh --checkpoint-source /path/to/hgpt/ckpts --with-manus
scripts/setup_pico_real_env.sh --skip-pico-setup  # Only if real G1 control was selected.
```

Omit `--with-manus` for another selected hand source; use `--with-tensorrt` on the real-environment setup only when that provider was selected. The HGPT collection template defaults to TensorRT: if the user chose the CPU provider instead, set `HGPT_POLICY_PROVIDER=cpu` in its local config and check that performance is adequate. From `tracker/humanoid_gpt/`, check the controller and run a bounded **no-robot, no-PICO** simulator smoke:

```bash
python scripts/check_pico_env.py --service-mode external --require-manus
scripts/run_pico_sim.sh --headless --no-mocap --max-steps 500 --no-visualize-retarget
python scripts/check_pico_real_env.py --policy-provider cpu  # Only after real-environment setup.
```

Omit `--require-manus` when MANUS was not installed; use the selected real policy provider for the last check. If testing live PICO in simulation, prepare its Robotics Service separately and select the matching service mode. From the repository root, check the HGPT collector separately:

```bash
cp -n collector/humanoid_gpt/scripts/hgpt_collection.env.example \
  collector/humanoid_gpt/scripts/hgpt_collection.env
python collector/humanoid_gpt/scripts/check_env.py
```

Pass: required imports and setup checks succeed; missing optional local-camera support is reported as such. The [HGPT collection guide](../collector/humanoid_gpt/README.md) contains the complete multi-terminal startup and 20 Hz conversion details. After confirming on-site readiness and authorization for both body and hand actuation, the agent may execute this sequence on the PC in separate process sessions:

```bash
# Separate PC terminal, from tracker/humanoid_gpt/:
scripts/run_hgpt_pose_manager.sh --hand-source manus \
  --pico-service-mode external --publish-wuji-hand
# Separate PC terminal, from tracker/humanoid_gpt/:
scripts/run_pico_real.sh --net ROBOT_INTERFACE \
  --pose-endpoint tcp://127.0.0.1:5556 --state-action-bind tcp://127.0.0.1:5558 \
  --publish-lowcmd
# Separate PC terminal, from the repository root:
collector/humanoid_gpt/scripts/run_collector_pc.sh --task-name YOUR_TASK
```

Start the configured robot-side camera/Wuji services first. `--publish-lowcmd` enables body commands; omitting it does **not** disable hand commands. The agent may start, monitor and stop both actuation paths after the corresponding authorization and readiness checks. Track both processes and verify cleanup; body-only flags do not disable hand motion.

For offline SONIC conversion, use a separate CPU environment and a **new** output directory, after the user identifies 20 Hz raw input:

```bash
python3 -m venv .venv-process
.venv-process/bin/python -m pip install -r collector/sonic/processing/requirements.txt
.venv-process/bin/python collector/sonic/processing/convert_to_lerobot.py \
  --input /path/to/raw_task --output /path/to/new_archive --limit-episodes 1 --dry-run
.venv-process/bin/python collector/sonic/processing/convert_to_lerobot.py \
  --input /path/to/raw_task --output /path/to/new_archive --limit-episodes 1
.venv-process/bin/python collector/sonic/processing/validate_lerobot.py \
  --root /path/to/new_archive
```

Run the writing conversion only after the preview and user approval. Pass: a record is published and the independent validator succeeds. The converter does not overwrite unrelated output. For a full export, use another new output directory without `--limit-episodes`; see the [format and safeguards](../collector/sonic/processing/README.md). Optional SONIC MuJoCo uses `.venv_sim`; check its section of `scripts/env/check_envs.sh` and follow the [SONIC guide](../tracker/sonic/README.md) for the selected simulation.

### D. HumanoidArena evaluation

Evaluation uses task-specific post-trained checkpoints. Do not include pretrain/midtrain downloads or training datasets in evaluation setup questions or commands. Ask only for the chosen evaluation task and its checkpoint choice initially; collect remaining asset paths and runtime choices when needed.

Set up the separate Isaac environment, pinned dependencies, public models, and released simulator assets according to the WB-WAM HumanoidArena environment setup guide ([English](../benchmark/humanoidarena/environment.md) · [中文](../benchmark/humanoidarena/environment_zh.md)). Treat that guide as the installation source of truth; the upstream HumanoidArena documentation may provide background but must not replace its pinned versions or paths. The WB-WAM policy server reuses the training `wbwam` environment. After all environment-guide checks pass, use the [benchmark guide](../benchmark/humanoidarena/README.md) for evaluation behavior and commands. Copy the runtime template without overwriting an existing local config:

```bash
cp -n benchmark/humanoidarena/configs/runtime.example.yaml benchmark/humanoidarena/runtime.local.yaml
```

Replace every placeholder; check `inference_python`, `simulation_python`, GPU IDs, checkpoint directory, SONIC release, Isaac paths, and writable cache/output. Then, only after a one-episode GPU/simulator smoke is approved:

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task hammer \
  --checkpoint /absolute/path/to/checkpoints/humanoid_arena_native50/hammer \
  --output /absolute/path/to/results/smoke \
  --seeds 0 --repeats 1
```

Pass: one trial JSON, one MP4, `server_health.json`, and a summary are written without runtime errors. A fall or timeout is an episode outcome, not an environment failure. The official run uses seeds `0,1,2` with 20 repeats per seed; after separate compute approval, a single-task example is:

```bash
/absolute/path/to/wbwam/bin/python benchmark/humanoidarena/run_eval.py \
  --config benchmark/humanoidarena/runtime.local.yaml \
  --task hammer \
  --checkpoint /absolute/path/to/checkpoints/humanoid_arena_native50/hammer \
  --output /absolute/path/to/results/native50 \
  --seeds 0,1,2 --repeats 20
```

For all seven tasks and resumable runs, use the [benchmark guide](../benchmark/humanoidarena/README.md#6-run-all-seven-tasks).

## 5. Report back

For deployment and real-robot collection, report commands actually executed and their verified results. If a manual fallback was needed, distinguish user-supplied output from agent-executed checks; never infer success from instructions alone. Always state separate **pass / failed / not tested** results for robot SSH access, robot-side service/environment deployment, head-camera discovery and real image reception, head-servo setup, Wuji USB discovery, both hands' motion/feedback test, and confirmed serial configuration. Explain every camera/Wuji failure or untested item and the next action in plain language, without exposing credentials or raw device identifiers. PC-only success must never be presented as completed robot deployment or collection setup.

Give the user a compact table of each requested workflow: environment and version, downloaded/reused assets with paths, config files changed, checks and smokes run with pass/fail evidence, skipped tests, and next command. Distinguish **environment ready**, **model smoke passed**, **simulator ran**, and **robot validated**. If a prerequisite or permission is missing, stop at that boundary and ask; do not silently choose a different model, instruction, dataset, GPU allocation, or robot-control mode.

For deployment, explicitly report left/right hand identification as **complete** or **pending**, including whether the reset/thumb sequence, two independent user-observed sides, saved-result/device verification, and robot-side configuration readback passed, without including raw serials or credentials. If any of these is pending, say **software preparation complete; deployment setup incomplete** when applicable, and name the next command or result needed from the user. A successful mock or model smoke cannot replace this required configuration step.

For real-robot SONIC/PICO or HGPT collection, first report the deployment prerequisite as **passed / failed / not verified**. Do not declare collection setup ready while deployment remains incomplete. For SONIC/PICO, additionally report the shared robot/Wuji checks, PICO wired route and tracking check, headset camera return, MANUS USB detection, both glove streams, and per-user calibration as complete/pending. Do not expose actual IPs or device identifiers in the report. Name the next user action for pending items; successful dependency installation alone does not mean collection setup is complete.
