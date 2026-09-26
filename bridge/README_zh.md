# WB-WAM 真机部署

部署或操作真实机器人前，请阅读[安全免责声明](../README_zh.md#安全免责声明)。

**头部舵机（必需）：** 部署/采集前完成[头部驱动安装、机器人端编译和测试](../collector/sonic/head/README_zh.md)。PC 仓库根目录运行 `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP`。推理和收数据时，机器人端使用 `run_camera_server.sh --head-motion-approved` 同时启动头部舵机保持服务与相机；默认使用 `HEAD_SERVO_MODE=raw`，移动到 `HEAD_JOINT0_ENCODER=3027`、`HEAD_JOINT1_ENCODER=1849` 并保持（单位为编码器计数，不是角度）。环境示例已提供这些默认值，无需头部标定或首次示教。舵机通信失败时不能只启动图像服务并宣告就绪。

[English](./README.md)

**硬件适配范围：** 当前部署仅支持配备[舞肌灵巧手](https://www.wuji.tech/zh/hand)和[宇树 G1 双自由度相机头部模组（RealSense D455）](https://www.unitree.com/cn/robocup)的 Unitree G1；其他手部或相机配置暂不支持。本页不涉及数据采集。除标明在机器人端执行的命令外，均从 PC 的仓库根目录运行。PC 需要 Linux、CUDA GPU 和 `uv`。

## 硬件接线与配置

| 设备与接线 | 真机部署 |
| --- | --- |
| 宇树 G1 ↔ PC | 网线连接，配置 PC 上连接机器人的网卡 |
| 头部相机 → 机器人 | USB 直连机器人，相机服务在机器人上运行 |
| 左右 Wuji 手 → 机器人 | USB 直连机器人，手部服务在机器人上运行 |

准备机器人 SSH 连接和 PC 连接机器人的网卡信息，确认头部相机 key、左右 Wuji 手序列号及 PC 服务地址。策略部署不需要 PICO 或 MANUS；采集所需的额外设备与输入检查见[数据采集说明](../collector/sonic/README_zh.md)。

**默认接线方式：** 机器人通过网线连接 PC，左右 Wuji 手和头部相机均通过 USB 直接连接在**机器人上**，相机和手部服务也在机器人上运行。请把独立的 [Wuji USB 序列号识别程序](../collector/sonic/scripts/discover_wuji_hands.py)复制到机器人，再用 `python3` 运行；在 PC 上运行无法通过网线枚举机器人上的 USB 设备。左右手确认是部署配置必做项：双手保持连接，运行识别程序复位双手，第一只手的拇指持续慢速小幅往复，确认其左右后才切换到第二只手，再独立确认第二只手。确认后，将 USB 序列号填入机器人端的 `collector/sonic/scripts/wuji_hand_server.env`。复制、识别和配置步骤见下文[机器人端相机与舞肌服务](#3-机器人端相机与舞肌服务)。

```text
机器人相机、身体与手部状态 → PC WB-WAM → SONIC Encoder → PC SONIC → 机器人
```

## 1. PC 环境与无硬件检查

检查 `third_party/GMR` 和 `third_party/HumanoidArena` 是否已克隆；如有缺失，在仓库根目录执行：

```bash
git submodule update --init --recursive
```

```bash
bridge/scripts/setup_env.sh
cp -n bridge/.env.example bridge/.env
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml \
  --dry-run --mock-policy --fake-camera --fake-state
```

最后一条命令不需要模型权重或机器人。

按 `tracker/sonic/docs/source/getting_started/installation_deploy.md` 中的 SONIC 原生部署说明配置本仓库的 `tracker/sonic/gear_sonic_deploy/`，构建 `g1_deploy_onnx_ref_vla`。真机还需要 SONIC encoder、decoder、observation config 和 planner 资产。

## 2. 部署配置

修改 `bridge/.env`：设置 WB-WAM checkpoint、SONIC encoder、文本缓存、机器人地址和相机 ID。开源权重的下载方法见[训练文档](../training/README_zh.md)；将 `WBWAM_BRIDGE_CHECKPOINT_PATH` 改为实际权重路径。同一个 checkpoint 必须配套对应的 `config.yaml` 和 `dataset_stats.json`，默认从权重旁查找；SONIC encoder 需另外准备。

```bash
cp -n collector/sonic/scripts/collector_pc.env.example \
  collector/sonic/scripts/collector_pc.env
```

当前 SONIC 启动脚本仍从 `collector_pc.env` 读取机器人网络接口及原生 decoder/planner 路径。在 [deploy.yaml](configs/deploy.yaml) 中修改 `task.prompt`，核对图像尺寸与控制参数。`bridge/.env` 的相对路径以仓库根目录为基准，已导出的环境变量优先。

配置完成后检查 checkpoint 的训练配置和统计量，无需加载模型权重：

```bash
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml --check-config
```

## 3. 机器人端相机与舞肌服务

**默认接线方式：** PC ↔ 网线 ↔ 机器人；左右 Wuji 手和头部相机都通过 USB 直接连接在机器人上。USB 序列号识别、相机和手部服务均在机器人端运行；PC 无法通过网线直接枚举这些 USB 设备，此接线方式使用 `remote_wuji_proxy`。

这两个服务当前复用 `collector/sonic/` 中的实现。首次配置机器人时，在 PC 上复制完整服务目录，不能只复制两个启动脚本：

```bash
ROBOT_SSH=USER@ROBOT_IP
ssh "$ROBOT_SSH" 'mkdir -p "$HOME/WB-WAM/collector/sonic"'
rsync -av --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='datasets/' --exclude='scripts/*.env' \
  collector/sonic/ "$ROBOT_SSH:WB-WAM/collector/sonic/"
ssh "$ROBOT_SSH" 'cd WB-WAM && \
  cp -n collector/sonic/scripts/camera_server.env.example collector/sonic/scripts/camera_server.env && \
  cp -n collector/sonic/scripts/wuji_hand_server.env.example collector/sonic/scripts/wuji_hand_server.env'
```

上述复制操作已包含独立的序列号识别程序。**在机器人上**用系统 Python 运行，无需先安装 SDK：

```bash
cd ~/WB-WAM
python3 collector/sonic/scripts/discover_wuji_hands.py
```

不带参数时，程序只读取 Linux USB 设备信息，不创建手部 SDK 连接，也不使能电机；下面的 `--identify` 模式会驱动手部。输出包括 USB 端口和 **USB 序列号**；`wujihandpy.Hand` 使用这个 USB 序列号，标签上的产品序列号可能不同，详见[舞肌 SDK 序列号说明](https://docs.wuji.tech/docs/zh/wujihandpy/latest/tutorial/)。程序识别 `0483:2000` 和旧版 `0483:7530` 两种 USB ID。若未找到设备，请检查是否在机器人上运行、手部供电、USB 接线及 `lsusb` 输出。

机器人端从 `~/WB-WAM` 安装独立环境；不需要复制 PC 虚拟环境或模型权重：

```bash
cd ~/WB-WAM
sudo apt update
sudo apt install -y python3-venv libusb-1.0-0 usbutils curl
python3 -m venv .venv_robot
.venv_robot/bin/python -m pip install --upgrade pip
.venv_robot/bin/python -m pip install \
  'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' \
  pyzmq pyrealsense2 wujihandpy
```

在机器人端的 `camera_server.env` 和 `wuji_hand_server.env` 中都设置 `COLLECTOR_PYTHON="$REPO_ROOT/.venv_robot/bin/python"`。核对相机端口；在手部配置中填写 PC 地址 `PC_ZMQ_HOST` 和左右手序列号。首次使用还需确认 RealSense 与舞肌设备的 USB 权限和 SDK 正常工作。机器人网络不要暴露到互联网。

### 双手保持连接，通过拇指动作确认左右手（必做）

安装好上述机器人端环境后，保持双手通过 USB 连接机器人。停止其他手部控制程序，确认手部周围无障碍，并在现场观察。在机器人上执行以下命令；`--motion-approved` 表示已准备好进行复位和拇指动作：

```bash
.venv_robot/bin/python collector/sonic/scripts/discover_wuji_hands.py \
  --identify --motion-approved --result ~/wuji-identification-01.json
```

程序先将双手缓慢复位到关节限位内的张开位置，然后提示 `RESET DONE`。第一只手的拇指 joint 1（从掌心往指尖数第二个关节）随后**持续慢速、小范围往复运动，直到确认才切换到第二只手**。幅度 30°（约 0.524 rad），伸出和返回各约 1.8 秒，两端各停留 0.3 秒，一次往返约 4.2 秒；其余关节保留张开目标。这里的复位是回到张开姿态，不是重新标定或清除故障。**忽略复位时的运动，观察当前提示的识别阶段。**

保持识别终端打开，在同一程序的标准输入中分两次确认。左右以机器人自身为准，面对机器人时，它的左手在观察者右侧：

1. 出现 `HAND 1/2 AWAITING CONFIRMATION` 后，观察持续运动的是哪只手。左手输入 `confirm 1 left`，右手输入 `confirm 1 right`，然后回车。没看清时继续观察，无需重新运行。
2. 程序收到确认后完成当前往返，回到张开位置并关闭第一只手电机，才开始第二只手的循环。出现 `HAND 2/2 AWAITING CONFIRMATION` 后，再次观察并输入 `confirm 2 left` 或 `confirm 2 right`。**必须实际观察并分别确认两只手，不能提前输入第二个答案或仅根据第一个答案推断。** 两次确认的左右必须不同。

第二次确认后的当前往返完成、电机关闭成功后，才保存结果。输入 `stop`、按 Ctrl+C、控制输入断开或任一阶段超过默认 300 秒均中止识别，尝试关闭电机，不保存完成结果，也不会自动切换到下一只手。可用 `--stage-timeout` 调整每阶段等待时间，最多 600 秒；超时后如需重试，使用新的结果文件名。通过 SSH 使用时需保持交互终端，例如 `ssh -tt <机器人SSH地址>`，不要使用 `ssh -n`、后台运行或预先管道输入确认。

两次确认均完成、识别程序成功退出后，校验本次结果并保存机器人端配置（以下命令不发送动作）：

```bash
python3 collector/sonic/scripts/discover_wuji_hands.py \
  --resolve ~/wuji-identification-01.json \
  --write-env collector/sonic/scripts/wuji_hand_server.env
```

程序核对当前双手与本次保存的两次独立确认后，只更新配置中的 `LEFT_WUJI_SERIAL` 和 `RIGHT_WUJI_SERIAL`，保留其他设置，并检查 shell 语法及写入后内容；不会输出序列号。配置不存在时，从相邻的 `wuji_hand_server.env.example` 创建，其他机器参数仍需按本节填写。省略 `--write-env` 可只输出序列号赋值而不修改文件。未完成两次确认与配置核对，部署环境配置仍未完成。旧版只记录动作顺序的结果不再接受，须重新识别；禁止根据旧记录或 USB 枚举顺序猜测左右。动作失败、确认不完整或设备变化时不生成映射，也不写入配置。

如需单独观察、调整拇指运动，可在识别程序退出后运行双手同步测试；这不会修改已确认的左右手映射：

```bash
.venv_robot/bin/python collector/sonic/scripts/discover_wuji_hands.py \
  --tune-thumbs --motion-approved --thumb-joint 1 --amplitude-deg 30 --period 4.2
```

程序每轮报告两只手的实际角度行程和往返周期。输入 `stop` 停止并关闭电机，然后用新的 `--amplitude-deg`、`--period` 重启。幅度表示一个拇指关节从张开位置到弯曲位置的总行程，不是正负两个方向各该角度。参数仍须满足双手关节限位及 0.3 rad/s 的速度限制；过快组合会被拒绝，应延长周期。双手同步测试不能代替上述逐只确认流程。

在机器人上分别打开两个终端：

```bash
collector/sonic/scripts/run_camera_server.sh --head-motion-approved
```

```bash
collector/sonic/scripts/run_wuji_hand_server.sh
```

手部服务会使能电机；启动前保持双手周围无人和障碍物，确认两个手部初始化均成功。

## 4. PC 端启动与操作

终端 1，启动 SONIC：

```bash
bridge/sonic/scripts/run_deploy_v4_pc.sh
```

终端 2，启动 WB-WAM policy：

```bash
bridge/scripts/run_bridge.sh --config bridge/configs/deploy.yaml
```

在 policy 终端按 `4 → 1 → 1 → 4 → 1` 操作：

- `4`，再按 `1`：预热推理，然后执行按键后新生成的动作片段。
- 运行中按 `1`：暂停 policy 动作，由 Planner 控制身体，双手逐渐张开。
- 暂停后按 `4`：回到初始姿态并再次预热。
- 回位完成后按 `1`：用新生成的动作片段继续；不能从暂停状态直接恢复执行。

**注意：从停止策略（`1`）到归位（`4`）时，机器人动作速度较快。归位前务必确认双手整个运动路径内没有人员或物品，确保双手不会碰撞任何东西。**

`e` 发出软件停止并退出；`q` 回到 Planner 空闲状态并退出。执行前确认身体和手部反馈持续更新、工作区域安全，并备好硬件急停。键盘停止**不是**硬件急停。
