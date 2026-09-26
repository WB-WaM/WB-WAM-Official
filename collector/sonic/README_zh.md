# SONIC/PICO 数据采集

部署或操作真实机器人前，请阅读[安全免责声明](../../README_zh.md#安全免责声明)。

**头部舵机（必需）：** 部署/采集前完成[头部驱动安装、机器人端编译和测试](head/README_zh.md)。PC 仓库根目录运行 `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP`。推理和收数据时，机器人端使用 `run_camera_server.sh --head-motion-approved` 同时启动头部舵机保持服务与相机；默认使用 `HEAD_SERVO_MODE=raw`，移动到 `HEAD_JOINT0_ENCODER=3027`、`HEAD_JOINT1_ENCODER=1849` 并保持（单位为编码器计数，不是角度）。环境示例已提供这些默认值，无需头部标定或首次示教。舵机通信失败时不能只启动图像服务并宣告就绪。

本页只介绍用 PICO 遥操作、相机和舞肌手采集同步数据；不启动 WB-WAM 模型推理。除标明机器人端的命令外，均从 PC 的仓库根目录运行。长时间运行的服务各占一个终端。

## 硬件接线与配置

| 设备与接线 | SONIC/PICO 数据采集 |
| --- | --- |
| 宇树 G1 ↔ PC | 网线连接，配置 PC 上连接机器人的网卡 |
| 头部相机 → 机器人 | USB 直连机器人，相机服务在机器人上运行 |
| 左右 Wuji 手 → 机器人 | USB 直连机器人，手部服务在机器人上运行 |
| PICO ↔ PC | **默认网线连接**，通过头显兼容的以太网转接器连接 PC 网口或同一有线网络 |
| MANUS 手套 → 无线接收器 → PC | **USB 无线接收器插在 PC 上**，手套与接收器配对 |

采集复用真机部署的机器人、相机与 Wuji 硬件检查，额外配置 PICO 和 MANUS；不需要安装 WB-WAM 策略环境或下载策略权重。记录 **PICO 有线 IP** 和 **PICO 能访问的 PC IP**。PICO 没有通用固定 IP，应从头显网络详情读取并核对实际有线路由。具体 IP 配置、头显设置、PICO/MANUS 检测命令和手套校准见下文 1A、1B。

**默认接线方式：** 机器人通过网线连接 PC，左右 Wuji 手和头部相机均通过 USB 直接连接在**机器人上**，相机和手部服务也在机器人上运行。请把独立的 [Wuji USB 序列号识别程序](scripts/discover_wuji_hands.py)复制到机器人，再用 `python3` 运行；在 PC 上运行无法通过网线枚举机器人上的 USB 设备。左右手确认是数采配置必做项：双手保持连接，运行识别程序复位双手，第一只手的拇指持续慢速小幅往复，确认其左右后才切换到第二只手，再独立确认第二只手。确认后，将 USB 序列号填入机器人端的 `collector/sonic/scripts/wuji_hand_server.env`。复制、识别和配置步骤见[真机部署说明](../../bridge/README_zh.md)。采集额外使用的 PICO 默认通过网线连接 PC，MANUS USB 无线接收器插在 PC 上；配置及必做数据检查见下文 1A、1B。

```text
PICO / 手部输入 → PC Pose Manager → PC SONIC 底层控制 → 机器人
                         └──────────────→ 机器人舞肌服务
相机、身体与手部反馈 ───────────────────────→ PC Collector → episode
```

## 1. 配置 PC

安装采集环境，并从模板创建本机配置：

```bash
scripts/env/setup_envs.sh teleop
cp -n collector/sonic/scripts/collector_pc.env.example \
  collector/sonic/scripts/collector_pc.env
```

teleop 安装会为当前 Python 编译 MANUS 绑定；主机需具备 C++ 编译器、`make` 和 ncurses 开发文件。示例配置默认使用 `HAND_CONTROL_MODE=manus`。

修改 `collector_pc.env` 中的任务名 `TASK_NAME`、输出目录 `OUTPUT_ROOT`、机器人网络接口 `ROBOT_INTERFACE`、相机 `CAMERA_HOST/CAMERA_PORT`、手部反馈地址、`PC_ZMQ_HOST` 和 `HAND_CONTROL_MODE`。默认 PICO 通过网线连接 PC，`XR_LISTEN` 应能被 PICO 访问，`XR_VIDEO_HOST` 填经过核对的 PICO 有线 IP。PC 上的 pose/body 连接使用本地端口 5556/5558；相机和手部服务地址要与机器人端配置一致。真实 `.env` 不提交 Git。

## 1A. PICO 有线连接与 IP 配置

默认 PICO 通过兼容的以太网转接器和网线连接 PC，或与 PC 接在同一有线网络；MANUS USB 无线接收器也插在 PC 上。机器人通过自己的有线链路连接 PC，相机和 Wuji 双手仍通过 USB 接机器人。采集必须完成[部署指南](../../bridge/README_zh.md)中的相机、网络和 Wuji 左右手硬件确认，但不要求安装 WB-WAM policy 环境或下载模型。

1. **确认有线地址。** 从头显以太网详情 / XRoboToolkit 的 Network 面板读取 PICO IP；在 PC 运行 `ip -br -4 addr`，再运行 `ip route get <PICO_IP>`，确认走的是连接 PICO 的有线网卡，并记录输出中的 `src` 作为 PC 面向 PICO 的地址。如果 Wi-Fi 同时开启，以实际有线地址和路由为准。`ping -c 3 <PICO_IP>` 可检查连通性，但不能替代应用数据检查。若面板显示的是 Wi-Fi 地址，可用已有的 ADB 连接运行 `adb shell ip -4 addr show` 辅助核对以太网地址；不需要为了常规采集一直插拔 USB。
2. **直连时分配地址。** 两端必须在同一子网，且有不同地址。若有线网络已有 DHCP，使用其分配的地址；单根网线直连并不会自动产生 DHCP 服务。在使用 NetworkManager 的 PC 上，若 PICO 专用网口尚无可用配置，先核对网卡和现有路由，再为该网口创建独立共享连接（把占位符替换为实际网卡名）：

   ```bash
   nmcli connection add type ethernet ifname '<PICO_INTERFACE>' \
     con-name wbwam-pico ipv4.method shared ipv6.method disabled
   nmcli connection up wbwam-pico
   nmcli -g IP4.ADDRESS device show '<PICO_INTERFACE>'
   ```

   PICO 端使用自动获取地址 / DHCP，再读取实际分配的 IP。复用现有可用连接，不要重复创建同名配置或改动连接机器人的网卡。若需要静态地址，根据已有网段选择互不冲突的 PC/PICO 地址和掩码，在设备支持的以太网设置中填写；不要套用截图中的地址。PC 多网卡时，机器人链路和 PICO 链路可以是不同子网，分别核对路由。
3. **安装服务和头显应用。** 按[XRoboToolkit 配置说明](../../tracker/sonic/docs/source/getting_started/vr_teleop_setup.md)安装与 PC 系统/架构匹配的 PC 服务和仓库所用 PICO 应用，启用头显开发者模式并安装 APK。下载 APK 时可临时使用互联网连接，正式采集默认走网线。在 PC 启动已安装的 XRoboToolkit 服务（标准安装入口为 `/opt/apps/roboticsservice/runService.sh`，先确认路径存在），头显应用的 `PC Service` 填 **PC 面向 PICO 的有线 IP**，点击 `Enter` / `Reconnect`，确认状态 `WORKING`。
4. **配置追踪。** 配对左右控制器和两个脚踝追踪器，按头显校准界面完成身体校准。在 XRoboToolkit 勾选 `Head`、`Controller`、`Send`，将 `Pico Motion Tracker` 设为 `Full body`。默认手指数据来自 MANUS。采集前佩戴好设备，再运行 `.venv_teleop/bin/python collector/sonic/scripts/probe_pico.py`；只有头显、双控制器、24 个身体关节姿态有效且时间戳持续推进才通过。此程序不启动 SONIC、不发布动作。
5. **填写 PC 配置并验证回传。** 编辑本机 `collector/sonic/scripts/collector_pc.env`：

   | 字段 / 位置 | 应填写的值 |
   | --- | --- |
   | PICO 应用 `PC Service` | PC 面向 PICO 的有线 IP，不是 PICO IP，也不是机器人 IP |
   | `XR_VIDEO_HOST` | PICO 自身的有线 IP，确保视频回传使用网线 |
   | `XR_LISTEN` | 默认 `0.0.0.0:13579`，是 PC 的相机请求监听地址；头显连接时使用 PC 的实际 IP |
   | `PC_ZMQ_HOST` | 机器人能访问的 PC IP；可能与 PICO 链路的 PC IP 不同 |
   | `CAMERA_HOST` / `ROBOT_HAND_HOST` | 机器人端相机 / 手部服务所在主机 IP |

   `XR_LISTEN` 属于采集器的相机请求服务，不是替代 XRoboToolkit PC 服务的地址。采集器启动后，在头显 Remote Vision / Camera Listen 中打开相机回传，核对 PC 日志的 `OPEN_CAMERA` 和 `stream_ip` 是否对应 PICO 有线地址，并在头显确认画面。具体按钮布局以所安装应用为准。PICO 网络变化后更新 `XR_VIDEO_HOST` 并重测。

## 1B. MANUS USB 接收器自动识别与手套检查

**数采必做项：** 使用 MANUS 进行数采前，必须执行下面的左右手套完整数据检查，首次配置和后续每次采集会话均适用。当前会话、硬件状态未改变时，刚完成的完整检查可以复用；接收器或手套重新连接、断电、重新配对或出现数据故障后必须重测。仅检测到 USB、导入 SDK 成功或以前检查通过，不能替代本次检查。

将 MANUS 无线接收器插入 **PC 的 USB 口**，给两只手套上电并与接收器配对。先运行：

```bash
python3 collector/sonic/scripts/probe_manus.py --usb-only
```

该步骤按本仓库 MANUS USB 厂商 ID `3325` 枚举候选设备、USB 路径和可读取的标识；不按 USB 顺序推断左右。若未发现设备，确认接收器插在 PC，检查 `lsusb`。权限不足时，检查仓库的 `tracker/sonic/decoupled_wbc/docker/70-manus-hid.rules`；需要安装规则时，将其安装到 `/etc/udev/rules.d/` 并重新加载 udev，避免为排查而运行机器人控制程序。

完成 `scripts/env/setup_envs.sh teleop` 后，停止其他占用 MANUS 的采集进程；戴好手套并轻轻活动双手，运行：

```bash
.venv_teleop/bin/python collector/sonic/scripts/probe_manus.py --duration-s 12
```

必须核对退出码为 0，且输出中左右两侧均有 `changing skeleton data received`、手套 ID 不同，并出现 `PASS: both glove streams detected`。通过后才能确认配对/连接及数据正常；当前操作者的校准仍需单独验证。未通过时禁止启动数采，检查供电、配对、权限并活动双手后重测，不得跳过检查。

程序使用仓库的 `ManusServer` Integrated SDK 自动取得左右手套 ID，要求两侧都出现变化的有效骨架数据；结束后关闭 SDK。它不发布机器人命令、不振动手套、不加载或修改校准。只有 USB 接收器、单侧数据或重复缓存均不能通过。Integrated 模式的配对和校准方法见 [MANUS 官方说明](https://docs.manus-meta.com/3.1.0/Plugins/SDK/getting%20started/)。SDK 报许可证或配对错误时按实际错误处理，不把 USB 可见等同于 SDK 可用。

通过后设置 `HAND_CONTROL_MODE=manus`。左右 ID 来自 SDK，不需要给无线接收器配置 IP，也不需要把它的 USB 标识填入 Wuji 序列号字段。使用当前操作者对应的左右手 `.mcal` 校准文件，填写 `MANUS_LEFT_CALIBRATION_FILE`、`MANUS_RIGHT_CALIBRATION_FILE`，保持 `MANUS_LOAD_CALIBRATION=1`。若尚无校准，先在 MANUS SDK Client 中按左右手分别完成并保存，再核对文件与操作者/手套对应关系；自动识别不替代校准。`save_manus_calibration.sh left/right <source.mcal>` 可复制校准文件，但会覆盖项目中的对应文件，使用前核对已有文件。不要用会发送动作的 `run_manus_hand_only_test.sh` 做接收器识别。

## 2. 配置机器人端相机和舞肌服务

默认接线为 PC ↔ 网线 ↔ 机器人，左右 Wuji 手和头部相机均通过 USB 直接连接在机器人上。序列号识别程序 `scripts/discover_wuji_hands.py` 必须复制到机器人并在机器人上运行，PC 无法通过网线枚举这些 USB 设备。完成下面的复制后，在机器人的 `~/WB-WAM` 下运行 `python3 collector/sonic/scripts/discover_wuji_hands.py`；按[真机部署说明](../../bridge/README_zh.md)保持双手连接，复位后先让第一只手的拇指持续慢速小幅往复，确认其左右后再切换到第二只手并独立确认，将对应的 USB 序列号填入机器人端的 `wuji_hand_server.env`。

首次安装时，从 PC 将完整的服务目录复制到机器人；仅复制启动脚本不够。机器人只运行相机和舞肌服务，不需要 PC 虚拟环境或模型权重。以下示例使用机器人上的 `~/WB-WAM`：

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

机器人端安装独立 Python 环境，不复制 PC 的虚拟环境：

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

在机器人的 `camera_server.env` 和 `wuji_hand_server.env` 中都设置 `COLLECTOR_PYTHON="$REPO_ROOT/.venv_robot/bin/python"`。核对相机端口；填写 `PC_ZMQ_HOST` 与左右手序列号。相机与舞肌 SDK/USB 权限必须正常；首次配置时按各 SDK 要求设置设备访问权限并确认设备可用。只在可信机器人网络开放 5560（相机）、5559（手部反馈）和 5556（PC 手部命令）。相机服务的物理采集帧率默认是 **60 FPS**，与下文的 Collector 采集帧率是两个独立设置。

## 3. 启动采集

先打印当前采集参数对应的多终端启动顺序：

```bash
collector/sonic/scripts/run_data_collection_flow.sh print
```

按输出顺序分别启动以下服务。机器人终端 1：

```bash
collector/sonic/scripts/run_camera_server.sh --head-motion-approved
```

机器人终端 2：

```bash
collector/sonic/scripts/run_wuji_hand_server.sh
```

PC 终端 1，启动遥操作底层控制：

```bash
tracker/sonic/gear_sonic_deploy/scripts/run_deploy_pc.sh
```

PC 终端 2，启动 PICO 管理器：

```bash
collector/sonic/scripts/run_pico_manager.sh
```

PC 终端 3，启动录制程序：

```bash
collector/sonic/scripts/run_collector_pc.sh
```

上面的 `run_deploy_pc.sh` 是采集时遥操作所需的 SONIC 底层控制入口，**不是** WB-WAM 模型部署入口。舞肌服务启动会使能电机；采集前确认相机、身体和双手反馈正常，清空工作区域，并备好硬件急停。

PICO 右手柄：摇杆按下开始 episode，`A` 保存、`B` 丢弃。PC Collector 终端也支持 `s` 开始、`q` 保存、`d` 丢弃和 `exit` 退出。

## 4. 采样与输入选项

默认录制频率为 **20 Hz**，采集模式如下：

| Collector 设置 | 录制行为 |
| --- | --- |
| `CAPTURE_MODE=collector_timer`、`CAPTURE_FPS=20`（默认；也支持 30） | 按 Collector 时钟取样，离线以最近值/保持值合并。 |
| `CAPTURE_MODE=official_latest`、`CAPTURE_FPS=50`（可选） | 每 20 ms 锁存最新收到的样本；相机因果选帧，并记录图像年龄、复用和过期信息。 |

按默认 20 Hz 模式启动：

```bash
# Default: 20 Hz collector_timer.
collector/sonic/scripts/run_collector_pc.sh
```

`CAMERA_FPS` 留空时，PC 请求的相机帧率会映射为：20/30 Hz 录制→20/30 FPS，50 Hz 录制→60 FPS。**这不会改动机器人相机服务自身默认的 60 FPS 配置**。`DEFER_DEPTH_COMPRESSION=1` 时，保存 episode 先写原始深度帧，退出或 Ctrl+C 时压缩；设为 `0` 则逐 episode 压缩。旧的 `auto`、`encoder_clock_exact` 模式不再接受。

PICO Camera Listen 默认使用 GStreamer。若 PC 尚未安装，需安装其系统插件和 Python 绑定：

```bash
sudo apt update
sudo apt install -y pkg-config libcairo2-dev libgirepository1.0-dev \
  gobject-introspection gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
uv pip install --python .venv_teleop/bin/python pycairo PyGObject
```

手部输入由 `HAND_CONTROL_MODE` 选择：`manus`、`wuji_glove`、`gesture_wuji` 或 `binary_trigger`，均生成左右手各 20 维舞肌目标。舞肌手套可先用 `collector/sonic/scripts/run_wuji_glove_probe.sh` 检查。

## 5. 转换为训练数据

离线转换工具接受 **20 Hz** SONIC Collector 数据，输出与 WB-WAM `real_archive`
布局兼容的原生 LeRobot v3 数据。使用 Python 3.10–3.12，在 PC 上运行；不需要机器人
SDK、GPU 或模型权重：

```bash
python3 -m venv .venv-process
.venv-process/bin/python -m pip install -r collector/sonic/processing/requirements.txt
.venv-process/bin/python collector/sonic/processing/convert_to_lerobot.py \
  --input /path/to/collected_task --output /path/to/new_archive
.venv-process/bin/python collector/sonic/processing/validate_lerobot.py \
  --root /path/to/new_archive
```

`--input` 也可以是包含多个任务的父目录。输出结构为
`<archive>/<task>/record_XXXX/`；读取单个数据集时传入 record 目录，WB-WAM 后训练配置
也可直接使用 archive 根目录。不会覆盖原始数据。`--dry-run` 预览选择；`--resume`
仅在输入和参数一致时校验并复用已完成的 record。30/50 Hz 数据明确报错，不自动重采样。

每条样本包含当前 RGB/state 和下一帧动作标签：实际身体关节/root、观测到的 SONIC token、
**遥操作双手目标**（不是双手实际反馈）。输出包含110维 state、136维 action、mask 和 RGB
（默认360×270），不导出深度。输入要求、显式排除 episode 和加载示例见
[格式与参数说明](processing/README_zh.md)。
