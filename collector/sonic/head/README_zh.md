# 宇树 G1 头部相机与舵机

## 默认位置：无需头部标定

部署和采集均使用 `HEAD_SERVO_MODE=raw`，机器人端 `collector/sonic/scripts/head_servo.env` 默认配置为：

```bash
HEAD_SERVO_MODE=raw
HEAD_JOINT0_ENCODER=3027
HEAD_JOINT1_ENCODER=1849
```

以上数值是编码器计数，不是角度。环境示例、启动脚本和控制程序使用相同默认位置。配置时创建文件并补齐缺失值，保留用户明确设置的覆盖值。**无需头部标定或首次示教。** 启动时先移动到该位置并保持，再启动相机取流。

真机部署、SONIC/PICO 和 HGPT 数据采集均需要本组件。RealSense D455 图像采集、CH340 USB 串口驱动和头部双轴舵机控制是三部分；能看到图像不代表舵机就绪。

源码随仓库提供：NVIDIA L4T 36.4.3 的 `ch341.c`、宇树提供包中的 Dynamixel C++ SDK，以及本仓库的 `head_servo`。不复制 PC 编译的内核模块到机器人，而是在机器人上用当前内核的 headers、配置和 `Module.symvers` 编译。来源、校验和与许可证见[英文说明](README.md)。

## 环境配置：部署和采集均必做

PC 仓库根目录执行以下命令，完成头部组件的复制、机器人端编译、串口配置和检测：

```bash
collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP
```

脚本复制组件及相机联动启动脚本到机器人 `~/WB-WAM`（相机 Python 服务及其环境仍按部署说明安装），从示例创建 `head_servo.env`，保留已有配置并补齐缺失的默认位置，自动发现唯一串口后填写稳定路径，编译 `head_servo`，按需编译/安装 CH341 内核模块，处理 brltty 对 CH340 的抢占及 `dialout` 权限，再通过新的 SSH 登录检测两个舵机。密码只在终端交互输入。需要的编译依赖为 `build-essential cmake python3 kmod rsync` 和与当前内核匹配的 `nvidia-l4t-kernel-headers`；缺失时安装对应版本，不升级内核。仅针对 CH340 修改 brltty 匹配规则，不卸载整个 brltty。

若系统已有 `ch341`，直接复用；缺失时仓库自带源码目前支持 L4T 36.4.3 / `5.15.148-tegra`。其他内核需提供匹配驱动，不强行加载旧模块。

机器人 `~/WB-WAM` 下也可分步运行：

```bash
bash collector/sonic/head/setup_robot.sh --build-only
bash collector/sonic/head/setup_robot.sh --install
# 重新 SSH 登录，使 dialout 组生效后：
collector/sonic/scripts/probe_head_servo.sh
```

检测自动选择唯一 CH340 `1a86:7523`，优先使用稳定的 `by-id` / `by-path` 串口路径。存在多个转换器时，在机器人端 `collector/sonic/scripts/head_servo.env` 中填写 `HEAD_SERIAL_DEVICE`；不要猜 `ttyUSB0`。检测必须收到两个舵机的真实回复，并输出 `PASS: both head servos replied; no control registers written`；仅有串口节点不算通过。此检测不使能舵机、不写控制寄存器。

## 检查配置位置

现场人员准备好在断力矩时支撑头部、确认周围无障碍后，在机器人端执行有界到位与保持测试：

```bash
collector/sonic/scripts/run_head_servo.sh --motion-approved --duration 3
collector/sonic/scripts/probe_head_servo.sh
```

要求位置反馈正常、退出后两个舵机 torque 均为 0。退出会释放舵机，停止前需支撑头部。`--duration 3` 指到位后保持 3 秒，移动时间另计。默认使用配置中的固定位置，不跟随 PICO。仅检查原位保持时可显式设置 `HEAD_SERVO_MODE=current`，但不能据此判定默认目标已验证。

## 可选：更换保存的位置

仅在需要调整默认位置时，在机器人仓库根目录执行：

```bash
HEAD_SERVO_MODE=teach collector/sonic/scripts/run_head_servo.sh --motion-approved
```

出现 `TEACH READY` 后，两轴已上力矩，使用 P=0、D=100 的阻尼模式，需用手托住相机调整。摆稳后输入 `hold`；程序检查读数稳定，在保持上电的情况下锁定当前位置并恢复原保持增益，输出 `CAPTURED`。将两轴读数填入本机 `head_servo.env` 的 `HEAD_JOINT0_ENCODER` / `HEAD_JOINT1_ENCODER`，设置 `HEAD_SERVO_MODE=raw`，后续使用该位置。**不要先停止再读取：断力矩会改变相机位置。** 此调整不是首次环境配置的必做项。

未确认的示教最多持续 300 秒；超时、输入断开、示教期间输入 `stop` 或故障会关闭力矩并恢复增益。锁定后持续保持，按 Ctrl+C 停止前需托住相机。

## 推理与数采启动

两种流程都在机器人端同一终端运行：

```bash
cd ~/WB-WAM
collector/sonic/scripts/run_camera_server.sh --head-motion-approved
```

默认先启动头部 `head_servo`，确认双轴已到达配置位置（默认 3027/1849）并保持后再启动相机。运行推理/数采期间保持该终端开启；舵机故障会停止相机，相机退出会停止舵机服务。Ctrl+C 会停止两者并请求关闭舵机力矩。USB 断开或强制杀进程可能使关闭指令无法送达，必须明确报告并让现场人员支撑、停止硬件。

仅排查图像时可用 `HEAD_SERVO_ENABLED=0 collector/sonic/scripts/run_camera_server.sh`，但不能据此宣布部署/数采头部检查完成。使用假相机的 dry-run 不启动真实头部服务。

若无串口，检查 USB、`modinfo ch341`、`lsmod`、匹配 headers 及内核日志中的 brltty 冲突。若安装前设备已被 brltty 抢占，可重新连接头部 USB 后检测；无需拔插 Wuji 双手。升级内核后需重新编译，禁止在串口控制器运行中卸载驱动。
