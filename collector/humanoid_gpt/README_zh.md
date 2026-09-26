# HumanoidGPT 数据采集

部署或操作真实机器人前，请阅读[安全免责声明](../../README_zh.md#安全免责声明)。

**头部舵机（必需）：** 部署/采集前完成[头部驱动安装、机器人端编译和测试](../sonic/head/README_zh.md)。PC 仓库根目录运行 `collector/sonic/scripts/setup_head_servo.sh USER@ROBOT_IP`。推理和收数据时，机器人端使用 `run_camera_server.sh --head-motion-approved` 同时启动头部舵机保持服务与相机；默认使用 `HEAD_SERVO_MODE=raw`，移动到 `HEAD_JOINT0_ENCODER=3027`、`HEAD_JOINT1_ENCODER=1849` 并保持（单位为编码器计数，不是角度）。环境示例已提供这些默认值，无需头部标定或首次示教。舵机通信失败时不能只启动图像服务并宣告就绪。

[English](./README.md)

本页只介绍 HGPT 遥操作数据的采集与转换，不负责模型部署。真实机器人启动前，须完成 `tracker/humanoid_gpt/README.md` 中的环境与真机安全检查；其中的 `h-gpt` 是独立的控制器环境。除注明外，命令从仓库根目录运行；长时间运行的服务各占一个终端。

```text
PICO / MANUS → Pose Manager ──参考动作──→ HGPT 控制器 → G1
                    │                         │
                    └──舞肌目标→ 舞肌服务 → 双手 │
                                              ↓
相机 RGB/深度 ───────────────────────────→ Collector ← 机器人状态、动作与手部反馈
                                              ↓
                                      20 Hz 原始 episode
                                              ↓
                                         LeRobot v3 转换
```

## 配置

先复制本机配置，并填写机器人网卡、机器人端手部反馈地址和相机地址：

```bash
cp -n collector/humanoid_gpt/scripts/hgpt_collection.env.example \
  collector/humanoid_gpt/scripts/hgpt_collection.env
python collector/humanoid_gpt/scripts/check_env.py
```

三个 PC 启动器共用 `hgpt_collection.env`。默认使用 MANUS 手部输入、外部 PICO Service、TensorRT，以及本机 pose `:5556`、状态/动作 `:5558`。相机默认连接 `tcp://127.0.0.1:5560`；如果相机服务运行在机器人上，设为 `tcp://ROBOT_IP:5560`，或将机器人端口转发到 PC。导出的 `HGPT_*` 环境变量可覆盖模板默认值；显式命令行选项优先。不要把填有真实地址的本机配置提交到 Git。

## 启动采集

先在机器人端启动舞肌反馈服务和相机服务，再在 PC 上依次启动 Pose Manager、HGPT 控制器和 Collector。以下 PC 命令分别在独立终端执行：

```bash
# 机器人端，仓库根目录；默认采集配置需要 60 FPS 相机输入
collector/sonic/scripts/run_wuji_hand_server.sh
CAMERA_FPS=60 collector/sonic/scripts/run_camera_server.sh --head-motion-approved

# PC 终端 1，仓库根目录
cd tracker/humanoid_gpt
scripts/run_hgpt_pose_manager.sh \
  --hand-source manus --pico-service-mode external --publish-wuji-hand

# PC 终端 2，从仓库根目录进入 tracker/humanoid_gpt/
cd tracker/humanoid_gpt
scripts/run_pico_real.sh \
  --net ROBOT_INTERFACE --pose-endpoint tcp://127.0.0.1:5556 \
  --state-action-bind tcp://127.0.0.1:5558 \
  --policy-provider tensorrt --publish-lowcmd

# PC 终端 3，仓库根目录
collector/humanoid_gpt/scripts/run_collector_pc.sh --task-name example_task
```

`--publish-lowcmd` 会启用真实机器人身体指令；省略它**不会**自动关闭舞肌手指令，做无驱动检查时还须单独禁用手部发布。机器人应先进入安全的 debug 状态，操作员握持遥控器并备好急停。具体使能顺序见上面的 HGPT 安全文档。

PICO 右手柄摇杆按下开始 episode，`A` 保存，`B` 丢弃；也可在 Collector 终端用 `s` 开始、`q` 保存、`d` 丢弃、`exit` 退出。Pose Manager 窗口中 `0` 是行走、`1` 是在线跟踪、`2+` 是离线动作。保存/丢弃按键仅在上升沿触发，同时按 `A+B` 会被忽略。

## 采集频率与原始数据

默认 `--capture-fps 20 --camera-fps 60 --downsample-method auto`。`auto` 在 20 Hz 时选择插值：每隔 50 ms 写一行，等待最多 40 ms 收集相邻的 50 Hz 遥测；连续量线性插值、四元数使用 SLERP。相机图像不插值，而是选择 PC 接收时间最接近的真实帧。对齐后的身体关节位置用于重算 20 Hz 关节速度。采集器本身不额外做 EMA；Pose Manager 对 PICO/GMR 参考动作的平滑仍会影响采到的参考值。

如需保留原先的 50 Hz 最新值采集方式，可显式指定：

```bash
collector/humanoid_gpt/scripts/run_collector_pc.sh \
  --task-name example_task --capture-fps 50
```

50 Hz 下 `auto` 选择 `causal_latest`，**不是**上述 20 Hz 插值模式。下面的 LeRobot 转换器目前只接受 `capture_fps=20` 的原始数据，不能直接转换 50 Hz episode。

默认原始输出位于 `datasets/humanoid_gpt/<task_name>/episode_*/`，包括 `data.json`、`motion.npz`、RGB 和可选深度图。状态、实际双手反馈、双手目标、HGPT 参考动作及控制器动作都保留在原始记录中；`motion.npz` 的根位置/姿态是 PICO/GMR 参考值，不是机器人世界坐标真值。

## 转成 LeRobot 训练数据

转换程序使用 `collector/humanoid_gpt/processing/` 中的本地工具，不再依赖旧的 `datasets/preprocess/` 代码。下文的 `datasets/` 路径仅用于本地数据输入和输出。

转换前先确认任务描述不是 `example_task` 等占位文本。对单个 20 Hz 任务，可先只检查，再正式转换：

```bash
python collector/humanoid_gpt/scripts/convert_hgpt_to_lerobot_v3.py \
  --source-root datasets/humanoid_gpt/YOUR_TASK \
  --task 'Describe the task here.' --dry-run

python collector/humanoid_gpt/scripts/convert_hgpt_to_lerobot_v3.py \
  --source-root datasets/humanoid_gpt/YOUR_TASK \
  --task 'Describe the task here.' \
  --output-root datasets/humanoid_gpt_lerobot_v3_20hz
```

转换结果使用 LeRobot v3 文件组织；每行图像和状态对应时刻 `t`，动作对应下一帧 `t+1`，不要在训练加载时再次后移。**它不是 Pico/Real 开源数据集的同一布局**：HGPT 转换器输出 `states` 146 维、`action` 136 维，额外保留 HGPT 参考 `qpos36`，并提供 HGPT 参考动作与物理机器人动作两种训练视图；选择哪一种须用对应的数据配置，不能直接套用 Pico/Real 的 110 维 `observation.state` 配置。转换参数及字段定义以[转换脚本](scripts/convert_hgpt_to_lerobot_v3.py)为准。

## 启动检查与排障

Collector 启动时检查输出目录的写入、同步、重命名和回读。进程中的首次开始请求还会检查身体/双手训练数据、参考动作有效性及来源新鲜度；失败会打印原因并保持空闲，修复后再次触发开始即可。相机缺帧也会阻止写帧。保存失败会进入 `save_failed`，保留当前 episode 供重试或人工处理。

```bash
python collector/humanoid_gpt/scripts/check_env.py
pytest tracker/humanoid_gpt/tests
ruff check collector/humanoid_gpt tracker/humanoid_gpt
```
