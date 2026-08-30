# dummyx-sim

整理后的 DummyX MuJoCo 仿真工作空间，目标平台为 Ubuntu 22.04。项目不包含真实机械臂、CAN/USB2CAN 或真实相机控制代码，也不提交本地数据集与实验录像。

## 目录

- `models/`：MuJoCo 机器人模型、场景和网格资源。
- `scripts/collect/`：纯仿真数据采集。
- `scripts/convert/`：数据集转换工具。
- `scripts/deploy/`：仿真策略部署；`deploy_screwdriver_client_vlsa_v2_7.py` 为当前主实验版本。
- `scripts/analysis/`：实验结果绘图。
- `scripts/tools/`：模型检查、标定和媒体工具。
- `archive/experiments/`：保留的旧实验版本，不作为主入口维护。
- `archive/broken_scenes/`：依赖缺失模型的旧场景，仅供追溯。

`datasets/` 与 `recordings/` 由程序按需生成，已通过 `.gitignore` 排除。

## Ubuntu 22.04 快速开始

当前机器的 Conda 环境不会被修改。待代码验证完成后，再从原环境导出精简的复现文件。

```bash
conda create -n dummyx-sim python=3.10
conda activate dummyx-sim
python -m pip install -r requirements.txt
python scripts/tools/check_xml.py
```

需要连接 OpenPI 策略服务器时，另行安装与服务器版本匹配的 `openpi-client`。

## 常用命令

```bash
# 采集纯仿真数据
python scripts/collect/data_collector_v2.py

# 检查默认 MuJoCo 场景
python scripts/tools/check_xml.py

# 将图像序列转成视频
python scripts/tools/img2video.py datasets/example/cam_wrist outputs/preview.mp4 --fps 60
```

脚本使用相对于项目根目录的路径，不依赖原电脑上的 `/home/jun/...` 绝对路径。

