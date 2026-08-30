import os
import glob
import cv2
import json
import numpy as np
import shutil
from pathlib import Path
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

def main(
    # 👇 修改位置 1：设定默认读取目录名称为 datasets
    data_dir: str = "datasets", 
    repo_id: str = "local/dummyx_real", 
    push_to_hub: bool = False
):
    # 👇 修改位置 2：根据 BASE_DIR 自动转换为绝对路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
        
    # 1. 查找所有 episode 文件夹
    ep_dirs = sorted(glob.glob(os.path.join(data_dir, "episode_*")))
    if not ep_dirs:
        print(f"❌ 错误：在 {os.path.abspath(data_dir)} 中未找到任何数据！")
        return
        
    print(f"🔍 找到 {len(ep_dirs)} 条真实轨迹，准备转换为 LeRobot 格式...")

    # 2. 读取第一个 episode 获取元数据
    with open(os.path.join(ep_dirs[0], "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # 🚨 核心修改 1：强制截断！不管原数据有几个相机，最多只转换前 2 个 (cam_0和cam_1)
    cameras_count = min(meta.get("cameras_count", 1), 2)
    record_fps = meta.get("fps_target", 30)
    
    # 3. 清理旧缓存
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # 4. 定义特征
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": ["motors"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["motors"]},
    }
    
    # 🚨 核心修改 2：去掉了 "cam_third"，只保留前两个
    cam_name_mapping = ["cam_fixed", "cam_wrist"]
    for i in range(cameras_count):
        feature_name = cam_name_mapping[i] if i < len(cam_name_mapping) else f"cam_{i}"
        features[f"observation.images.{feature_name}"] = {
            "dtype": "image",
            "shape": (240, 424, 3), 
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="dummyx",
        fps=record_fps,
        features=features,
    )

    # 5. 开始转换
    for ep_dir in tqdm.tqdm(ep_dirs, desc="🔄 转换中"):
        with open(os.path.join(ep_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            task_instruction = meta.get("instruction", "manipulate object")
            
        with open(os.path.join(ep_dir, "data.json"), "r", encoding="utf-8") as f:
            steps = json.load(f)
            
        num_frames = len(steps)
        
        for i in range(num_frames):
            step = steps[i]
            next_step = steps[i+1] if i + 1 < num_frames else step
            
            # 提取 1-6 轴位置
            state_arm = [step["positions"][str(j)] for j in range(1, 7)]
            action_arm = [next_step["positions"][str(j)] for j in range(1, 7)]
            
            # 👇 删掉之前的二值化，直接透传真实采集的连续值！
            gripper_state = step["positions"]["7"]
            gripper_action = next_step["positions"]["7"]
            
            # 合并为最终数组
            state_arr = np.array(state_arm + [gripper_state], dtype=np.float32)
            action_arr = np.array(action_arm + [gripper_action], dtype=np.float32)
            
            frame_dict = {
                "observation.state": state_arr,
                "action": action_arr,
                "task": task_instruction,
            }
            
            # 处理多相机图像 (此时 cameras_count 最大只有 2)
            for cam_idx in range(cameras_count):
                feature_name = cam_name_mapping[cam_idx] if cam_idx < len(cam_name_mapping) else f"cam_{cam_idx}"
                img_file_name = step["images"].get(f"cam_{cam_idx}")
                
                if img_file_name:
                    img_abs_path = os.path.join(ep_dir, "images", img_file_name)
                    img_bgr = cv2.imread(img_abs_path)
                    if img_bgr is not None:
                        frame_dict[f"observation.images.{feature_name}"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
            dataset.add_frame(frame_dict)
            
        dataset.save_episode()

    print(f"\n🎉 转换成功！数据集位置: {output_path}")

if __name__ == "__main__":
    tyro.cli(main)