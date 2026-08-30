import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def main():
    # 1. 找到 dataset 文件夹下最新生成的一个 .npz 文件
    dataset_dir = "dataset"
    files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    
    if not files:
        print("❌ 错误：在 dataset 文件夹下没有找到 .npz 文件！")
        return
        
    # 按创建时间排序，取最新的一个
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 正在读取最新采集的双摄轨迹: {latest_file}\n")
    
    # 2. 💡 加载压缩包里的数据 (适配最新双摄版)
    data = np.load(latest_file)
    images_fixed = data['images_fixed']
    images_wrist = data['images_wrist']
    qpos = data['qpos']
    actions = data['actions']
    
    # 3. 打印数据维度
    print("=== 📊 数据集维度质检 ===")
    print(f"✅ 全局视角图像 (Images Fixed) : {images_fixed.shape} -> 含义：(帧数, 高度256, 宽度256, RGB 3通道)")
    print(f"✅ 手腕视角图像 (Images Wrist) : {images_wrist.shape} -> 含义：(帧数, 高度256, 宽度256, RGB 3通道)")
    print(f"✅ 本体感觉 (Qpos)           : {qpos.shape}    -> 含义：(帧数, 6个当前关节角度)")
    print(f"✅ 专家动作 (Actions)        : {actions.shape}    -> 含义：(帧数, 6个目标关节角度)")
    
    # 4. 抽取关键帧进行双重视觉验证
    num_frames = images_fixed.shape[0]
    if num_frames < 3:
        print("数据帧数过少，无法进行视觉验证。")
        return
        
    indices_to_check = [0, num_frames // 2, num_frames - 1]
    
    # 💡 创建 2行 3列 的画布，尺寸稍微拉高一点
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, idx in enumerate(indices_to_check):
        # --- 第一行：全局上帝视角 ---
        img_f = images_fixed[idx] 
        # 如果画面倒了可以用 np.flipud(img_f)
        axes[0, i].imshow(img_f)
        axes[0, i].set_title(f"[Fixed Cam] Step: {idx}")
        axes[0, i].axis("off")
        
        # --- 第二行：第一人称手腕视角 ---
        img_w = images_wrist[idx] 
        axes[1, i].imshow(img_w)
        axes[1, i].set_title(f"[Wrist Cam] Step: {idx}")
        axes[1, i].axis("off")
        
    plt.tight_layout() # 自动调整间距
    save_img_path = "verify_camera_dual.png"
    plt.savefig(save_img_path, bbox_inches='tight')
    
    print(f"\n📸 双摄视觉质检完毕！")
    print(f"我已经把抓取的 起点、中点、终点 截图拼接并保存为了 '{save_img_path}'")
    print("💡 导师要求：")
    print("1. 请去目录打开这张图片，确认全局视野是否覆盖桌面。")
    print("2. 重点确认第二排【手腕视角】！看看它是不是以第一人称的视角，逐渐靠近目标绿块。")

if __name__ == "__main__":
    main()