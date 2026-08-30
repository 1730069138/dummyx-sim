import cv2
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

def images_to_video(image_folder, video_name, fps=30):
    """
    将文件夹内的图片序列转换为视频。
    """
    # 1. 获取文件夹内所有图片文件（支持 .jpg 和 .png）
    images = [img for img in os.listdir(image_folder) if img.lower().endswith((".jpg", ".png"))]
    
    if not images:
        print(f"错误: 在 '{image_folder}' 中未找到任何 .jpg 或 .png 图片！")
        return

    # 2. 对图片进行排序以保证视频帧的连贯性
    # 默认按字母顺序排序。如果你的数据集是以数字命名的 (例如 1.png, 2.png ... 10.png)，
    # 为了防止 10.png 排在 2.png 前面，这里使用提取数字的方式进行排序：
    try:
        images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    except ValueError:
        # 如果文件名中没有数字，则回退到常规排序
        images.sort()

    # 3. 读取第一张图片，用于获取视频的宽高分辨率
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"错误: 无法读取图片 '{first_image_path}'。")
        return
        
    height, width, layers = frame.shape

    # 4. 初始化视频写入对象 (VideoWriter)
    # 使用 'mp4v' 编码器生成 .mp4 格式视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"开始生成视频，共检测到 {len(images)} 张图片...")
    print(f"视频分辨率: {width}x{height}, 帧率: {fps} FPS")

    # 5. 遍历图片并写入视频
    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        
        # 确保读取到的图片尺寸与第一张图一致，否则会导致写入失败
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
            
        video.write(frame)
        
        # 打印进度
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(images)} 张图片...")

    # 6. 释放资源
    video.release()
    cv2.destroyAllWindows()
    print(f"\n视频生成完毕！已成功保存为: {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image sequence to MP4.")
    parser.add_argument("image_folder", help="Directory containing PNG/JPEG frames")
    parser.add_argument("output_video", help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=60, help="Output frame rate")
    args = parser.parse_args()
    output_video = project_path(args.output_video)
    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    images_to_video(project_path(args.image_folder), output_video, args.fps)
