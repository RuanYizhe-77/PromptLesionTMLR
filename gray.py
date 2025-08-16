import cv2
import numpy as np
import glob

# 输入和输出路径
input_path = "*.png"  # 替换为你的RGB图片路径
output_path = "gray/"  # 替换为保存处理图片的路径

# 获取所有图片文件路径
image_files = glob.glob(input_path)

for file in image_files:
    # 读取RGB图像
    image = cv2.imread(file)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用Canny边缘检测（调整threshold以获得更好的边缘）
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    # 反转颜色，使边界为灰色，背景为白色
    # 边界区域为较低值，背景为较高值，先归一化到0-255
    gray_edges = cv2.normalize(edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray_edges = 255 - gray_edges  # 反转

    # 保存结果
    output_file = f"{output_path}{file.split('/')[-1].split('.')[0]}_contour.jpg"
    cv2.imwrite(output_file, gray_edges)

    print(f"Processed and saved: {output_file}")
