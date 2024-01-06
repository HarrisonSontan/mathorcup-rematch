import cv2
import numpy as np
import os


def gray_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def ostu_img(img_gray):
    ret, threshold = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold


def dilate_binary_img(binary_image, kernel_size, iterations):
    # 定义膨胀操作的结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 对二值图像进行膨胀操作
    dilated_image = cv2.dilate(binary_image, kernel, iterations=iterations)

    return dilated_image


def erode_binary_img(binary_image, kernel_size, iterations):
    # 定义腐蚀操作的结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 对二值图像进行腐蚀操作
    eroded_image = cv2.erode(binary_image, kernel, iterations=iterations)

    return eroded_image


def canny_edge_detection(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测算法
    edges = cv2.Canny(gray, 80, 100)

    return edges


# import os
# import cv2

# 指定待处理图片的目录
input_dir = r"C:\Users\19156\Desktop\MATHORCUP2\code\dataset\ann_p"
output_dir = r"C:\Users\19156\Desktop\MATHORCUP2\code\dataset\ann_z"

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历指定目录中的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 确保文件是图像文件
        # 读取图像
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 进行图像处理（这里以直方图均衡化为例）
        equalized_image = cv2.equalizeHist(image)

        # 构建输出文件路径
        output_path = os.path.join(output_dir, filename)

        # 保存处理后的图像
        cv2.imwrite(output_path, equalized_image)

print("批量处理完成")


# # 指定目录路径
# directory = r'C:\Users\19156\Desktop\MATHORCUP2\code\dataset\annotated'

# # 遍历目录下的所有文件
# for filename in os.listdir(directory):
#     if filename.endswith('.jpg'):
#         # 构建完整的图像路径
#         image_path = os.path.join(directory, filename)

#         # 进行边缘检测
#         edges = canny_edge_detection(image_path)

#         # 构建结果文件名
#         result_filename = 'edge_' + filename

#         # 保存结果图像
#         cv2.imwrite(os.path.join(directory, result_filename), edges)
