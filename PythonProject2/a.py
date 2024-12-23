import numpy as np
import cv2
from matplotlib import pyplot as plt

# Sobel算子边缘检测
def sobel_operator(image):
    # Sobel算子
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = np.uint8(sobel)
    return sobel

# 自定义卷积核滤波
def custom_filter(image, kernel):
    # 应用卷积核
    return cv2.filter2D(image, -1, kernel)

# 颜色直方图计算
def calculate_histogram(image):
    # 计算直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist
# 纹理特征提取（灰度共生矩阵GLCM）
def glcm_texture_features(image):
    # 此处省略GLCM计算的代码，因为之前提供的代码存在错误
    pass

# 主函数
def main(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接以灰度模式读取
    if image is None:
        print("Image not found. Please check the path.")
        return
    # Sobel边缘检测
    sobel_image = sobel_operator(image)
    # 自定义卷积核
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4  # 归一化卷积核
    filtered_image = custom_filter(image, kernel)
    # 颜色直方图
    hist = calculate_histogram(image)
    # 纹理特征
    # texture_features = glcm_texture_features(sobel_image)
    # np.save("texture_features.npy", texture_features)
    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(222), plt.imshow(sobel_image, cmap='gray'), plt.title('Sobel Image')
    plt.subplot(223), plt.imshow(filtered_image, cmap='gray'), plt.title('Filtered Image')
    plt.subplot(224), plt.plot(hist), plt.title('Histogram')
    plt.show()

if __name__ == "__main__":
    main("E:\\a.jpg")