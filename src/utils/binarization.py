import numpy as np
import cv2

def NMS(img):
    # 在第三个维度上找到每个位置的最大值
    img = img.copy()
    max_values = np.max(img, axis=2, keepdims=True)
    
    # 创建布尔掩码，将不等于最大值的元素置零
    img[img != max_values] = 0
    img[img == max_values] = 1
    return img

def binarize(img, threshold=0.5):

    # 在第三个维度上找到每个位置的最大值
    img = img.copy()
    max_values = np.max(img, axis=-1, keepdims=True)
    max_values = np.where(max_values > threshold, max_values, -1)
    img = (img == max_values).astype(float)
    
    # 创建布尔掩码，将不等于最大值的元素置零
    # img[img != max_values] = 0
    # img[img == max_values] = 1
    return img


def closing(img, kernel_size=5):
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)