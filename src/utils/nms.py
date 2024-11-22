import numpy as np
def NMS(img):
    # 在第三个维度上找到每个位置的最大值
    img = img.copy()
    max_values = np.max(img, axis=2, keepdims=True)
    
    # 创建布尔掩码，将不等于最大值的元素置零
    img[img != max_values] = 0
    img[img == max_values] = 1
    return img

def capture_image(image, t):
    global images_per_step
    images_per_step.append(image.cpu().clone())