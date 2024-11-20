"""

"""
import numpy as np

def mask2quadra(x, border=10) -> np.ndarray:
    w, h, z = x.shape
    non_black_pixels = np.where(x > 0)
    if len(non_black_pixels[0]) == 0:
        return None 
    
    top, bottom = min(non_black_pixels[0]), max(non_black_pixels[0])
    left, right = min(non_black_pixels[1]), max(non_black_pixels[1])
    
    width = right - left
    height = bottom - top
    
    side = max(width, height) + 2 * border
    center = (top+bottom) // 2, (left+right) // 2
    new_x = x[center[0]-side//2:center[0]+side//2, center[1]-side//2:center[1]+side//2, ...]

    quadra = np.empty((*new_x.shape, 4))
    for c in range(4):
        quadra[..., c] = np.where(new_x == c, 1, 0)
        
    return quadra

    