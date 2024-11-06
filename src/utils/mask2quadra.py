"""

"""

import os
import motti
motti.append_current_dir(os.path.abspath(''))

import numpy as np
import matplotlib.pyplot as plt

import h5py

def mask2quadra(x, border=2) -> np.ndarray:
    w, h, z = x.shape
    non_black_pixels = np.where(x > 0)
    if len(non_black_pixels[0]) == 0:
        return None 
    
    top, bottom = min(non_black_pixels[0]), max(non_black_pixels[0])
    left, right = min(non_black_pixels[1]), max(non_black_pixels[1])
    
    width = right - left
    height = bottom - top
    
    side = max(width, height) + 2 * border
    anchor = (side-height) // 2, (side-width) // 2
    
    new_x = np.zeros(shape=(side, side, x.shape[-1])) 
    new_x[anchor[0]:anchor[0]+height, 
          anchor[1]:anchor[1]+width, 
          ...] = x[top:top+height, left:left+width, ...]

    quadra = np.empty((*new_x.shape, 4))
    for c in range(4):
        quadra[..., c] = np.where(new_x == c, 1, 0)
        
    return quadra

if __name__ == '__main__':
    print(123)
    