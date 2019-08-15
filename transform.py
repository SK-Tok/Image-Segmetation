# import torch module
import torch
import torch.utils.data as data_utils

# import python module
import numpy as np
import os
import random

# Crop
class NPSegRandomCrop(object):
    def __init__(self,output_size):
        self.output_size = output_size
    def __call__(self, arr_list):
        image_arr, GT_arr = arr_list
        h,w = image_arr.shape[1:]
        new_h, new_w = self.output_size, self.output_size
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        image_arr = image_arr[:,top:top+new_h,left:left+new_w]
        GT_arr = GT_arr[:,top:top+new_h,left:left+new_w]
        arr_list = [image_arr, GT_arr]
        return arr_list

# Flip
class NPSegFlip(object):
    def __init__(self):
        pass
    def __call__(self,arr_list):
        image_arr, GT_arr = arr_list
        
        if random.choices([True, False]):
            image_arr = np.flip(image_arr, 1).copy()
            GT_arr = np.flip(GT_arr, 1).copy()
            
        if random.choices([True, False]):
            image_arr = np.flip(image_arr, 2).copy()
            GT_arr = np.flip(GT_arr, 2).copy()
            arr_list = [image_arr, GT_arr]
            return arr_list
        
# Rotate
class NPRandomRotate(object):
    def __init__(self):
        pass
    def __call__(self, arr_list):
        image_arr, GT_arr = arr_list
        
        n = random.choices([0,1,2,3])
        image_arr = np.rot90(image_arr, n[0], (1,2)).copy()
        GT_arr = np.rot90(GT_arr, n[0], (1,2)).copy()
        
        arr_list = [image_arr, GT_arr]
        return arr_list
