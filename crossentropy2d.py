# import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# I refered the code from https://github.com/doiken23/focal_segmentation/blob/master/crossentropy2d.py

class CrossEntropy2d(nn.Module):
    def __init__(self, dim=1, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropy2d, self).__init__()
        self.dim = dim
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.criterion = nn.NLLLoss2d(self.weight, self.size_average, self.ignore_index)
        
    def forward(self, input, target):
        if len(target.size()) == 4:
            n,c,h,w = target.size()
            target = target.view(n,h,w)
        return self.criterion(F.log_softmax(input, dim=self.dim),target)
