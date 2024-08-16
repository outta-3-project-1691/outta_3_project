import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.models.vgg as vgg

# 모델의 층을 초기화 시킬 때 쓴 코드로, 필요하지 않으시다면 사용하지 않으셔도 됩니다.
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]

    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()


class segmentation_model(nn.Module):
    def __init__(self, n_class):
        super(segmentation_model, self).__init__()
        # [1] 빈칸을 작성하시오.
        pass

    def _initialize_weights(self):
        # [2] 빈칸을 작성하시오.
        pass
        

    def forward(self, x):
        # [3] 빈칸을 작성하시오.
        pass
    