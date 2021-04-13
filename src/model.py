import torch
from torch import nn
import math
import numpy as np
import scipy


class GaussianBlurLayer(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(GaussianBlurLayer, self).__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(self.n_channels, self.n_channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=n_channels)
        )

        self._init_kernel()

    def forward(self, image):
        if not len(image.size()) == 4:
            print(f"'GaussianBlurLayer' requires a 4D tensor as input.")
            exit()
        elif not image.size(1) == self.n_channels:
            print(f"In 'GaussianBlurLayer', the required number of channels "
                  f"{self.n_channels} is not the same as input {image.size(1)}.")
            exit()
        return self.op(image)

    def _init_kerne(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))
























