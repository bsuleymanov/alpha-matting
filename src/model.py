import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import scipy

from backbones import SUPPORTED_BACKBONES

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

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))


class IBNorm1(nn.Module):
    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, image):
        image_bn = self.bnorm(image[:, :self.bnorm_channels, ...].contiguous())
        image_in = self.inorm(image[:, self.bnorm_channels:, ...].contiguous())
        return torch.cat((image_bn, image_in), 1)

class IBNorm(nn.Module):
    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        self.inorm = nn.InstanceNorm2d(in_channels, affine=True)

    def forward(self, image):
        image_in = self.inorm(image)
        return image_in

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 use_ibn=True, use_relu=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if use_ibn:
            layers.append(IBNorm(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, image):
        return self.layers(image)


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        batch_size, n_channels, _, _ = image.size()
        w = self.pool(image).view(batch_size, n_channels)
        w = self.fc(w).view(batch_size, n_channels, 1, 1)
        return image * w.expand_as(image)


class LRBranch(nn.Module):
    def __init__(self, backbone):
        super(LRBranch, self).__init__()
        enc_channels = backbone.enc_channels
        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = ConvBlock(enc_channels[4], enc_channels[3], kernel_size=5,
                                    stride=1, padding=2)
        self.conv_lr8x = ConvBlock(enc_channels[3], enc_channels[2], kernel_size=5,
                                   stride=1, padding=2)
        self.conv_lr = ConvBlock(enc_channels[2], 1, kernel_size=3,
                                   stride=2, padding=1, use_ibn=False, use_relu=False)

    def forward(self, image, mode):
        enc_embedding = self.backbone.forward(image)
        enc2x, enc4x, enc32x = enc_embedding[0], enc_embedding[1], enc_embedding[4]
        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode="bilinear", align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode="bilinear", align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        semantic_pred = None
        if mode == "train":
            lr = self.conv_lr(lr8x)
            semantic_pred = torch.sigmoid(lr)

        return semantic_pred, lr8x, [enc2x, enc4x]

class HRBranch(nn.Module):
    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()
        self.tohr_enc2x = ConvBlock(enc_channels[0], hr_channels, 1,
                                    stride=1, padding=0)
        self.conv_enc2x = ConvBlock(hr_channels + 3, hr_channels, 3,
                                    stride=2, padding=1)
        self.tohr_enc4x = ConvBlock(enc_channels[1], hr_channels, 1,
                                    stride=1, padding=0)
        self.conv_enc4x = ConvBlock(2 * hr_channels, 2 * hr_channels, 3,
                                    stride=1, padding=1)
        self.conv_hr4x = nn.Sequential(
            ConvBlock(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            ConvBlock(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            ConvBlock(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        self.conv_hr2x = nn.Sequential(
            ConvBlock(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            ConvBlock(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            ConvBlock(hr_channels, hr_channels, 3, stride=1, padding=1),
            ConvBlock(hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        self.conv_hr = nn.Sequential(
            ConvBlock(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            ConvBlock(hr_channels, 1, 1, stride=1, padding=0,
                      use_ibn=False, use_relu=False),
        )

    def forward(self, image, enc2x, enc4x, lr8x, mode):
        image2x = F.interpolate(image, scale_factor=1/2, mode="bilinear",
                                align_corners=False)
        image4x = F.interpolate(image, scale_factor=1/4, mode="bilinear",
                                align_corners=False)
        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((image2x, enc2x), dim=1))
        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))
        lr4x = F.interpolate(lr8x, scale_factor=2, mode="bilinear",
                             align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, image4x), dim=1))
        hr2x = F.interpolate(hr4x, scale_factor=2, mode="bilinear",
                             align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        detail_pred = None
        if mode == "train":
            hr = F.interpolate(hr2x, scale_factor=2, mode="bilinear",
                               align_corners=False)
            hr = self.conv_hr(torch.cat((hr, image), dim=1))
            detail_pred = torch.sigmoid(hr)

        return detail_pred, hr2x

class FusionBranch(nn.Module):
    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = ConvBlock(enc_channels[2], hr_channels, 5,
                                   stride=1, padding=2)
        self.conv_f2x = ConvBlock(2 * hr_channels, hr_channels, 3,
                                  stride=1, padding=1)
        self.conv_f = nn.Sequential(
            ConvBlock(hr_channels + 3, int(hr_channels / 2), 3,
                      stride=1, padding=1),
            ConvBlock(int(hr_channels / 2), 1, 1,
                      stride=1, padding=0, use_ibn=False, use_relu=False)
        )

    def forward(self, image, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear',
                             align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear',
                             align_corners=False)
        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        f = self.conv_f(torch.cat((f, image), dim=1))
        matte_pred = torch.sigmoid(f)

        return matte_pred


class MODNet(nn.Module):
    def __init__(self, in_channels=3, hr_channels=32, backbone_arch="mobilenetv2",
                 backbone_pretrained=True):
        super(MODNet, self).__init__()
        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()

        #self.freeze_bn()
        self.freeze_backbone()

    def forward(self, image, mode):
        semantic_pred, lr8x, [enc2x, enc4x] = self.lr_branch(image, mode)
        detail_pred, hr2x = self.hr_branch(image, enc2x, enc4x, lr8x, mode)
        matte_pred = self.f_branch(image, lr8x, hr2x)

        return semantic_pred, detail_pred, matte_pred

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def freeze_bn(self):
        norm_types = [nn.BatchNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode="fan_in", nonlinearity="relu")
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
