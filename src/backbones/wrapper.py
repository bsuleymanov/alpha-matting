import os
from functools import reduce, partial

import torch
import torch.nn as nn

from .mobilenetv2 import MobileNetV2
from .resnet import ResNet, BasicBlock
from .pvt import PyramidVisionTransformer

from hydra.utils import to_absolute_path

class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class PVTBackbone(BaseBackbone):
    """ResNet18 Backbone"""

    def __init__(self, in_channels):
        super(PVTBackbone, self).__init__(in_channels)
        self.model = PyramidVisionTransformer(
            patch_size=4, embed_dims=[16, 32, 64, 128, 128], n_heads=[1, 2, 4, 8, 8], mlp_ratios=[8, 8, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1, 1], dropout_rate=0.0, drop_path_rate=0.1)
        self.enc_channels = []
        features = self.model(torch.ones([1, 3, 256, 256]))
        for x in features[1:]:
            self.enc_channels.append(x.size(1))
        print(self.enc_channels)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        return self.model(x)




class ResNet18Backbone(BaseBackbone):
    """ResNet18 Backbone"""

    def __init__(self, in_channels):
        super(ResNet18Backbone, self).__init__(in_channels)
        self.model = ResNet(in_channels, BasicBlock, [2, 2, 2, 2], num_classes=None)
        self.enc_channels = []
        features = self.model(torch.ones([1, 3, 256, 256]))
        for x in features[1:]:
            self.enc_channels.append(x.size(1))
        print(self.enc_channels)


    def forward(self, x):
        return self.model(x)
        # # Stage 1
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # enc2x = x
        #
        # # Stage 2
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # enc4x = x
        #
        # # Stage 3
        # x = self.model.layer2(x)
        # enc8x = x
        #
        # # Stage 4
        # x = self.model.layer3(x)
        # enc16x = x
        #
        # # Stage 5
        # x = self.model.layer4(x)
        # enc32x = x
        #
        # return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained network is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = to_absolute_path('./pretrained/resnet18_human_seg.ckpt')
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet18 backbone')
            exit()

        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
        # trained_dict = torch.load("pretrained/resnet18_human_seg.ckpt", map_location="cpu")
        # self.backbone.load_state_dict(trained_dict, strict=False)

class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone 
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone, self).__init__(in_channels)

        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x):
        # x = reduce(lambda x, n: self.network.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.network.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.network.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.network.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.network.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained network is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/mobilenetv2_human_seg.ckpt'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()
        
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt)
