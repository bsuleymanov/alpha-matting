import torch
from torch import nn
from torch import nn, einsum
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from functools import partial


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 activation=nn.GELU, dropout_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        print(f"mlp in_f: {in_features}")

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        print(f"mlp x size: {x.size()}")
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        dim_per_head = dim // n_heads
        self.scale = qk_scale or dim_per_head ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_value = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        print(f"x.size: {x.size()}")
        batch_size, seq_len, n_channels = x.size()
        n_heads = self.n_heads

        query = self.query(x)#.chunk(3, dim=-1)
        print(n_heads)
        print(f"query size: {query.size()}")
        query = rearrange(
                query,
                "batch_size n_channels (n_heads dim) -> batch_size n_heads n_channels dim",
                n_heads=n_heads)
        print(f"q size: {query.size()}")
        #q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = rearrange(
                x,
                "batch_size (height width) n_channels -> batch_size n_channels height width",
                height = height, width = width
            )
            x_ = self.sr(x_)
            x_ = rearrange(x_, "batch_size n_channels d1 d2 -> batch_size (d1 d2) n_channels")
            x_ = self.norm(x_)
            key_value = self.key_value(x_).chunk(2, dim=-1)
            #key_value = self.key_value(x_)\
            #    .reshape(batch_size, -1, 2, self.n_heads, n_channels // self.n_heads).permute(2, 0, 3, 1, 4)
        else:
            key_value = self.key_value(x).chunk(2, dim=-1)
            #key_value = self.key_value(x) \
            #    .reshape(batch_size, -1, 2, self.n_heads, n_channels // self.n_heads).permute(2, 0, 3, 1, 4)


        #k, v = key_value[0], key_value[1]
        #print(f"k, v: {k.size()}, {v.size()}")

        print(f"key_value: {key_value[0].size()}")

        key, value = map(
            lambda t: rearrange(
                t,
                "batch_size n_channels (n_heads dim) -> batch_size n_heads n_channels dim",
                n_heads=n_heads),
            key_value
        )

        print(f"query, key, value: {query.size(), key.size(), value.size()}")

        scores = einsum(
            #"batch_size n_heads i dim, batch_size n_heads j dim -> batch_size n_heads i j",
            "b h i j, b h k j -> b h i k",
            query, key
        ) * self.scale
        #scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        print('kke')

        output = einsum(
            "b h i j, b h j k -> b h i k",
            #"batch_size n_heads i j, batch_size n_heads j dim -> batch_size n_heads i dim",
            attn_weights, value
        )
        output = rearrange(
            output,
            "batch_size n_heads seq_len dim -> batch_size seq_len (n_heads dim)"
        )
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, dropout_rate=0., attn_drop=0., drop_path=0.,
                 activation=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            n_heads=n_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=dropout_rate, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hid_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hid_dim,
                       activation=activation, dropout_rate=dropout_rate)

    def forward(self, x, height, width):
        print(f"block x size 1: {x.size()}")
        x = self.attn(self.norm1(x), height, width)
        print(f"block x size 2: {x.size()}")
        x = x + self.drop_path(x)
        print(f"block x size 3: {x.size()}")
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)

        self.image_size = image_size
        self.patch_size = patch_size
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, \
            f"image_size {image_size} should be divided by patch_size {patch_size}."
        self.height, self.width = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        height, width = height // self.patch_size[0], width // self.patch_size[1]

        return x, (height, width)

class PyramidVisionTransformer(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3,
                 n_classes=1000, embed_dims=[64, 128, 256, 512],
                 n_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, dropout_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=5, F4=False):
        super().__init__()
        self.depths = depths
        self.F4 = F4
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(
                image_size=image_size if i == 0 else image_size // (2 ** (i + 1)),
                patch_size=patch_size if i == 0 else 2,
                in_channels=in_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])
            if i != num_stages - 1:
                num_patches = patch_embed.num_patches
            else:
                num_patches = patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=dropout_rate)

            print(f"embed_dims: {embed_dims[i]}")

            block = nn.ModuleList([Block(
                dim=embed_dims[i], n_heads=n_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale, dropout_rate=dropout_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        ...
        #if isinstance(pretrained, str):
        #    load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, height, width):
        if height * width == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.height, patch_embed.width, -1).permute(0, 3, 1, 2),
                size=(height, width), mode="bilinear").reshape(1, -1, height * width).permute(0, 2, 1)

    def forward_features(self, x):
        outs = [x]

        B = x.shape[0]

        for i in range(self.num_stages):
            print(f"stage {i}")
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
        x = self.forward_features(x)

        if self.F4:
            x = x[3:4]

        print(f"PVT size: {len(x)}")
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def pvt_tiny(pretrained=None, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], n_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1], dropout_rate=0.0, drop_path_rate=0.1)
    return model


# class Attention(nn.Module):
#     def __init__(self, dim, n_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#
#         self.dim = dim
#         self.n_heads = n_heads
#         dim_per_head = dim // n_heads
#         self.scale = qk_scale or dim_per_head ** -0.5
#
#         self.query = nn.Linear(dim, dim, bias=qkv_bias)
#         # multiply out features by 2
#         self.key = nn.Linear(dim, dim, bias=qkv_bias)
#         self.values = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x, height, width):
#         batch_size, seq_len, n_channels = x.size()
#         query = self.query().view(batch_size, seq_len,
#                                   self.n_heads, n_channels // self.n_heads)\
#                             .permute(0, 2, 1, 3)
#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(batch_size, n_channels, height, width)
#             x_ = self.sr(x_).reshape(batch_size, n_channels, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             key = self.key(x_).view(batch_size, -1, 2,
#                                     self.n_heads, n_channels // self.n_heads)\
#                               .permute(2, 0, 3, 1, 4)
#             value = self.key(x_).view(batch_size, -1, 2,
#                                     self.n_heads, n_channels // self.n_heads) \
#                                 .permute(2, 0, 3, 1, 4)
#         else:
#             key = self.key(x).reshape(batch_size, -1, 2,
#                                       self.n_heads, n_channels // self.n_heads)\
#                              .permute(2, 0, 3, 1, 4)
#             value = self.value(x).reshape(batch_size, -1, 2,
#                                       self.n_heads, n_channels // self.n_heads) \
#                                  .permute(2, 0, 3, 1, 4)
#
#         scores = torch.matmul(query, key.permute(0, 1, 3, 2)) * self.scale
#         attn_weights = torch.softmax(scores, dim=-1)
#         attn_weights = self.attn_drop(attn_weights)
#
#         output = torch.matmul(attn_weights, value)\
#                                .permute(0, 2, 1, 3)\
#                                .reshape(batch_size, seq_len, n_channels)
#         output = self.proj(output)
#         output = self.proj_drop(output)
#
#         return output



















