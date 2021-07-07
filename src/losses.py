import torch
import torch.nn.functional as F
from torch import autograd

def semantic_loss_fn(semantic_pred, matte_true, boundary, blurer, average=True):
    if average:
        reduction = 'mean'
    else:
        reduction = 'none'
    semantic_true = F.interpolate(matte_true, scale_factor=1/16, mode="bilinear")
    semantic_true = blurer(semantic_true)
    loss = F.mse_loss(semantic_pred, semantic_true, reduction=reduction)
    #if average:
    #    return loss.mean()
    return loss

def detail_loss_fn(detail_pred, trimap, boundary, matte_true, average=True):
    if average:
        reduction = 'mean'
    else:
        reduction = 'none'
    boundary_detail_pred = torch.where(boundary, trimap, detail_pred)
    detail_true = torch.where(boundary, trimap, matte_true)
    loss = F.l1_loss(boundary_detail_pred, detail_true, reduction=reduction)
    #if average:
    #    return loss.mean()
    return loss

def matte_loss_fn(matte_pred, matte_true, trimap, boundary, images, average=True):
    if average:
        reduction = 'mean'
    else:
        reduction = 'none'
    boundary_matte_pred = torch.where(boundary, trimap, matte_pred)
    matte_l1_loss = (F.l1_loss(matte_pred, matte_true, reduction=reduction) +
                     4. * F.l1_loss(boundary_matte_pred, matte_true, reduction=reduction))
    matte_compositional_loss = (F.l1_loss(images * matte_pred, images * matte_true, reduction=reduction) +
        4. * F.l1_loss(images * boundary_matte_pred, images * matte_true, reduction=reduction))
    loss = matte_l1_loss + matte_compositional_loss
    #print(loss)
    #if average:
    #    return loss.mean()
    return loss

def modnet_loss(semantic_pred, detail_pred, matte_pred,
                matte_true, trimap, images, blurer,
                semantic_scale, detail_scale, matte_scale, average=True, detailed=False):
    boundary = (trimap == 0) + (trimap == 1)
    semantic_loss = semantic_loss_fn(semantic_pred, matte_true, boundary, blurer, average)
    detail_loss = detail_loss_fn(detail_pred, trimap, boundary, matte_true, average)
    matte_loss = matte_loss_fn(matte_pred, matte_true, trimap, boundary, images, average)
    if not average:
        semantic_loss = semantic_loss.view(semantic_loss.size(0), -1).mean(1)
        detail_loss = detail_loss.view(detail_loss.size(0), -1).mean(1)
        matte_loss = matte_loss.view(matte_loss.size(0), -1).mean(1)
    #print(semantic_loss.size(), detail_loss.size(), matte_loss.size())
    loss = (semantic_loss * semantic_scale + detail_loss * detail_scale +
            matte_loss * matte_scale)
    if average:
        return loss.mean()
    if detailed:
        return (semantic_loss * semantic_scale, detail_loss * detail_scale, matte_loss * matte_scale, loss)
    return loss

class ModNetLoss:
    def __init__(self, blurer, semantic_scale, detail_scale, matte_scale, average, detailed, device):
        self.device = device
        self.blurer = blurer.to(self.device)
        self.semantic_scale = semantic_scale
        self.detail_scale = detail_scale
        self.matte_scale = matte_scale
        self.average = average
        self.detailed = detailed

    def __call__(self, semantic_pred, detail_pred, matte_pred,
                 matte_true, trimap, images):
        boundary = (trimap == 0) + (trimap == 1)
        semantic_loss = semantic_loss_fn(semantic_pred, matte_true, boundary, self.blurer, self.average)
        detail_loss = detail_loss_fn(detail_pred, trimap, boundary, matte_true, self.average)
        matte_loss = matte_loss_fn(matte_pred, matte_true, trimap, boundary, images, self.average)
        if not self.average:
            semantic_loss = semantic_loss.view(semantic_loss.size(0), -1).mean(1)
            detail_loss = detail_loss.view(detail_loss.size(0), -1).mean(1)
            matte_loss = matte_loss.view(matte_loss.size(0), -1).mean(1)

        # print(semantic_loss.size(), detail_loss.size(), matte_loss.size())
        loss = (semantic_loss * self.semantic_scale + detail_loss * self.detail_scale +
                matte_loss * self.matte_scale)
        if self.detailed and self.average:
            return (semantic_loss * self.semantic_scale,
                    detail_loss * self.detail_scale,
                    matte_loss * self.matte_scale, loss)
        if self.average:
            return loss.mean()
        return loss

# class ModNetLoss(autograd.Function):
#     def __init__(self, blurer, semantic_scale, detail_scale, matte_scale):
#         super(ModNetLoss, self).__init__()
#         self.blurer = blurer
#         self.semantic_scale = semantic_scale
#         self.detail_scale = detail_scale
#         self.matte_scale = matte_scale
#
#     @staticmethod
#     def forward(self, semantic_pred, detail_pred, matte_pred,
#                 matte_true, trimap, images):
#         boundary = (trimap == 0) + (trimap == 1)
#         semantic_loss = semantic_loss_fn(semantic_pred, matte_true, boundary, self.blurer)
#         detail_loss = detail_loss_fn(detail_pred, trimap, boundary, matte_true)
#         matte_loss = matte_loss_fn(matte_pred, matte_true, trimap, boundary, images)
#         return (semantic_loss * self.semantic_scale + detail_loss * self.detail_scale +
#                 matte_loss * self.matte_scale)