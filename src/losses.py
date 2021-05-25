import torch
import torch.nn.functional as F


def semantic_loss_fn(semantic_pred, matte_true, boundary, blurer):
    semantic_true = F.interpolate(matte_true, scale_factor=1/16, mode="bilinear")
    semantic_true = blurer(semantic_true)
    return F.mse_loss(semantic_pred, semantic_true).mean()

def detail_loss_fn(detail_pred, trimap, boundary, matte_true):
    boundary_detail_pred = torch.where(boundary, trimap, detail_pred)
    detail_true = torch.where(boundary, trimap, matte_true)
    return F.l1_loss(boundary_detail_pred, detail_true).mean()

def matte_loss_fn(matte_pred, matte_true, trimap, boundary, images):
    boundary_matte_pred = torch.where(boundary, trimap, matte_pred)
    matte_l1_loss = (F.l1_loss(matte_pred, matte_true) +
                     4. * F.l1_loss(boundary_matte_pred, matte_true))
    matte_compositional_loss = (F.l1_loss(images * matte_pred, images * matte_true) +
        4. * F.l1_loss(images * boundary_matte_pred, images * matte_true))
    return (matte_l1_loss + matte_compositional_loss).mean()

def modnet_loss(semantic_pred, detail_pred, matte_pred,
                matte_true, trimap, blurer, images,
                semantic_scale, detail_scale, matte_scale):
    boundary = (trimap == 0) + (trimap == 1)
    semantic_loss = semantic_loss_fn(semantic_pred, matte_true, boundary, blurer)
    detail_loss = detail_loss_fn(detail_pred, trimap, boundary, matte_true)
    matte_loss = matte_loss_fn(matte_pred, matte_true, trimap, boundary, images)
    return (semantic_loss * semantic_scale + detail_loss * detail_scale +
            matte_loss * matte_scale)