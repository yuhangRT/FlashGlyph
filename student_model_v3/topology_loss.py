import torch
import torch.nn.functional as F


def _soft_erode(img):
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skel(img, iters=10):
    img = img.clamp(0.0, 1.0)
    skel = torch.zeros_like(img)
    for _ in range(int(iters)):
        opened = _soft_open(img)
        delta = (img - opened).clamp(min=0.0)
        skel = torch.max(skel, delta)
        img = _soft_erode(img)
    return skel


def cldice_loss(pred, target, mask=None, iters=10, eps=1e-6):
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    if mask is not None:
        pred = pred * mask
        target = target * mask
        if mask.sum() <= 0:
            return pred.new_tensor(0.0)
    skel_pred = soft_skel(pred, iters=iters)
    skel_target = soft_skel(target, iters=iters)
    tprec = (skel_pred * target).sum(dim=(1, 2, 3)) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    tsens = (skel_target * pred).sum(dim=(1, 2, 3)) / (skel_target.sum(dim=(1, 2, 3)) + eps)
    cldice = 1.0 - (2.0 * tprec * tsens / (tprec + tsens + eps))
    cldice = torch.nan_to_num(cldice, nan=0.0, posinf=0.0, neginf=0.0)
    return cldice.mean()
