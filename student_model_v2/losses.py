import torch
import torch.nn as nn
import torch.nn.functional as F

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft  # type: ignore


class FocalFrequencyLoss(nn.Module):
    """Frequency-domain loss with dynamic focal weighting.

    Ref: Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021).
    """

    def __init__(
        self,
        loss_weight=1.0,
        alpha=1.0,
        patch_factor=1,
        ave_spectrum=False,
        log_matrix=False,
        batch_matrix=False,
    ):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.alpha = float(alpha)
        self.patch_factor = int(patch_factor)
        self.ave_spectrum = bool(ave_spectrum)
        self.log_matrix = bool(log_matrix)
        self.batch_matrix = bool(batch_matrix)

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        if h % patch_factor != 0 or w % patch_factor != 0:
            raise ValueError("patch_factor must divide image height and width")

        patch_h = h // patch_factor
        patch_w = w // patch_factor
        patches = []
        for i in range(patch_factor):
            for j in range(patch_factor):
                patches.append(
                    x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
                )
        y = torch.stack(patches, 1)

        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm="ortho")
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            if self.batch_matrix:
                denom = matrix_tmp.max().clamp(min=1e-8)
                matrix_tmp = matrix_tmp / denom
            else:
                denom = matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
                denom = denom.clamp(min=1e-8)
                matrix_tmp = matrix_tmp / denom

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            weight_matrix = torch.clamp(matrix_tmp, min=0.0, max=1.0).detach()

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


class MultiDomainTextLoss(nn.Module):
    def __init__(
        self,
        ffl_weight=0.05,
        grad_weight=0.05,
        ffl_alpha=1.0,
        ffl_patch_factor=1,
        ffl_ave_spectrum=False,
        ffl_log_matrix=False,
        ffl_batch_matrix=False,
        text_weight=5.0,
    ):
        super().__init__()
        self.ffl_weight = float(ffl_weight)
        self.grad_weight = float(grad_weight)
        self.text_weight = float(text_weight)

        self.ffl = FocalFrequencyLoss(
            loss_weight=1.0,
            alpha=ffl_alpha,
            patch_factor=ffl_patch_factor,
            ave_spectrum=ffl_ave_spectrum,
            log_matrix=ffl_log_matrix,
            batch_matrix=ffl_batch_matrix,
        )

        self.register_buffer(
            "kernel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "kernel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )

    def masked_gradient_loss(self, pred, target, mask=None):
        pred_f = pred.float()
        target_f = target.float()
        c = pred_f.shape[1]
        kx = self.kernel_x.repeat(c, 1, 1, 1)
        ky = self.kernel_y.repeat(c, 1, 1, 1)

        pred_grad_x = F.conv2d(pred_f, kx, padding=1, groups=c)
        pred_grad_y = F.conv2d(pred_f, ky, padding=1, groups=c)
        target_grad_x = F.conv2d(target_f, kx, padding=1, groups=c)
        target_grad_y = F.conv2d(target_f, ky, padding=1, groups=c)

        loss = F.l1_loss(pred_grad_x, target_grad_x, reduction="none") + F.l1_loss(
            pred_grad_y, target_grad_y, reduction="none"
        )

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[-2:] != pred_f.shape[-2:]:
                mask = F.interpolate(mask, size=pred_f.shape[-2:], mode="nearest")
            mask = mask.to(device=pred_f.device, dtype=pred_f.dtype)
            weight_map = 1.0 + (self.text_weight - 1.0) * mask
            loss = loss * weight_map

        return loss.mean()

    def forward(self, pred_x0, target_x0, mask=None):
        loss_lcm = F.huber_loss(pred_x0, target_x0, delta=1.0)

        loss_ffl = pred_x0.new_tensor(0.0)
        loss_grad = pred_x0.new_tensor(0.0)

        if self.ffl_weight > 0:
            loss_ffl = self.ffl(pred_x0.float(), target_x0.float())
        if self.grad_weight > 0:
            loss_grad = self.masked_gradient_loss(pred_x0, target_x0, mask)

        total_loss = loss_lcm + self.ffl_weight * loss_ffl + self.grad_weight * loss_grad
        return total_loss, {"lcm": loss_lcm, "ffl": loss_ffl, "grad": loss_grad}
