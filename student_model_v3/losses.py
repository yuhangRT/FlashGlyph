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


class HighFreqTextLoss(nn.Module):
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
        self.register_buffer("gaussian_kernel", self._build_gaussian_kernel(kernel_size=5, sigma=1.0))

    @staticmethod
    def _build_gaussian_kernel(kernel_size=5, sigma=1.0):
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_2d = torch.outer(gauss_1d, gauss_1d)
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    @staticmethod
    def _prepare_mask(mask, spatial_size, device, dtype):
        if mask is None:
            return None
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() == 4 and mask.shape[1] > 1:
            mask = mask.max(dim=1, keepdim=True).values
        if mask.shape[-2:] != spatial_size:
            mask = F.interpolate(mask, size=spatial_size, mode="nearest")
        return (mask.to(device=device, dtype=dtype) > 0.5).float()

    def _build_soft_window(self, mask):
        if mask is None:
            return None
        window = mask
        # Expand text support a bit, then smooth to reduce spectral ringing.
        window = F.max_pool2d(window, kernel_size=3, stride=1, padding=1)
        window = F.max_pool2d(window, kernel_size=3, stride=1, padding=1)
        c = window.shape[1]
        kernel = self.gaussian_kernel.to(device=window.device, dtype=window.dtype).repeat(c, 1, 1, 1)
        window = F.conv2d(window, kernel, padding=kernel.shape[-1] // 2, groups=c)
        return window.clamp(0.0, 1.0)

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

    def forward(
        self,
        pred_x0,
        target_x0,
        mask=None,
        masked_x=None,
        use_residual=True,
        use_soft_window=True,
    ):
        pred = pred_x0.float()
        target = target_x0.float()
        mask_lat = self._prepare_mask(mask, pred.shape[-2:], pred.device, pred.dtype)

        if use_residual and masked_x is not None:
            masked = masked_x.float().to(device=pred.device, dtype=pred.dtype)
            if masked.shape[-2:] != pred.shape[-2:]:
                masked = F.interpolate(masked, size=pred.shape[-2:], mode="nearest")
            if masked.shape[1] != pred.shape[1]:
                if masked.shape[1] == 1:
                    masked = masked.repeat(1, pred.shape[1], 1, 1)
                else:
                    raise ValueError("masked_x channel count must match pred_x0 or be 1")
            pred = pred - masked
            target = target - masked

        soft_window = self._build_soft_window(mask_lat) if (mask_lat is not None and use_soft_window) else mask_lat

        loss_ffl = pred_x0.new_tensor(0.0)
        loss_grad = pred_x0.new_tensor(0.0)

        if self.ffl_weight > 0:
            pred_ffl = pred * soft_window if soft_window is not None else pred
            target_ffl = target * soft_window if soft_window is not None else target
            loss_ffl = self.ffl(pred_ffl, target_ffl)
        if self.grad_weight > 0:
            loss_grad = self.masked_gradient_loss(pred, target, mask_lat)

        total_loss = self.ffl_weight * loss_ffl + self.grad_weight * loss_grad
        return total_loss, {"ffl": loss_ffl, "grad": loss_grad}
