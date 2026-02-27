import torch

from student_model_v3.losses import HighFreqTextLoss


def run_self_check():
    torch.manual_seed(0)
    b, c, h, w = 2, 4, 16, 16
    pred_x0 = torch.randn(b, c, h, w, requires_grad=True)
    target_x0 = torch.randn(b, c, h, w)
    masked_x = torch.randn(b, c, h, w)
    mask = (torch.rand(b, 1, h * 2, w * 2) > 0.5).float()

    criterion = HighFreqTextLoss(
        ffl_weight=0.05,
        grad_weight=0.05,
        text_weight=5.0,
    )
    total, stats = criterion(
        pred_x0=pred_x0,
        target_x0=target_x0,
        mask=mask,
        masked_x=masked_x,
    )

    assert total.ndim == 0, "total loss should be a scalar"
    assert "ffl" in stats and "grad" in stats
    assert torch.isfinite(total), "total loss should be finite"
    total.backward()
    assert pred_x0.grad is not None, "pred_x0 should receive gradients"
    assert torch.isfinite(pred_x0.grad).all(), "gradients should be finite"


if __name__ == "__main__":
    run_self_check()
    print("highfreq residual self-check passed")
