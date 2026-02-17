import torch

class VelocityLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_pred, x_gt):
        vel_pred = x_pred[:, 1:, :] - x_pred[:, :-1, :]
        vel_gt = x_gt[:, 1:, :] - x_gt[:, :-1, :]
        return torch.nn.functional.mse_loss(vel_pred, vel_gt, reduction="mean")


class AccelerationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_pred, x_gt):
        acc_pred = x_pred[:, 2:, :] - 2 * x_pred[:, 1:-1, :] + x_pred[:, :-2, :]
        acc_gt = x_gt[:, 2:, :] - 2 * x_gt[:, 1:-1, :] + x_gt[:, :-2, :]
        return torch.nn.functional.mse_loss(acc_pred, acc_gt, reduction="mean")