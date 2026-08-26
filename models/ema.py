import torch
import torch.nn as nn


class EMA:
    def __init__(
        self, model: nn.Sequential, theta_new: float = 1.0, theta_old: float = 0.0
    ):
        self.theta_new = theta_new
        self.theta_old = theta_old
        self.shadow: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            self.shadow[name] = param.data.clone()

    def update(self, model: nn.Sequential):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            ema = self.theta_new * param.data + self.theta_old * self.shadow[name]
            self.shadow[name] = ema.clone()

    def apply_shadow(self, model: nn.Sequential):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param.data.copy_(self.shadow[name])
