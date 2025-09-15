from typing import Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.layers.adaptive_scaling import AdaptiveScale

Tensor = torch.Tensor

class CondLayer(nn.Module):
    "Conditional Layer Norm"
    def __init__(
        self,
        out_channels: int,
        emb_channels: int,
        case: int = 2,
        film_act_fun: Callable[[Tensor], Tensor] = F.silu,
        act_fun: Callable[[Tensor], Tensor] = None,
        device: Any | None = None, 
        dtype: torch.dtype = torch.float32
    ):
        super(CondLayer, self).__init__()

        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.case = case
        self.film_act_fun = film_act_fun
        self.act_fun = act_fun
        self.device = device
        self.dtype = dtype

        self.norm = nn.GroupNorm(
            min(max(self.out_channels // 4, 1), 32),
            self.out_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.film = AdaptiveScale(
            emb_channels=self.emb_channels,
            input_channels=self.out_channels,
            input_dim=self.case,
            act_fun=self.film_act_fun,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = x.clone()
        h = self.norm(h)
        h = self.film(h, emb)
        if self.act_fun is not None:
            h = self.act_fun(h)
        return h















