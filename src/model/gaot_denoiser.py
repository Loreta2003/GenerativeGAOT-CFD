import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from typing import Any, Sequence
from dataclasses import dataclass
import sys
import os
from src.model import GOAT2D_VX
from src.model import GOAT2D_FX
from src.model import GOAT2D_FX_Ablation

Tensor = torch.Tensor

class PreconditionedGAOTDenoiserFX(GOAT2D_FX_Ablation):
    "Preconditioned Denoiser based on Geometry-Aware Operator Transformer (GAOT)"

    def __init__(
        self,
        input_size: int,
        output_size: int, 
        config: Optional[dataclass] = None,
        sigma_data : float = 1.0,
        timesteps: int = 1,
    ):
        super().__init__(
            input_size = input_size,
            output_size = output_size,
            config = config,
            use_encoder_decoder = False
        )

        self.sigma_data = sigma_data
        self.timesteps = timesteps
    
    def forward(
        self,
        latent_tokens_coord: Tensor,
        xcoord: Tensor,
        sigma: Tensor,
        target: Tensor, ##### THIS IS THE INPUT, IT SHOULD BE NOISED VERSION WHEN TRAINING AND COMPLETE NOISE WHEN INFERENCE
        pndata: Tensor = None, #For stacking initial condition
        query_coord: Optional[Tensor] = None,
        encoder_nbrs: Optional[list] = None,
        decoder_nbrs: Optional[list] = None,
        time: Tensor = None,
    ) -> Tensor:

        if sigma.dim() < 1:
            sigma = sigma.expand(pndata.shape[0])

        if sigma.dim() != 1 or pndata.shape[0] != sigma.shape[0]:
            raise ValueError(
                "sigma must be 1D and have the same leading (batch) dim as x"
                f" ({pndata.shape[0]})"
            )
        
        total_var = self.sigma_data**2 + sigma**2
        c_skip = self.sigma_data**2 / total_var
        c_out = sigma * self.sigma_data / torch.sqrt(total_var)
        c_in = 1 / torch.sqrt(total_var)
        c_noise = 0.25 * torch.log(sigma)
        
        expand_shape = [-1] + [1] * (pndata.dim() - 1)
        c_in = c_in.view(*expand_shape)
        c_out = c_out.view(*expand_shape)
        c_skip = c_skip.view(*expand_shape)

        inputs =c_in * target
        if pndata is not None:
            inputs = torch.cat((inputs, pndata), dim=2) 
            B, N, _ = inputs.shape
            sigma_expanded = sigma.view(B, 1, 1)
            sigma_expanded = sigma_expanded.expand(-1, N, 1)
            inputs = torch.cat((inputs, sigma_expanded), dim=-1)
            ###### TODO Check if the concatenation is done on the right dimension ######
        
        if time is not None:
            cond = torch.cat((time, sigma), dim = 1)
            ###### TODO Check if the concatenation is done on the right dimension ######
        else:
            cond = sigma

        f_x = super().forward(latent_tokens_coord = latent_tokens_coord, 
                              xcoord = xcoord, 
                              pndata = inputs,  
                              query_coord = query_coord, 
                              encoder_nbrs = encoder_nbrs,
                              decoder_nbrs = decoder_nbrs,
                              #condition = cond
                             )
        return c_skip * target + c_out * f_x

    

class PreconditionedGAOTDenoiserVX(GOAT2D_VX):
    "Preconditioned Denoiser based on Geometry-Aware Operator Transformer (GAOT)"

    def __init__(
        self,
        input_size: int,
        output_size: int, 
        config: Optional[dataclass] = None,
        sigma_data : float = 1.0,
        timesteps: int = 1,
    ):
        super().__init__(
            input_size = input_size,
            output_size = output_size,
            config = config
        )

        self.sigma_data = sigma_data
        self.timesteps = timesteps
    
    def forward(
        self,
        latent_tokens_coord: Tensor,
        xcoord: Tensor,
        sigma: Tensor,
        target: Tensor, ##### THIS IS THE INPUT, IT SHOULD BE NOISED VERSION WHEN TRAINING AND COMPLETE NOISE WHEN INFERENCE
        pndata: Tensor = None, #For stacking initial condition
        query_coord: Optional[Tensor] = None,
        encoder_nbrs: Optional[list] = None,
        decoder_nbrs: Optional[list] = None,
        time: Tensor = None,
    ) -> Tensor:

        if sigma.dim() < 1:
            sigma = sigma.expand(pndata.shape[0])

        if sigma.dim() != 1 or pndata.shape[0] != sigma.shape[0]:
            raise ValueError(
                "sigma must be 1D and have the same leading (batch) dim as x"
                f" ({pndata.shape[0]})"
            )
        
        total_var = self.sigma_data**2 + sigma**2
        c_skip = self.sigma_data**2 / total_var
        c_out = sigma * self.sigma_data / torch.sqrt(total_var)
        c_in = 1 / torch.sqrt(total_var)
        c_noise = 0.25 * torch.log(sigma)
        
        expand_shape = [-1] + [1] * (pndata.dim() - 1)
        c_in = c_in.view(*expand_shape)
        c_out = c_out.view(*expand_shape)
        c_skip = c_skip.view(*expand_shape)

        inputs =c_in * target
        if pndata is not None:
            inputs = torch.cat((inputs, pndata), dim=2) 
            B, N, _ = inputs.shape
            sigma_expanded = sigma.view(B, 1, 1)
            sigma_expanded = sigma_expanded.expand(-1, N, 1)
            inputs = torch.cat((inputs, sigma_expanded), dim=-1)
            ###### TODO Check if the concatenation is done on the right dimension ######
        
        if time is not None:
            cond = torch.cat((time, sigma), dim = 1)
            ###### TODO Check if the concatenation is done on the right dimension ######
        else:
            cond = sigma

        f_x = super().forward(latent_tokens_coord = latent_tokens_coord, 
                              xcoord = xcoord, 
                              pndata = inputs,  
                              query_coord = query_coord, 
                              encoder_nbrs = encoder_nbrs,
                              decoder_nbrs = decoder_nbrs,
                              #condition = cond
                             )
        return c_skip * target + c_out * f_x