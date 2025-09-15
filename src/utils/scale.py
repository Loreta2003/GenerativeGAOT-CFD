import torch 
from dataclasses import dataclass
from typing import Union, Tuple

EPSILON = 1e-8 

def rescale(x:torch.Tensor, lims=(-1,1))->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        ND tensor
    
    Returns
    -------
    x_normalized: torch.Tensor
        ND tensor
    """
    return (x-x.min()) / (x.max()-x.min()) * (lims[1] - lims[0]) + lims[0]


class CoordinateScaler:
    def __init__(self, target_range = (-1, 1), mode = "per_dim_scaling"):
        """
        Parameters
        ----------
        target_range: Tuple[float, float]
            The range to scale the coordinates to.
        mode: str
            Scaling strategy: 'global_scaling' or 'per_dim_scaling'.
        """
        if not isinstance(target_range, (tuple, list)) or len(target_range) != 2:
            raise ValueError("target_range must be a tuple or list of length 2.")
        if mode not in ['global_scaling', 'per_dim_scaling']:
            raise ValueError("mode must be either 'global_scaling' or 'per_dim_scaling'.")
        self.target_range = target_range
        self.out_min, self.out_max = target_range
        self.mode = mode
    
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            ND tensor, [num_nodes, num_dimensions] (eg: [N,2] or [N,3]).
        Returns
        -------
        x_scaled: torch.Tensor
            ND tensor scaled to the target range.
        """

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"x must be a torch.Tensor. but got {type(x)}")
        if x.numel() == 0:
            raise ValueError("x is empty.")
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor.")
        if self.mode == 'global_scaling':
            x_min = x.min()
            x_max = x.max()
        elif self.mode == 'per_dim_scaling':
            x_min = torch.min(x, dim=0, keepdim=True)[0] # Shape: [1, num_dimensions]
            x_max = torch.max(x, dim=0, keepdim=True)[0] # Shape: [1, num_dimensions]

        x_range = x_max - x_min + EPSILON
        normalized = (x - x_min) / x_range

        scaled_x = normalized * (self.out_max - self.out_min) + self.out_min
        
        return scaled_x

@dataclass
class MeanStd:
    mean:torch.Tensor
    std:torch.Tensor

def normalize(x:torch.Tensor, mean=None, std=None, return_mean_std:bool=False
              )->Union[torch.Tensor, Tuple[torch.Tensor, MeanStd]]:
    """
    Parameters
    ----------
    x: torch.Tensor
        ND tensor
    mean: Optional[float]
        mean of the data
    std: Optional[float]
        standard deviation of the data
    
    Returns
    -------
    x_normalized: torch.Tensor
        1D tensor
    """
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    if return_mean_std:
        return (x - mean) / std, MeanStd(mean, std)
    else:
        return (x - mean) / std
