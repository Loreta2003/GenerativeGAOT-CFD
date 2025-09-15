# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic model class for use in gradient descent mini-batch training."""

import dataclasses
from typing import Any, Optional, Mapping, Callable, Union, Sequence
from abc import ABC, abstractmethod
import torch
import torch.profiler
import torch.nn as nn
import numpy as np
import src.diffusion as dfn_lib
from src.trainer.utils.plot import plot_estimates
import matplotlib.pyplot as plt
import wandb

Tensor = torch.Tensor
TensorDict = Mapping[str, Tensor]
BatchType = Mapping[str, Union[np.ndarray, Tensor]]
ModelVariable = Union[dict, tuple[dict, ...], Mapping[str, dict]]
PyTree = Any
LossAndAux = tuple[Tensor, tuple[TensorDict, PyTree]]
Metrics = dict  # Placeholder for metrics that are implemented!


class BaseModel(ABC):
    """Base class for models.

    Wraps flax module(s) to provide interfaces for variable
    initialization, computing loss and evaluation metrics. These interfaces are
    to be used by a trainer to perform gradient updates as it steps through the
    batches of a dataset.

    Subclasses must implement the abstract methods.
    """

    @abstractmethod
    def initialize(self) -> ModelVariable:
        """Initializes variables of the wrapped flax module(s).

        This method by design does not take any sample input in its argument. Input
        shapes are expected to be statically known and used to create
        initialization input for the model. For example::

          import torch.nn as nn

          class MLP(BaseModel):
            def __init__(self, input_shape: tuple[int], hidden_size: int):
              super().__init__()
              self.model = nn.Sequential(
                nn.Linear(np.prod(input_shape), hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, np.pord(input_shape))
              )
              self.input_shape = input_shape

        Returns:
          The initial variables for this model - can be a single or a tuple/mapping
          of PyTorch variables.
        """
        raise NotImplementedError

    @abstractmethod
    def loss_fn(
        self,
        params: Union[PyTree, tuple[PyTree, ...]],
        batch: BatchType,
        mutables: PyTree,
        **kwargs,
    ) -> LossAndAux:
        """Computes training loss and metrics.

        It is expected that gradient would be taken (via `jax.grad`) wrt `params`
        during training. This can also be required if via torch autograd is used!

        Arguments:
          params: model parameters wrt which the loss would be differentiated.
          batch: a single batch of data.
          mutables: model variables which are not differentiated against; can be
            mutable if so desired.
          **kwargs: additional static configs.

        Returns:
          loss: the (scalar) loss function value.
          aux: two-item auxiliary data consisting of
            metric_vars: a dict with values required for metric compute and logging.
              They can either be final metric values computed inside the function or
              intermediate values to be further processed into metrics.
            mutables: non-differentiated model variables whose values may change
              during function execution (e.g. batch stats).
        """
        raise NotImplementedError

    def eval_fn(
        self,
        variables: Union[tuple[PyTree, ...], PyTree],
        batch: BatchType,
        **kwargs,
    ) -> TensorDict:
        """Computes evaluation metrics."""
        raise NotImplementedError

    @staticmethod
    def inference_fn(variables: PyTree, **kwargs) -> Callable[..., Any]:
        """Returns an inference function with bound variables."""
        raise NotImplementedError


"""Training a denoising model for diffusion-based generation."""

@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingModelGAOT2D(BaseModel):

    denoiser: nn.Module
    noise_sampling: dfn_lib.NoiseLevelSampling
    noise_weighting: dfn_lib.NoiseLossWeighting
    num_eval_noise_levels: int = 5
    num_eval_cases_per_lvl: int = 1
    min_eval_noise_lvl: float = 1e-3
    max_eval_noise_lvl: float = 50.0

    consistent_weight: float = 0
    device: Any | None = None
    dtype: torch.dtype = torch.float32

    timesteps: int = 1
    has_query_coord : bool = False

    def initialize(
        self,
        batch_size: int,
        num_nodes_x: int,
        num_nodes_latent: int,
        num_nodes_query: int,
        timesteps: int = 1,
        input_channels: int = 1,
        output_channels: int = 1,
    ):
        """Method necessary for a dummy initialization!"""

        u = torch.ones(
            (batch_size,) + (timesteps, ) + (num_nodes_query,) + (output_channels,),
            dtype=self.dtype,
            device=self.device,
        )  # Target condition
        c = torch.ones(
            (batch_size,) + (timesteps, ) + (num_nodes_x,) + (input_channels,),
            dtype=self.dtype,
            device=self.device,
        )  # Initial condition

        x = torch.ones(
            (batch_size,) + (timesteps, ) + (num_nodes_x, ) + (2, ),
            dtype = self.dtype,
            device = self.device,
        )  # x coordinates

        latentx = torch.ones(
            (num_nodes_latent, ) + (2, ),
            dtype = self.dtype,
            device = self.device,
        )  # latent token coordinates

        queryx = torch.ones(
            (batch_size, ) + (timesteps, ) + (num_nodes_query, ) + (2, ),
            dtype = self.dtype,
            device = self.device,
        )  # latent token coordinates

        if timesteps == 1:
            time = None
        else:
            time = torch.ones((batch_size,), dtype=self.dtype, device=self.device)

        return self.denoiser(
            xcoord = x,
            latent_tokens_coord = latentx,
            query_coord = queryx,
            pndata = c,
            target = u,
            time = time,
            sigma = torch.ones((batch_size,), dtype=self.dtype, device=self.device),
            encoder_nbrs = None,
            decoder_nbrs = None,
        )

    def loss_fn(self, batch: dict, latent_x_array, encoder_nbrs, decoder_nbrs, global_step, mutables: Optional[dict] = None):
        """Computes the denoising loss on a training batch.

        Args:
          batch: A batch of training data expected to contain an `c` field with a
            shape of `(batch, timesteps, nodes, channel_info)`, representing the unnoised
            samples. Optionally, it may also contain a `cond` field, which is a
            dictionary of conditional inputs.
          mutables: The mutable (non-diffenretiated) parameters of the denoising
            model (e.g. batch stats); *currently assumed empty*.

        Returns:
          The loss value and a tuple of training metric and mutables.
        """

        c_array = batch["initial_cond"]
        u_array = batch["target_cond"]
        x_array = batch["x_coord"]
        query_x_array = batch["query_coord"] if self.has_query_coord else None
        time = None if self.timesteps == 1 else batch["lead_time"]

        batch_size = len(u_array)

        u_array_squared = torch.square(u_array)

        sigma = self.noise_sampling(shape=(batch_size,))

        weights = self.noise_weighting(sigma)
        if weights.ndim != u_array.ndim:
            weights = weights.view(-1, *([1] * (u_array.ndim - 1)))
        
        noise = torch.randn(u_array.shape, dtype = self.dtype, device = self.device)

        if sigma.ndim != u_array.ndim:
            noised = u_array + noise * sigma.view(-1, *([1] * (u_array.ndim - 1)))
        else:
            noised = u_array + noise * sigma
        
        denoised = self.denoiser.forward(xcoord = x_array,
                                             latent_tokens_coord = latent_x_array,
                                             query_coord = query_x_array,
                                             pndata = c_array,
                                             target = noised,
                                             sigma = sigma,
                                             encoder_nbrs = encoder_nbrs,
                                             decoder_nbrs = decoder_nbrs,
                                             time = time)
        #IF VX
        fig = plot_estimates(
            u_inp = noised[-1].detach().cpu().numpy(),
            u_gtr = u_array[-1].cpu().numpy(),
            u_prd = denoised[-1].detach().cpu().numpy(),
            x_inp = x_array[-1].cpu().numpy(),
            x_out = x_array[-1].cpu().numpy(),
        )
        # #IF FX
        # fig = plot_estimates(
        #     u_inp = noised[-1].detach().cpu().numpy(),
        #     u_gtr = u_array[-1].cpu().numpy(),
        #     u_prd = denoised[-1].detach().cpu().numpy(),
        #     x_inp = x_array.cpu().numpy(),
        #     x_out = x_array.cpu().numpy(),
        # )
        LOG_EVERY = 50
        if wandb.run is not None and (global_step % LOG_EVERY == 0):
            wandb.log(
                {"denoising_example": wandb.Image(fig)},
                step = global_step
            )
        fig.savefig("/cluster/work/math/larzumanya/GAOT/.results/time_indep/denoising.png" ,dpi=300,bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        #print(f"Plot saved to cluster/work/math/larzumanya/GAOT/results/time_indep/denoising.png")
        
        denoised_squared = torch.square(denoised)

        rel_norm = torch.mean(torch.square(u_array) / torch.mean(torch.square(u_array_squared)))
        loss = torch.mean(weights * torch.square(denoised - u_array))
        loss += (
            self.consistent_weight
            * rel_norm
            * torch.mean(weights * torch.square(denoised_squared - u_array_squared))
        )

        metrics = {"loss": loss.item()}

        return loss, metrics
    
    def eval_fn(self, batch: dict, latent_x_array, encoder_nbrs, decoder_nbrs) -> dict:
        """Compute denoising metrics on an eval batch.

        Randomly selects members of the batch and noise them to a number of fixed
        levels. Each level is aggregated in terms of the average L2 error.

        Args:
          variables: Variables for the denoising module.
          batch: A batch of evaluation data expected to contain an `c` field with a
            shape of `(batch, timesteps, nodes, channel_info)`, representing the unnoised
            samples. Optionally, it may also contain a `cond` field, which is a
            dictionary of conditional inputs.

        Returns:
          A dictionary of denoising-based evaluation metrics.
        """
        initial_cond = batch["initial_cond"]
        target_cond = batch["target_cond"]
        x_coords = batch["x_coord"]
        query_x_coords = batch["query_coord"] if self.has_query_coord else None
        time = None if self.timesteps == 1 else batch["lead_time"]

        rand_idx_set = torch.randint(
            0,
            initial_cond.shape[0],
            (self.num_eval_noise_levels, self.num_eval_cases_per_lvl),
            device = self.device,
        )

        c_array = initial_cond[rand_idx_set]
        u_array = target_cond[rand_idx_set]
        x_array = x_coords[rand_idx_set]

        if time is not None:
            time_inputs = time[rand_idx_set]
        
        if query_x_coords is not None:
            query_x_array = query_x_coords[rand_idx_set]
        
        sigma = torch.exp(
            torch.linspace(
                np.log(self.min_eval_noise_lvl),
                np.log(self.max_eval_noise_lvl),
                self.num_eval_noise_levels,
                dtype = self.dtype,
                device = self.device,
            )
        )

        noise = torch.randn(u_array.shape, device=self.device, dtype = self.dtype)

        if sigma.ndim != u_array.ndim:
            noised = u_array + noise * sigma.view(-1, *([1] * (u_array.ndim - 1)))
        else:
            noised = u_array + noise * sigma
        
        denoise_fn = self.inference_fn(
            denoiser = self.denoiser,
            lead_time = False if time is None else True,
        )
        #Defined cond to pass in the sampler for extra arguments that normal diffusion models do not require.
        if time is not None:
            if query_x_coords is not None:
                denoised = torch.stack(
                    [
                        denoise_fn(
                            x = c_array[i], 
                            y = noised[i], 
                            sigma = sigma[i].unsqueeze(0), 
                            cond = dict(xcoord = x_array[i], latent_tokens_coord = latent_x_array, query_coord = query_x_array[i], encoder_nbrs = [encoder_nbrs[i]], decoder_nbrs = [decoder_nbrs[i]]),
                            time = time_inputs[i])
                        for i in range(self.num_eval_noise_levels)
                    ]
                )
            else:
                denoised = torch.stack(
                    [
                        denoise_fn(
                            x = c_array[i], 
                            y = noised[i], 
                            sigma = sigma[i].unsqueeze(0), 
                            cond = dict(xcoord = x_array[i], latent_tokens_coord = latent_x_array, query_coord = None, encoder_nbrs = [encoder_nbrs[i]], decoder_nbrs = [decoder_nbrs[i]]),
                            time = time_inputs[i]
                        )
                        for i in range(self.num_eval_noise_levels)
                    ]
                )
        else:
            if query_x_coords is not None:
                denoised = torch.stack(
                    [
                        denoise_fn(
                            x = c_array[i], 
                            y = noised[i], 
                            sigma = sigma[i].unsqueeze(0), 
                            cond = dict(xcoord = x_array[i], latent_tokens_coord = latent_x_array, query_coord = query_x_array[i], encoder_nbrs = [encoder_nbrs[i]], decoder_nbrs = [decoder_nbrs[i]])
                        )
                        for i in range(self.num_eval_noise_levels)
                    ]
                )
            else:
                denoised = torch.stack(
                    [
                        denoise_fn(
                            x = c_array[i], 
                            y = noised[i], 
                            sigma = sigma[i].unsqueeze(0), 
                            cond = dict(xcoord = x_array[i], latent_tokens_coord = latent_x_array, query_coord = None, encoder_nbrs = [encoder_nbrs[i]], decoder_nbrs = [decoder_nbrs[i]])
                        )
                        for i in range(self.num_eval_noise_levels)
                    ]
                )

        ema_losses = torch.mean(
            torch.square(denoised - u_array), dim=[i for i in range(1, u_array.ndim)]
        )
        eval_losses = {
            f"denoise_lvl{i}": loss.item() for i, loss in enumerate(ema_losses)
        }
        return eval_losses[f"denoise_lvl{self.num_eval_noise_levels - 1}"]

    @staticmethod
    def inference_fn(
        denoiser: nn.Module, lead_time: bool = False
    ) -> Tensor:
        """Returns the inference denoising function.
        Args:
          denoiser: Neural Network (NN) Module for the forward pass
          lead_time: If set to True it can be used for datasets which have time
            included. This time value can then be used for conditioning. Commonly
            done for an All2All training strategy.

        Return:
          _denoise: corresponding denoise function
        """
        
        denoiser.eval()

        if lead_time == False:

            def _denoise(
                x: Tensor,
                sigma: float | Tensor,
                y: Tensor,
                cond: Mapping[str, Tensor] | None = None,
            ) -> Tensor:

                if not torch.is_tensor(sigma):
                    sigma = sigma * torch.ones((x.shape[0],))

                k = denoiser.forward(xcoord = cond["xcoord"], 
                                        latent_tokens_coord = cond["latent_tokens_coord"], 
                                        pndata = y, # The names of x and y are flipped here
                                        # This is because the denoiser expects the input to be the noisy version
                                        # and the target to be the complete noise
                                        target = x, #This should be complete noise when inference
                                        sigma=sigma,
                                        query_coord = cond["query_coord"],
                                        encoder_nbrs = cond["encoder_nbrs"],
                                        decoder_nbrs = cond["decoder_nbrs"])
                return k

        elif lead_time == True:

            def _denoise(
                x: Tensor,
                sigma: float | Tensor,
                y: Tensor,
                time: float | Tensor,
                cond: Mapping[str, Tensor] | None = None,
            ) -> Tensor:

                if not torch.is_tensor(sigma):
                    sigma = sigma * torch.ones((x.shape[0],))

                if not torch.is_tensor(time):
                    time = time * torch.ones((x.shape[0],))

                return denoiser.forward(xcoord=cond["xcoord"],
                                        latent_tokens_coord=cond["latent_tokens_coord"],
                                        pndata=y, 
                                        target=x,
                                        sigma = sigma,
                                        time = time,
                                        query_coord=cond["query_coord"],
                                        encoder_nbrs=cond["encoder_nbrs"],
                                        decoder_nbrs=cond["decoder_nbrs"])  
        else:
            raise ValueError(
                "Lead Time needs to be a boolean, if a time condition is required"
            )

        return _denoise