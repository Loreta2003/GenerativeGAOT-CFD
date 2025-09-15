# Copyright 2024 The CAM Lab at ETH Zurich.
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

import torch
import torch.distributed as dist
from typing import Sequence

Tensor = torch.Tensor

class StatsRecorder:
    """StatsRecorder which keeps track of metrics for the Ground Truth
    and the Generated Data
    """

    def __init__(
        self,
        batch_size: int,
        ndim: int,
        channels: int,
        data_shape: Sequence[int],
        monte_carlo_samples: int,
        coords: Tensor,
        # time_samples: int = 1, ### For time dependent data we will have time_samples > 1
        num_samples: int = 1000,
        device: torch.device = None,
        world_size: int = 1,
        time_cond: bool = False
    ):
        self.device = device
        self.world_size = world_size

        #information about the dataset
        self.batch_size = batch_size
        self.ndim = ndim
        self.axis = self._set_axis(ndim)
        self.channels = channels
        self.data_shape = data_shape
        self.monte_carlo_samples = monte_carlo_samples
        self.num_samples = num_samples
        self.coords = coords

        #storage for sampled points
        # if time_cond:
        #     self.gen_samples = torch.zeros(
        #         (monte_carlo_samples, time_samples, num_samples, channels), device = device
        #     )
        #     self.gt_samples = torch.zeros(
        #         (monte_carlo_samples, time_samples, num_samples, channels), device = device
        #     )
        # else:
        self.gen_samples = torch.zeros(
            (monte_carlo_samples, num_samples, channels), device = device
        )
        self.gt_samples = torch.zeros(
            (monte_carlo_samples, num_samples, channels), device = device
        )

        #Pre-sample indices for unique points
        self.flat_indices, self.sample_indices = self._sample_unique_indices(
            self.coords, ndim
        )

        #Data_shape should contain (Time, Nodes, Channels) if time_cond is True and (Nodes, Channels) otherwise
        #initialize mean and std
        self.mean_gt = torch.zeros(data_shape, device=device)
        self.mean_gen = torch.zeros(data_shape, device=device)
        self.std_gt = torch.zeros(data_shape, device=device)
        self.std_gen = torch.zeros(data_shape, device=device)

        #global results over all threads / processes
        self.global_mean_gt = (
            torch.zeros(data_shape, device=device) if world_size > 1 else None
        )
        self.global_mean_gen = (
            torch.zeros(data_shape, device=device) if world_size > 1 else None
        )
        self.global_std_gt = (
            torch.zeros(data_shape, device=device) if world_size > 1 else None
        )
        self.global_std_gen = (
            torch.zeros(data_shape, device=device) if world_size > 1 else None
        )
        self.global_observation = 0 if workd_size > 1 else None

        #track the number of observation
        self.observation = 0
    
    def _set_axis(self, ndim: int):
        """Set the axis for the mean and std based on the number of dimensions."""
        if ndim == 2:
            return ( 1, 2) ##TODO Change this to the correct axis for 2D data
        elif ndim == 3:
            return ( 1, 2, 3) ##TODO Change this to the correct axis for 3D data
        else:
            raise ValueError(f"Only 2D and 3D data supported, got {ndim}D data.")
    
    def _sample_unique_indices(self, coords, ndim: int):
        """Sample unique coordinate values to store the generated and gt results"""
        """ For GAOT we might have variable coordinates"""
        #Coords should be of shape [num_nodes, ndim]
        if ndim <2 or ndim >3 or coords.shape[1]!= ndim:
            raise ValueError(f"Invalid coordinates shape: {coords.shape}, expected [num_nodes, {ndim}]")
        
        if self.num_samples > coords.shape[0]:
            raise ValueError(
                f"num_samples ({self.num_samples}) cannot be greater than the number of coordinates ({coords.shape[0]})"
            )
        else:
            flat_indices = torch.randperm(coords.shape[0])[:self.num_samples]
            return flat_indices, coords[flat_indices, :]
    
    def _validate_data(self, gen_data: Tensor, gt_data: Tensor):
        """Validate the dimesnionality and types of gen_data and gt_data."""
        
        if not (isinstance(gen_data, Tensor) and isinstance(gt_data, Tensor)):
            raise TypeError("gen_data and gt_data must be PyTorch Tensors.")
        
        expected_dim = 3 #TODO Check if this shoudl be different for 3D data
        assert gen_data.ndim == expected_dim and gt_data.ndim == expected_dim, (
            f"Expected shape (bs, number of node, channel) for 2D"
            f"got gen_data {gen_data.shape} and gt_data {gt_data.shape}."
        )

    def update_step(self, gen_data: Tensor, gt_data: Tensor):
        """Update all relevant metrics"""

        self._validate_data(gen_data, gt_data)
        self.update_step_mean_and_std(gen_data, gt_data)
        self.sample_points(gen_data, gt_data, self.ndim)

        self.observation += self.batch_size

    def sample_points(self, gen_data: Tensor, gt_data: Tensor, ndim: int):
        """Extract sampled points based on precomputed indices."""

        # Retrieve the current step
        current_step = self.observation  # current number of observations
        final_step = self.observation + self.batch_size

        if ndim == 2 or ndim == 3:
            sampled_gen = gen_data[:, self.flat_indices, :]
            sampled_gt = gt_data[:, self.flat_indices, :]
        else:
            raise ValueError(f"Only 2D or 3D supported, not {self.ndim}D")

        # Store the current Monte Carlo step
        self.gen_samples[current_step:final_step] = sampled_gen
        self.gt_samples[current_step:final_step] = sampled_gt
    
    def update_step_mean_and_std(self, gen_data: Tensor, gt_data: Tensor):
        """Keeps track of the mean of the dataset

        Instead of computing the mean of the error, the ground truth
        and the generated results, this method updates the mean of the
        dataset itself for the gt as well as for the generated data
        """

        m = self.observation  # current number of observations
        n = self.batch_size  # observations added
        # new number of observations would then be m + n
        mean_gt_data = gt_data.mean(dim=0)
        mean_gen_data = gen_data.mean(dim=0)
        std_gt_data = (
            gt_data.std(dim=0)
            if gt_data.size(0) > 1
            else gt_data.std(dim=0, correction=0)
        )
        std_gen_data = (
            gen_data.std(dim=0)
            if gen_data.size(0) > 1
            else gen_data.std(dim=0, correction=0)
        )

        # first we will update the standard deviation before we update the mean
        var_gt = (
            (m / (m + n)) * (self.std_gt**2)
            + (n / (m + n)) * (std_gt_data**2)
            + (m * n / (m + n) ** 2) * (self.mean_gt - mean_gt_data) ** 2
        )
        self.std_gt = torch.sqrt(var_gt)

        var_gen = (
            (m / (m + n)) * (self.std_gen**2)
            + (n / (m + n)) * (std_gen_data**2)
            + (m * n / (m + n) ** 2) * (self.mean_gen - mean_gen_data) ** 2
        )
        self.std_gen = torch.sqrt(var_gen)

        # update the mean values
        self.mean_gt = m / (m + n) * self.mean_gt + n / (m + n) * mean_gt_data
        self.mean_gen = m / (m + n) * self.mean_gen + n / (m + n) * mean_gen_data

    def aggregate_all_processes(self) -> None:
        """Aggregates the results across all processes.

        Take all local results from each rank and produces global solutions for
        the mean and standard deviation as well as the number of observations."""

        # Each thread has a local variance value
        local_var_gt = self.std_gt**2
        local_var_gen = self.std_gen**2

        # number of observation for each thread
        local_obs = self.observation

        # mean values local for each thread
        local_mean_gt = self.mean_gt
        local_mean_gen = self.mean_gen

        # First step is to compute the global mean over all threads
        global_obs = torch.tensor(local_obs, dtype=torch.int, device=self.device)
        dist.all_reduce(global_obs, op=dist.ReduceOp.SUM)
        global_obs = global_obs.item()

        # Global mean estimate with sum(local_obs * local_mean)/global_obs
        global_mean_gt = local_obs * local_mean_gt
        global_mean_gen = local_obs * local_mean_gen
        # sum over all threads
        dist.all_reduce(global_mean_gt, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_mean_gen, op=dist.ReduceOp.SUM)
        global_mean_gt /= global_obs
        global_mean_gen /= global_obs

        # Global var estimate with
        # sum(local_obs * (local_var + (local_mean - global_mean) ** 2)) / n_global
        global_var_gt = local_obs * (
            local_var_gt + (local_mean_gt - global_mean_gt) ** 2
        )
        global_var_gen = local_obs * (
            local_var_gen + (local_mean_gen - global_mean_gen) ** 2
        )
        # sum over all threads
        dist.all_reduce(global_var_gt, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_var_gen, op=dist.ReduceOp.SUM)
        global_var_gt /= global_obs
        global_var_gen /= global_obs

        # Store the results
        self.global_std_gt = torch.sqrt(global_var_gt)
        self.global_std_gen = torch.sqrt(global_var_gen)
        self.global_mean_gt = global_mean_gt
        self.global_mean_gen = global_mean_gen
        self.global_observation = global_obs

    def gather_all_samples(self) -> None:
        """Gather all taken monte carlo samples from each process and store them."""

        tot_gen_samples = torch.zeros(
            (
                self.monte_carlo_samples * self.world_size,
                self.num_samples,
                self.channels,
            ),
            device=self.device,
        )
        tot_gt_samples = torch.zeros(
            (
                self.monte_carlo_samples * self.world_size,
                self.num_samples,
                self.channels,
            ),
            device=self.device,
        )

        local_gen_samples = self.gen_samples
        local_gt_samples = self.gt_samples

        # Gather the results and place them into the tensor tot_gt_samples
        dist.all_gather_into_tensor(tot_gen_samples, local_gen_samples)
        dist.all_gather_into_tensor(tot_gt_samples, local_gt_samples)

        self.gen_samples = tot_gen_samples
        self.gt_samples = tot_gt_samples