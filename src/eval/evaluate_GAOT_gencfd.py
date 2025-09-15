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

"""Main File to Run Inference.

Options are to compute statistical metrics or visualize results.
"""

import time
import os
import sys
import torch
import torch.distributed as dist

from GenCFD.train.train_states import DenoisingModelTrainState
from GenCFD.utils.parser_utils import inference_args
from GenCFD.utils.dataloader_builder import get_dataset_loader
from GenCFD.utils.gencfd_builder import (
    create_denoiser,
    create_sampler,
    load_json_file,
    replace_args,
)
from GenCFD.utils.denoiser_utils import get_latest_checkpoint

from src.eval.metrics.stats_recorder import StatsRecorder
from src.eval import evaluation_loop
from src import diffusion as dfn_lib
from src import solvers

from argparse import Namespace


torch.set_float32_matmul_precision("high")  # Better performance on newer GPUs!
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# Setting global seed for reproducibility
torch.manual_seed(SEED)  # For CPU operations
torch.cuda.manual_seed(SEED)  # For GPU operations
torch.cuda.manual_seed_all(SEED)  # Ensure all GPUs (if multi-GPU) are set


def init_distributed_mode(args):
    """Initialize a Distributed Data Parallel Environment"""

    args.local_rank = int(os.getenv("LOCAL_RANK", -1))  # Get from environment variable

    if args.local_rank == -1:
        raise ValueError(
            "--local_rank was not set. Ensure torchrun is used to launch the script."
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl", rank=args.local_rank, world_size=args.world_size
        )
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        dist.init_process_group(
            backend="gloo", rank=args.local_rank, world_size=args.world_size
        )
        device = torch.device("cpu")
        print(" ")

    print(f"DDP initialized with rank {args.local_rank} and device {device}.")

    return args, device

def run_eval_loop(args, denoising_model):
    """Run the evaluation loop with the given arguments and model."""
    args = Namespace(
        local_rank=,
        world_size=,
        monte_carlo_samples=,
        batch_size=,
    )

    # Initialize distributed mode (if multi-GPU)
    if args.world_size > 1:
        args, device = init_distributed_mode(args)
    else:
        print(" ")
        print(f"Used device: {device}")

    # Print number of Parameters:
    model_params = sum(
        p.numel() for p in denoising_model.denoiser.parameters() if p.requires_grad
    )
    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        print(" ")
        print(f"Total number of model parameters: {model_params}")
        print(" ")

    denoise_fn = denoising_model.inference_fn(
        denoiser=denoising_model.denoiser,
        lead_time=time_cond
    )

    diffusion_scheme = dfn_lib.Diffusion.create_variance_exploding(
        sigma=dfn_lib.tangent_noise_schedule(device=device),
        data_std=denoising_model.denoiser.sigma_data,
    )

    integrator = solvers.EulerMaruyama(
        time_axis_pos=0,
        terminal_only=True
    )

    tspan = dfn_lib.edm_noise_decay(
        scheme = diffusion_scheme,
        rho = 7,
        #Minimum of 30 steps required
        num_steps = 128,
        end_sigma = 1e-3,
        dtype = torch.float32,
        device = device
    )
    
    # Create Sampler
    sampler = dfn_lib.SdeSampler(
        input_shape = y_sample.shape[1:], ###TODO Change the y_sample to the correct shape
        scheme = diffusion_scheme,
        denoise_fn = denoise_fn,
        tspan = tspan,
        integrator = integrator,
        guidance_transforms = (),
        apply_denoise_at_end = True,
        return_full_paths = False,
        device = device,
        dtype = torch.float32
    )

    # compute the effective number of monte carlo samples if world_size is greater than 1
    if args.world_size > 1:
        if args.monte_carlo_samples % args.world_size != 0:
            if args.local_rank == 0:
                print(
                    "Number of monte carlo samples should be divisible through the number of processes used!"
                )

        effective_samples = (
            args.monte_carlo_samples // (args.world_size * args.batch_size)
        ) * (args.world_size * args.batch_size)

        if effective_samples <= 0:
            error_msg = (
                f"Invalid configuration: Number of Monte Carlo samples ({args.monte_carlo_samples}), "
                f"batch size ({args.batch_size}), and world size ({args.world_size}) result in zero effective samples. "
                f"Ensure monte_carlo_samples >= world_size * batch_size."
            )
            if args.local_rank == 0:
                print(error_msg)
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(0)

    # Initialize stats_recorder to keep track of metrics
    stats_recorder = StatsRecorder(
        batch_size=args.batch_size,
        ndim=len(out_shape) - 1, ## TODO Change this to the correct number of dimensions
        channels=dataset.output_channel, ### TODO Change this to the correct number of channels
        data_shape=out_shape, ### TODO Change this to the correct output shape
        monte_carlo_samples=(
            args.monte_carlo_samples
            if args.world_size <= 1
            else effective_samples // args.world_size
        ),
        num_samples=1000,
        device=device,
        world_size=args.world_size,
    )

    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        if args.compute_metrics:
            tot_samples = (
                args.monte_carlo_samples if args.world_size <= 1 else effective_samples
            )
            print(
                f"Run Evaluation Loop with {tot_samples} Monte Carlo Samples and Batch Size {args.batch_size}"
            )
        if args.visualize:
            print(f"Run Visualization Loop")

    start_train = time.time()

    if args.world_size > 1:
        dist.barrier(device_ids=[args.local_rank])

    evaluation_loop.run(
        sampler=sampler,
        monte_carlo_samples=(
            args.monte_carlo_samples if args.world_size <= 1 else effective_samples
        ),
        stats_recorder=stats_recorder,
        # Dataset configs
        dataloader=dataloader,  ## TODO Change this to the correct dataloader
        time_cond=time_cond,
        # Denoising configs
        latent_tokens_coord = , ## TODO Change this to the correct latent tokens coordinates
        query_coord = , ## TODO Change this to the correct query coordinates
        # Eval configs
        compute_metrics=args.compute_metrics,
        visualize=args.visualize,
        save_gen_samples=args.save_gen_samples,
        device=device,
        save_dir=args.save_dir,
        #For denormalization
        u_std = , ## TODO Change this to the correct u_std
        u_mean = , ## TODO Change this to the correct u_mean
        c_std = , ## TODO Change this to the correct c_std
        c_mean = , ## TODO Change this to the correct c_mean
        # DDP configs
        world_size=args.world_size,
        local_rank=args.local_rank,
    )

    end_train = time.time()
    elapsed_train = end_train - start_train
    if (args.world_size > 1 and args.local_rank == 0) or args.world_size == 1:
        print(" ")
        print(f"Done evaluation. Elapsed time {elapsed_train / 3600} h")

    if dist.is_initialized():
        dist.destroy_process_group()
