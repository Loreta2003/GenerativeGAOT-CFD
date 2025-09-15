import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import xarray as xr
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from .base import TrainerBase
from .utils.train_setup import manual_seed
from .utils.data_pairs import DynamicPairDataset, TestDataset
from .utils.cal_metric import compute_batch_errors, compute_final_metric
from .utils.plot import plot_estimates
from .utils.io_norm import compute_stats

from src.utils.dataclass import shallow_asdict
from src.data.dataset import Metadata, DATASET_METADATA
from src.model import init_model
from src.utils.scale import rescale, CoordinateScaler

EPSILON = 1e-10

class SequentialTrainer_FX(TrainerBase):
    """
    Trainer for sequential data using a single dataset and a neural operator model.
    """

    def __init__(self, args):
        """
        Initialize the SequentialTrainer.

        Args:
            args (Namespace): Configuration arguments for the trainer.
        """
        super().__init__(args)

    def _load_and_split_data(self, dataset_config):
        """
        Load and split the dataset into training and testing sets.

        Args:
            dataset_config (dict): Configuration for the dataset.

        Returns:
            tuple: Training and testing datasets.
        """
        # --- Load the dataset ---
        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        self.poseidon_dataset_name = ["CE-RP","CE-Gauss","NS-PwC",
                                      "NS-SVS","NS-Gauss","NS-SL",
                                      "ACE", "Wave-Layer"]
        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None
            if self.metadata.group_x is not None:
                x_array = ds[self.metadata.group_x].values  # Shape: [1, 1, num_nodes, num_dims]
            else:
                # Generate x coordinates if not available, but sequential data doesn't need it currenty  (e.g., for structured grids)
                domain_x = self.metadata.domain_x  # ([xmin, ymin], [xmax, ymax])
                nx, ny = u_array.shape[2], u_array.shape[3]  # Spatial dimensions
                x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
                y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')  # [nx, ny]
                x_array = np.stack([xv, yv], axis=-1)  # [nx, ny, num_dims]
                x_array = x_array.reshape(-1, 2)  # [num_nodes, num_dims]
                x_array = x_array[None, None, ...]  # Add sample and time dimensions
        # --- Dataset Specific Handling ---
        if dataset_name in self.poseidon_dataset_name and dataset_config.use_sparse:
            u_array = u_array[:,:,:9216,:] 
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            x_array = x_array[:,:,:9216,:]
        
        # --- Select Variables & Check Shapes ---
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]

        # --- Compute dataset sizes ---
        total_samples = u_array.shape[0]
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"

        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        u_train = np.ascontiguousarray(u_array[train_indices])
        u_val = np.ascontiguousarray(u_array[val_indices])
        u_test = np.ascontiguousarray(u_array[test_indices])

        if c_array is not None:
            c_train = np.ascontiguousarray(c_array[train_indices])
            c_val = np.ascontiguousarray(c_array[val_indices])
            c_test = np.ascontiguousarray(c_array[test_indices])
        else:
            c_train = c_val = c_test = None

        if self.metadata.domain_t is not None:
            t_start, t_end = self.metadata.domain_t
            t_values = np.linspace(t_start, t_end, u_array.shape[1]) # shape: [num_timesteps]
        else:
            raise ValueError("metadata.domain_t is None. Cannot compute actual time values.")
        
        return {
            "train": {"c": c_train, "u": u_train, "x": x_array[0, 0], "t": t_values},
            "val": {"c": c_val, "u": u_val, "x": x_array[0, 0], "t": t_values},
            "test": {"c": c_test, "u": u_test, "x": x_array[0, 0], "t": t_values},
        }

    def _generate_latent_queries(self, token_size = (64, 64), coord_scaler = None):
        """Generates latent query points on a regular grid."""
        phy_domain = self.metadata.domain_x
        x_min, y_min = phy_domain[0]
        x_max, y_max = phy_domain[1]

        if not isinstance(token_size[0], int) or not isinstance(token_size[1], int):
            raise ValueError("token_size must be a tuple of two integers.")

        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, token_size[0], dtype=self.dtype), 
            torch.linspace(y_min, y_max, token_size[1], dtype=self.dtype), 
            indexing='ij' 
        )
        latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1,2)
        latent_queries = coord_scaler(latent_queries)

        return latent_queries

    def _collate_fn(self, batch):
        """
        Custom collate function to prepare batches for the DataLoader.

        Args:
            batch (list): List of tuples where each tuple contains input and output arrays.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing batched inputs and outputs as tensors.
        """

        input_list, output_list = zip(*batch) # unzip the batch, both inputs and outputs are lists of tuples
        inputs = np.stack(input_list) # shape: [batch_size, num_nodes, input_dim]
        outputs = np.stack(output_list) # shape: [batch_size, num_nodes, output_dim]

        inputs = torch.tensor(inputs, dtype=self.dtype)
        outputs = torch.tensor(outputs, dtype=self.dtype)

        return inputs, outputs

    def init_dataset(self, dataset_config):
        # --- 1. Load and split the dataset ---
        print("Loading and preprocessing data...")
        data_splits = self._load_and_split_data(dataset_config)
        # Extract tensors for convenience
        c_train, u_train, x_train, t_train = data_splits["train"].values()
        c_val, u_val, x_val, t_val = data_splits["val"].values()
        c_test, u_test, x_test, t_test = data_splits["test"].values()

        # --- 2. Prepare for Latent Tokens --- 
        coord_scaler = CoordinateScaler(target_range=(-1, 1))
        latent_queries = self._generate_latent_queries(
            token_size = self.model_config.latent_tokens_size,
            coord_scaler = coord_scaler
        )
        self.latent_tokens_coord = latent_queries
        self.coord = coord_scaler(torch.tensor(x_train, dtype=self.dtype))

        # --- 3. Create datasets ---
        max_time_diff = getattr(dataset_config, "max_time_diff", None)
        self.stats = compute_stats(u_train, c_train, t_train, self.metadata,max_time_diff,
                                  sample_rate=dataset_config.sample_rate,
                                  use_metadata_stats=dataset_config.use_metadata_stats,
                                  use_time_norm=dataset_config.use_time_norm)
        self.train_dataset = DynamicPairDataset(u_train, c_train, t_train, self.metadata, 
                                                max_time_diff = max_time_diff, 
                                                stepper_mode=dataset_config.stepper_mode,
                                                stats=self.stats,
                                                use_time_norm = dataset_config.use_time_norm)
        self.val_dataset = DynamicPairDataset(u_val, c_val, t_val, self.metadata, 
                                              max_time_diff = max_time_diff, 
                                              stepper_mode=dataset_config.stepper_mode,
                                              stats=self.stats,
                                              use_time_norm = dataset_config.use_time_norm)
        self.test_dataset = DynamicPairDataset(u_test, c_test, t_test, self.metadata, 
                                               max_time_diff = max_time_diff, 
                                               stepper_mode=dataset_config.stepper_mode,
                                               stats=self.stats,
                                               use_time_norm = dataset_config.use_time_norm)

        
        batch_size = dataset_config.batch_size
        shuffle = dataset_config.shuffle
        num_workers = dataset_config.num_workers
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def init_model(self, model_config):
        """
        Initialize the model based on the provided configuration.

        Args:
            model_config (dict): Configuration for the model.
        """
        in_channels = self.stats["u"]["mean"].shape[0] + 2 # add lead time and time difference
        
        if model_config.use_conditional_norm:
            in_channels = in_channels - 1 

        if "c" in self.stats:
            in_channels += self.stats["c"]["mean"].shape[0]

        out_channels = self.stats["u"]["mean"].shape[0]
        
        self.model = init_model(
            input_size=in_channels,
            output_size=out_channels,
            model=model_config.name,
            config=model_config
        ).to(self.device)
    
    def train_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing batched inputs and outputs.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """

        batch_inputs, batch_outputs = batch
        batch_inputs = batch_inputs.to(self.device) # Shape: [batch_size, num_nodes, num_channels]
        batch_outputs = batch_outputs.to(self.device) 
        latent_tokens_coord = self.latent_tokens_coord.to(self.device) # Shape: [num_latent_tokens, 2]
        coord = self.coord.to(self.device)

        if self.model_config.use_conditional_norm:
            pred = self.model(
                latent_tokens_coord = latent_tokens_coord,
                xcoord = coord,
                pndata = batch_inputs[...,:-1],
                condition = batch_inputs[...,0,-2:-1]
            ) # [batch_size, num_nodes, num_channels]
        else:
            pred = self.model(
                latent_tokens_coord = latent_tokens_coord,
                xcoord = coord,
                pndata = batch_inputs
            ) # [batch_size, num_nodes, num_channels]
        
        return self.loss_fn(pred, batch_outputs)
    
    def validate(self, loader):
        """
        Validate the model on a given dataset loader.

        Args:
            loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            float: The average loss over the validation dataset.
        """

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_outputs in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_outputs = batch_outputs.to(self.device)
                latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                coord = self.coord.to(self.device)

                if self.model_config.use_conditional_norm:
                    pred = self.model(
                        latent_tokens_coord = latent_tokens_coord,
                        xcoord = coord,
                        pndata = batch_inputs[...,:-1],
                        condition = batch_inputs[...,0,-2:-1]
                    )
                else:
                    pred = self.model(
                        latent_tokens_coord = latent_tokens_coord,
                        xcoord = coord,
                        pndata = batch_inputs
                    )

                loss = self.loss_fn(pred, batch_outputs)
                total_loss += loss.item()
        return total_loss / len(loader)
    
    def autoregressive_predict(self, x_batch, time_indices):
        """
        Autoregressive prediction of the output variables at the specified time indices.

        Args:
            x_batch (torch.Tensor): Initial input batch at time t=0. Shape: [batch_size, num_nodes, input_dim]
            time_indices (np.ndarray): Array of time indices for prediction (e.g., [0, 2, 4, ..., 14])

        Returns:
            torch.Tensor: Predicted outputs over time. Shape: [batch_size, num_timesteps - 1, num_nodes, output_dim]
        """
        
        batch_size, num_nodes, input_dim = x_batch.shape
        num_timesteps = len(time_indices)

        predictions = []

        t_values = self.test_dataset.t_values
        start_times_mean = self.stats["start_time"]["mean"]
        start_times_std = self.stats["start_time"]["std"]
        time_diffs_mean = self.stats["time_diffs"]["mean"]
        time_diffs_std = self.stats["time_diffs"]["std"]

        u_mean = torch.tensor(self.stats["u"]["mean"], dtype=self.dtype).to(self.device)
        u_std = torch.tensor(self.stats["u"]["std"], dtype=self.dtype).to(self.device)

        u_in_dim = self.stats["u"]["mean"].shape[0]
        c_in_dim = self.stats["c"]["mean"].shape[0] if "c" in self.stats else 0
        time_feature_dim = 2
        if c_in_dim > 0:
            c_in = x_batch[..., u_in_dim:u_in_dim+c_in_dim] # Shape: [batch_size, num_nodes, c_in_dim]
        else:
            c_in = None
        
        current_u_in = x_batch[..., :u_in_dim] # Shape: [batch_size, num_nodes, u_in_dim]
        
        for idx in range(1, num_timesteps):
            t_in_idx = time_indices[idx-1]
            t_out_idx = time_indices[idx]
            start_time = t_values[t_in_idx]
            time_diff = t_values[t_out_idx] - t_values[t_in_idx]
            
            start_time_norm = (start_time - start_times_mean) / start_times_std
            time_diff_norm = (time_diff - time_diffs_mean) / time_diffs_std

            # Prepare time features (expanded to match num_nodes)
            start_time_expanded = torch.full((batch_size, num_nodes, 1), start_time_norm, dtype=self.dtype).to(self.device)
            time_diff_expanded = torch.full((batch_size, num_nodes, 1), time_diff_norm, dtype=self.dtype).to(self.device)

            input_features = [current_u_in]  # Use the previous u_in (either initial or previous prediction)
            if c_in is not None:
                input_features.append(c_in)  # Use the same c_in as in x_batch (assumed constant over time)
            input_features.append(start_time_expanded)
            input_features.append(time_diff_expanded)
            x_input = torch.cat(input_features, dim=-1)  # Shape: [batch_size, num_nodes, input_dim]
            
            # Forward pass
            with torch.no_grad():    
                if self.model_config.use_conditional_norm:
                    latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                    xcoord = self.coord.to(self.device)
                    pred = self.model(
                        latent_tokens_coord = latent_tokens_coord,
                        xcoord = xcoord,
                        pndata = x_input[...,:-1],
                        condition = x_input[...,0,-2:-1]
                    )
                else:
                    pred = self.model(
                        latent_tokens_coord = self.latent_tokens_coord.to(self.device),
                        xcoord = self.coord.to(self.device),
                        pndata = x_input
                    )
                
                if self.dataset_config.stepper_mode == "output":
                    pred_de_norm = pred * u_std + u_mean
                    next_input = pred
                
                elif self.dataset_config.stepper_mode == "residual":
                    res_mean = torch.tensor(self.stats["res"]["mean"], dtype=self.dtype).to(self.device)
                    res_std = torch.tensor(self.stats["res"]["std"], dtype=self.dtype).to(self.device)
                    pred_de_norm = pred * res_std + res_mean

                    u_input_de_norm = current_u_in * u_std + u_mean

                    
                    pred_de_norm = u_input_de_norm + pred_de_norm
                    next_input = (pred_de_norm - u_mean)/u_std
                
                elif self.dataset_config.stepper_mode == "time_der":
                    der_mean = torch.tensor(self.stats["der"]["mean"], dtype=self.dtype).to(self.device)
                    der_std = torch.tensor(self.stats["der"]["std"], dtype=self.dtype).to(self.device)
                    pred_de_norm = pred * der_std + der_mean
                    u_input_de_norm = current_u_in * u_std + u_mean
                    # time difference
                    time_diff_tensor = torch.tensor(time_diff, dtype=self.dtype).to(self.device)
                    pred_de_norm = u_input_de_norm + time_diff_tensor * pred_de_norm
                    next_input = (pred_de_norm - u_mean)/u_std

            # Store prediction
            predictions.append(pred_de_norm)

            # Update current_u_in for next iteration
            current_u_in = next_input
        
        predictions = torch.stack(predictions, dim=1) # Shape: [batch_size, num_timesteps - 1, num_nodes, output_dim]
        
        return predictions
        
    def test(self):
        """
        Evaluate the trained model on the test dataset and optionally plot results.

        Returns:
            None
        """

        self.model.eval()
        self.model.to(self.device)
        if self.dataset_config.predict_mode == "all":
            modes = ["autoregressive", "direct", "star"]
        else:
            modes = [self.dataset_config.predict_mode]

        errors_dict = {}
        example_data = None # To store for plotting

        for mode in modes:
            all_relative_errors = []
            if mode == "autoregressive":
                time_indices = np.arange(0, 15, 2)  # [0, 2, 4, ..., 14]
            elif mode == "direct":
                time_indices = np.array([0, 14])
            elif mode == "star":
                time_indices = np.array([0, 4, 8, 12, 14])
            elif mode == "fracture":
                time_indices = np.arange(0, 10, 2)
            else:
                raise ValueError(f"Unknown predict_mode: {mode}")
    
            test_dataset = TestDataset(
                u_data = self.test_dataset.u_data,
                c_data = self.test_dataset.c_data,
                t_values = self.test_dataset.t_values,
                metadata = self.metadata,
                time_indices = time_indices,
                stats = self.stats
            ) # x is normalized, y is not normalized
            # TEST = True
            # if TEST:
            #     test_dataset = TestDataset(
            #         u_data = self.train_dataset.u_data,
            #         c_data = self.train_dataset.c_data,
            #         t_values = self.train_dataset.t_values,
            #         metadata = self.metadata,
            #         time_indices = time_indices,
            #         stats = self.stats
            #         )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.test_loader.batch_size,
                shuffle=False,
                num_workers=self.test_loader.num_workers,
                collate_fn=self._collate_fn
            )
            
            pbar = tqdm(total=len(test_loader), desc=f"Testing ({mode})", colour="blue")
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(test_loader):
                    # TODO: Figure out whether from CPU to GPU is the compuation bottleneck
                    # x_batch is normalized, y_batch is not normalized
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device) # Shape: [batch_size, num_nodes, num_channels] 
                    pred = self.autoregressive_predict(x_batch, time_indices) # Shape: [batch_size, num_timesteps - 1, num_nodes, num_channels]
                    
                    y_batch_de_norm = y_batch
                    pred_de_norm = pred

                    if self.dataset_config.metric == "final_step":
                        relative_errors = compute_batch_errors(
                            y_batch_de_norm[:,-1,:,:][:,None,:,:], 
                            pred_de_norm[:,-1,:,:][:,None,:,:], 
                            self.metadata)
                    elif self.dataset_config.metric == "all_step":
                        relative_errors = compute_batch_errors(
                            y_batch_de_norm, 
                            pred_de_norm, 
                            self.metadata)
                    else:
                        raise ValueError(f"Unknown metric: {self.dataset_config.metric}")
                    all_relative_errors.append(relative_errors)
                    pbar.update(1)
                    # Store example data for plotting (only once)
                    if example_data is None:
                        u_in_dim = self.stats["u"]["std"].shape[0]
                        if "c" in self.stats:
                            c_in_dim = self.stats["c"]["std"].shape[0]
                            x_batch_input_u = x_batch[...,:u_in_dim].cpu().numpy() * self.stats["u"]["std"] + self.stats["u"]["mean"]
                            x_batch_input_c = x_batch[...,u_in_dim:u_in_dim + c_in_dim].cpu().numpy() * self.stats["c"]["std"] + self.stats["c"]["mean"]
                            x_batch_input = np.stack([x_batch_input_u,x_batch_input_c], axis=-1)
                        else:
                            x_batch_input = x_batch[...,:u_in_dim].cpu().numpy() * self.stats["u"]["std"] + self.stats["u"]["mean"]
                        example_data = {
                            'input': x_batch_input[-1],
                            'coords': self.coord.cpu().numpy(),
                            'gt_sequence': y_batch_de_norm[-1].cpu().numpy(),
                            'pred_sequence': pred_de_norm[-1].cpu().numpy(),
                            'time_indices': time_indices,
                            't_values': self.test_dataset.t_values
                        }

                pbar.close()

            all_relative_errors = torch.cat(all_relative_errors, dim=0)
            final_metric = compute_final_metric(all_relative_errors)
            errors_dict[mode] = final_metric
        print(errors_dict)
        if self.dataset_config.predict_mode == "all":
            self.config.datarow["relative error (direct)"] = errors_dict["direct"]
            self.config.datarow["relative error (auto2)"] = errors_dict["autoregressive"]
            self.config.datarow["relative error (auto4)"] = errors_dict["star"]
        else:
            mode = self.dataset_config.predict_mode
            self.config.datarow[f"relative error ({mode})"] = errors_dict[mode]

        if example_data is not None:
            if self.metadata.names['c']:
                fig = plot_estimates(
                    u_inp = example_data['input'].squeeze(1),
                    u_gtr = np.stack([example_data['gt_sequence'][-1],example_data['input'][...,-1]],axis=-1).squeeze(1),
                    u_prd = np.stack([example_data['pred_sequence'][-1],example_data['input'][...,-1]],axis=-1).squeeze(1),
                    x_inp = example_data['coords'],
                    x_out = example_data['coords'],
                    names = self.metadata.names['u'] + self.metadata.names['c'],
                    symmetric = self.metadata.signed['u'] + self.metadata.signed['c'],
                    domain = self.metadata.domain_x
                )
            else:
                fig = plot_estimates(
                    u_inp = example_data['input'],
                    u_gtr = example_data['gt_sequence'][-1],
                    u_prd = example_data['pred_sequence'][-1],
                    x_inp = example_data['coords'],
                    x_out = example_data['coords'],
                    names = self.metadata.names['u'],
                    symmetric = self.metadata.signed['u'],
                    domain = self.metadata.domain_x
                )
            fig.savefig(self.path_config.result_path,dpi=300,bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        
        if self.setup_config.measure_inf_time:
            self.measure_inference_time()
