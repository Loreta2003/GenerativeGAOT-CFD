from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

# model default config
from src.model.layers.attn import TransformerConfig
from src.model.layers.magno2d_vx import MAGNOConfig
from ..optimizers import OptimizerargsConfig

from omegaconf import OmegaConf

def merge_config(default_config_class, user_config):
    default_config_struct = OmegaConf.structured(default_config_class)
    merged_config = OmegaConf.merge(default_config_struct, user_config)
    return OmegaConf.to_object(merged_config)

@dataclass
class SetUpConfig:
    seed: int = 42                                              # Random seed for reproducibility
    device: str = "cuda:0"                                      # Computation device, (e.g., "cuda:0", "cpu")          
    dtype: str = "torch.float32"                                # Data type for computation (e.g., "torch.float32", "torch.float64")        
    trainer_name: str = "sequential"                            # Type of trainer to use, support [static_fx, static_vx, sequential_fx]
    train: bool = True                                          # Whether to run the training phase       
    test: bool = False                                          # Whether to run the testing phase
    ckpt: bool = False                                          # Whether to load a checkpoint for training or testing            
    use_variance_test: bool = False  
    separate_training: bool = False                           # TODO Whether to use variance testing  
    measure_inf_time: bool = False                              # TODO Whether to measure inference time, support in a seperate repo now.
    visualize_encoder_output: bool = False                      # Whether to visualize the output of the encoder/processor output            
    vis_component: str = "encoder"                              # Component to visualize, support ['encoder', 'processor']
    # Parameters for distributed mode
    distributed: bool = False                                   # Whether to enable distributed training           
    world_size: int = 1                                         # Total number of processes in distributed training
    rank: int = 0                                               # Rank of the current process in distributed training
    local_rank: int = 0                                         # Local rank of the current process in distributed training
    backend: str = "nccl"                                       # Backend for distributed training, e.g., 'nccl' (NVIDIA Collective Communications Library)

@dataclass
class ModelArgsConfig:
    magno: MAGNOConfig = field(default_factory=MAGNOConfig)                     # Configuration for the MAGNO (encoder and decoder) module
    transformer: TransformerConfig = field(default_factory=TransformerConfig)   # Configuration for the Transformer (processor) module

@dataclass
class ModelConfig:
    name: str = "goat2d_vx"                                         # Name of the model, support ['goat2d_vx', 'goat2d_fx'], 3D models are supported in a separate repo.     
    use_conditional_norm: bool = False                              # Whether to use time-conditional normalization (not supported in this repo)
    use_encoder_decoder: bool = True                                # Whether to use encoder/decoder (only applies to ablation model)
    use_ema: bool = True                                            # Whether to use Exponential Moving Average (EMA) for model parameters
    latent_tokens_size: Tuple[int, int] = (64, 64)                  # Size (H, W) of latent tokens
    args: ModelArgsConfig = field(default_factory=ModelArgsConfig)  # Configuration for the model's components

@dataclass
class DatasetConfig:
    name: str = "CE-Gauss"                                                  # Name of the dataset, e.g., "CE-Gauss"
    metaname: str = "rigno-unstructured/CE-Gauss"                           # Metadata name (identifier) for the dataset, used for loading the dataset
    base_path: str = "/cluster/work/math/camlab-data/rigno-unstructured/"   # Base path where the dataset is stored
    train_size: int = 1024                                                  # Number of samples in the training set
    val_size: int = 128                                                     # Number of samples in the validation set
    test_size: int = 256                                                    # Number of samples in the test set
    coord_scaling: str = "per_dim_scaling"                                  # Coordinate scaling strategy, supports ['global_scaling', 'per_dim_scaling']
    batch_size: int = 64                                                    # Batch size for training and evaluation
    num_workers: int = 4                                                    # Number of worker threads for data loading
    shuffle: bool = True                                                    # Whether to shuffle the dataset during training
    use_metadata_stats: bool = False                                        # Whether to use metadata statistics for normalization. 
    sample_rate: float = 0.1                                                # Sample rate for the dataset (for point clouds or subsampling)
    use_sparse: bool = False                                                # Whether to use sparse representations for the dataset, only for the PDEGym
    rand_dataset: bool = False                                              # Whether to randomize the sequence of loaded dataset
    # Designed for Time-dependent dataset
    max_time_diff: int = 14                                                 # Max time difference for creating data pairs
    use_time_norm: bool = True                                              # whether to use normalization for lead time and time_difference
    metric: str = "final_step"                                              # Metric for evaluation, supports ['final_step', 'all_step']
    predict_mode: str = "all"                                               # Inference mode, supports ['all', 'autoregressive', 'direct', 'star'], only for time-dependent dataset
    stepper_mode: str = "output"                                            # [output, residual, time_der]

@dataclass
class OptimizerConfig:
    name: str = "adamw"                                                     # Name of the optimizer, support ['adamw', 'adam']
    args: OptimizerargsConfig = field(default_factory=OptimizerargsConfig)  # Detailed arguments for the optimizer

@dataclass
class PathConfig:
    ckpt_path: str = ".ckpt/test/test.pt"                   # Path to save or load the model checkpoint
    loss_path: str = ".loss/test/test.png"                  # Path to save loss curve plots
    result_path: str = ".result/test/test.png"              # Path to save result visualization plots
    database_path: str = ".database/test/test.csv"          # Path to save a CSV database of experiment results and parameters (train setup, model config, etc.)
    ckpt_path_ema: str = ".ckpt/test/test_ema.pt"            # Path to save or load the EMA model checkpoint, only for models with use_ema=True


