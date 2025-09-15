import os
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

from .optimizers import AdamOptimizer, AdamWOptimizer
from .utils.train_setup import manual_seed, load_ckpt, save_ckpt
from .utils.default_set import SetUpConfig, ModelConfig, DatasetConfig, OptimizerConfig, PathConfig, merge_config
from src.data.dataset import DATASET_METADATA
from .utils.plot import plot_learning_rate

class TrainerBase:
    """
    Base class for all trainers, define the init_dataset, initi_model, 
    init_optimizer, train_step, validate, test for coreresponding trainers.

    Attributes:
    ----------
    """
    def __init__(self, args):
        # --- Config setup ---
        self.config = args
        self.setup_config = merge_config(SetUpConfig, self.config.setup)
        self.model_config = merge_config(ModelConfig, self.config.model)
        self.dataset_config = merge_config(DatasetConfig, self.config.dataset)
        self.optimizer_config = merge_config(OptimizerConfig, self.config.optimizer)
        self.path_config = merge_config(PathConfig, self.config.path)
        ## Dataset Setup
        self.metadata = DATASET_METADATA[self.dataset_config.metaname]

        ## initialization the distributed learning environment
        if self.setup_config.distributed:
            self.init_distributed_mode()
            torch.cuda.set_device(self.setup_config.local_rank)
            self.device = torch.device('cuda', self.setup_config.local_rank)
        else:
            self.device = torch.device(self.setup_config.device)
        
        manual_seed(self.setup_config.seed + self.setup_config.rank)

        if self.setup_config.dtype in ["float", "torch.float32", "torch.FloatTensor"]:
            self.dtype = torch.float32
        elif self.setup_config.dtype in ["double", "torch.float64", "torch.DoubleTensor"]:
            self.dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype: {self.setup_config.dtype}")
        self.loss_fn = nn.MSELoss()
        
        self.init_dataset(self.dataset_config)
        self.init_model(self.model_config)
        self.init_optimizer(self.optimizer_config)

        if self.setup_config.rank == 0:
            nparam = sum(
                [p.numel() * 2 if p.is_complex() else p.numel() for p in self.model.parameters()]
            )
            nbytes = sum(
                [p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size() for p in self.model.parameters()]
            )
            print(f"Number of parameters: {nparam}")
            args.datarow['nparams'] = nparam
            args.datarow['nbytes'] = nbytes
            
# ------------ init ------------ #
    def init_dataset(self, dataset_config):
        raise NotImplementedError
    
    def init_model(self, model_config):
        raise NotImplementedError

    def init_optimizer(self, optimizer_config):
        """Initialize the optimizer"""
        self.optimizer = {
            "adam": AdamOptimizer,
            "adamw": AdamWOptimizer
        }[self.optimizer_config.name](self.model.parameters(), self.optimizer_config.args)

    def init_distributed_mode(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.setup_config.rank = int(os.environ['RANK'])
            self.setup_config.world_size = int(os.environ['WORLD_SIZE'])
            self.setup_config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            print('Not using distributed mode')
            self.setup_config.distributed = False
            self.setup_config.rank = 0
            self.setup_config.world_size = 1
            self.setup_config.local_rank = 0
            return

        dist.init_process_group(
            backend=self.setup_config.backend,
            init_method='env://',
            world_size=self.setup_config.world_size,
            rank=self.setup_config.rank
        )
        dist.barrier()

# ------------ utils ------------ #
    def to(self, device):
        self.model.to(device)
    
    def type(self, dtype):
        # TODO: check if this is necessary
        self.model.type(dtype)

    def load_ckpt(self):
        load_ckpt(self.path_config.ckpt_path, model = self.model)
        if self.model_config.use_ema:
            load_ckpt(self.path_config.ckpt_path_ema, model = self.ema_model)
        return self
    
    def save_ckpt(self):
        """Save checkpoint to the config.ckpt_path"""
        os.makedirs(os.path.dirname(self.path_config.ckpt_path), exist_ok=True)
        save_ckpt(self.path_config.ckpt_path, model = self.model)
        if self.model_config.use_ema:
            ema_path = self.path_config.ckpt_path.replace(".pt", "_ema.pt")
            save_ckpt(ema_path, model=self.ema_model)

        return self

    def compute_test_errors(self):
        # compute test errors (need to modulate based on the type of dataset, based on metadata)
        raise NotImplementedError

# ------------ train ------------ #
    def train_step(self, batch):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        x_batch, y_batch = x_batch.squeeze(1), y_batch.squeeze(1)
        pred = self.model(self.rigraph, x_batch)
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch, y_batch = x_batch.squeeze(1), y_batch.squeeze(1)
                pred = self.model(self.rigraph, x_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, verbose=False):
        self.to(self.device)
        #self.type(self.dtype)

        result = self.optimizer.optimize(self)
        self.config.datarow['training time'] = result['time']
        
        self.save_ckpt()

        if len(result['train']['loss'])==0:
            if self.setup_config.use_variance_test:
                self.variance_test()
            else:
                self.test()
        else:
            kwargs = {
                "epochs":result['train']['epoch'],
                "losses":result['train']['loss'],
                "ema_losses":result['train']['ema_losses']
            }
        
            if "valid" in result:
                kwargs['val_epochs'] = result['valid']['epoch']
                kwargs['val_losses']= result['valid']['loss']
            
            if "best" in result:
                kwargs['best_epoch'] = result['best']['epoch']
                kwargs['best_loss']  = result['best']['loss']
            
            self.plot_losses(
                **kwargs
            )

            kwargs["losses"] = result["train"]['ema_losses']
            self.plot_losses(**kwargs)

            plot_learning_rate(self.optimizer.lr_list, ".results/time_indep/naca0012_lr.png")

            if self.setup_config.use_variance_test:
                self.variance_test()
            else:
                self.test()

# ------------ plot ------------ #
    def plot_losses(self, 
                    epochs,
                    losses, 
                    ema_losses = None,
                    val_epochs = None,
                    val_losses = None,
                    best_epoch = None,
                    best_loss  = None):

        if val_losses is None:
            # plot only train loss
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, losses)
            ax.scatter([best_epoch],[best_loss], c='r', marker='o', label="best loss")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss vs Epoch')
            ax.legend()
            ax.set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax.set_yscale('log')
            np.savez(self.path_config["loss_path"][:-4]+".npz", epochs=epochs, losses=losses)

        else:
            if ema_losses is not None:
                # also plot valid loss
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            else:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            ax[0].plot(epochs, losses)
            ax[0].scatter([best_epoch],[best_loss], c='r', marker='o', label="best loss")
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].set_title('Loss vs Epoch')
            ax[0].legend()
            ax[0].set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax[0].set_yscale('log')

            ax[1].plot(val_epochs, val_losses)
            # if best_epoch is not None and best_loss is not None:
            #     ax[1].scatter([best_epoch],[best_loss], c='r', marker='o', label="best validation loss")
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('relative error')
            ax[1].set_title('Loss vs relative error')
            ax[1].legend()
            ax[1].set_xlim(left=0)
            if (np.array(val_losses) > 0).all():
                ax[1].set_yscale('log')

            if ema_losses is not None:
                ax[2].plot(epochs, ema_losses)
                ax[2].set_xlabel('Epoch')
                ax[2].set_ylabel('EMA_Loss')
                ax[2].set_title('EMA_Loss vs Epoch')
                ax[2].legend()
                ax[2].set_xlim(left=0)
                if (np.array(ema_losses) > 0).all():
                    ax[2].set_yscale('log')

            plt.savefig(self.path_config.loss_path)
            np.savez(self.path_config.loss_path[:-4]+".npz", epochs=epochs, losses=losses, val_epochs=val_epochs, val_losses=val_losses)
                
    def plot_results(self):
        raise NotImplementedError

# ------------ test ------------ #
    def variance_test(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
        
