import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
import numpy as np

from .layers.attn import Transformer
from .layers.magno2d_fx import MAGNOEncoder, MAGNODecoder


class GOAT2D_FX_Ablation(nn.Module):
    """
    GOAT2D_FX with ablation study support:
    - Can run full pipeline: MAGNO Encoder + ViT + MAGNO Decoder
    - Can run ViT-only for regular grids (ablation study)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: Optional[dataclass] = None,
                 use_encoder_decoder: bool = True):
        nn.Module.__init__(self)
        # --- Define model parameters ---
        self.input_size = input_size
        self.output_size = output_size
        self.use_encoder_decoder = use_encoder_decoder
        self.node_latent_size = config.args.magno.lifting_channels 
        self.patch_size = config.args.transformer.patch_size
        latent_tokens_size = config.latent_tokens_size
        self.H = latent_tokens_size[0]
        self.W = latent_tokens_size[1]
        self.grid_indices = None

        # Initialize components based on configuration
        if self.use_encoder_decoder:
            ## Initialize encoder, processor, and decoder (full GAOT)
            self.encoder = self.init_encoder(input_size, self.node_latent_size, config.args.magno)
            self.decoder = self.init_decoder(output_size, self.node_latent_size, config.args.magno)
        else:
            ## For ablation: direct input/output projection layers
            self.input_projection = nn.Linear(input_size, self.node_latent_size)
            self.output_projection = nn.Linear(self.node_latent_size, output_size)
        
        self.processor = self.init_processor(self.node_latent_size, config.args.transformer)
    
    def init_encoder(self, input_size, latent_size, gno_config):
        return MAGNOEncoder(
            in_channels = input_size,
            out_channels = latent_size,
            gno_config = gno_config
        )
    
    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor
        self.patch_linear = nn.Linear(self.patch_size * self.patch_size * node_latent_size,
                                      self.patch_size * self.patch_size * node_latent_size)
    
        self.positional_embedding_name = config.positional_embedding
        self.positions = self._get_patch_positions()

        setattr(config.attn_config, 'H', self.H)
        setattr(config.attn_config, 'W', self.W)

        return Transformer(
            input_size=self.node_latent_size * self.patch_size * self.patch_size,
            output_size=self.node_latent_size * self.patch_size * self.patch_size,
            config=config
        )

    def init_decoder(self, output_size, latent_size, gno_config):
        # Initialize the GNO decoder
        return MAGNODecoder(
            in_channels=latent_size,
            out_channels=output_size,
            gno_config=gno_config
        )

    def _get_patch_positions(self):
        """
        Generate positional embeddings for the patches.
        """
        num_patches_H = self.H // self.patch_size
        num_patches_W = self.W // self.patch_size
        positions = torch.stack(torch.meshgrid(
            torch.arange(num_patches_H, dtype=torch.float32),
            torch.arange(num_patches_W, dtype=torch.float32),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)

        return positions

    def _compute_absolute_embeddings(self, positions, embed_dim):
        """
        Compute absolute embeddings for the given positions.
        """
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(positions.size(0), -1)
        return pos_emb

    def _sort_coordinates_to_grid(self, xcoord, pndata):
        """
        Sort irregular coordinates and data to regular grid format.
        
        Parameters:
        -----------
        xcoord: torch.Tensor [N, 2] - coordinates (x, y)
        pndata: torch.Tensor [batch_size, N, input_size] - physical node data
        
        Returns:
        --------
        sorted_data: torch.Tensor [batch_size, H, W, input_size] - grid-sorted data
        """
        device = xcoord.device
        batch_size = pndata.shape[0]
        N = xcoord.shape[0]
        input_size = pndata.shape[2]
        
        # Extract x and y coordinates
        x_coords = xcoord[:, 0].cpu().numpy()
        y_coords = xcoord[:, 1].cpu().numpy()

        if self.grid_indices is None:
            # Find unique coordinates and create grid
            unique_x = np.unique(x_coords)
            unique_y = np.unique(y_coords)
            
            # Verify this is indeed a regular grid
            assert len(unique_x) * len(unique_y) == N, \
                f"Not a regular grid: {len(unique_x)} * {len(unique_y)} != {N}"
            assert len(unique_x) == self.W and len(unique_y) == self.H, \
                f"Grid size mismatch: expected ({self.H}, {self.W}), got ({len(unique_y)}, {len(unique_x)})"
            
            # Create sorting indices
            # Find the index mapping from (x,y) coordinates to grid positions
            grid_indices = torch.zeros(N, dtype=torch.long, device=device)
            
            for i in range(N):
                x_idx = np.searchsorted(unique_x, x_coords[i])
                y_idx = np.searchsorted(unique_y, y_coords[i])
                grid_idx = y_idx * self.W + x_idx
                grid_indices[i] = grid_idx
            self.grid_indices = grid_indices
        else:
            grid_indices = self.grid_indices
    
        # Sort data according to grid indices
        sorted_pndata = torch.zeros(batch_size, self.H * self.W, input_size, device=device)
        sorted_pndata[:, grid_indices, :] = pndata
        
        # Reshape to grid format
        sorted_data = sorted_pndata.view(batch_size, self.H, self.W, input_size)
        
        return sorted_data, grid_indices

    def _unsort_from_grid(self, grid_data, grid_indices):
        """
        Convert grid format back to original coordinate order.
        
        Parameters:
        -----------
        grid_data: torch.Tensor [batch_size, H, W, output_size] - grid format data
        grid_indices: torch.Tensor [N] - sorting indices from _sort_coordinates_to_grid
        
        Returns:
        --------
        unsorted_data: torch.Tensor [batch_size, N, output_size] - original order data
        """
        batch_size = grid_data.shape[0]
        output_size = grid_data.shape[3]
        N = len(grid_indices)
        
        # Flatten grid data
        flat_data = grid_data.view(batch_size, self.H * self.W, output_size)
        
        # Unsort according to original coordinate order
        unsorted_data = flat_data[:, grid_indices, :]
        
        return unsorted_data

    def encode(self, x_coord: torch.Tensor, 
               pndata: torch.Tensor, 
               latent_tokens_coord: torch.Tensor, 
               encoder_nbrs: list) -> torch.Tensor:
        
        if self.use_encoder_decoder:
            encoded = self.encoder(
                x_coord = x_coord, 
                pndata = pndata,
                latent_tokens_coord = latent_tokens_coord,
                encoder_nbrs = encoder_nbrs)
        else:
            # For ablation: sort to grid and project
            grid_data, _ = self._sort_coordinates_to_grid(x_coord, pndata)
            batch_size = grid_data.shape[0]
            # Apply input projection
            grid_data = grid_data.view(batch_size, self.H * self.W, -1)
            encoded = self.input_projection(grid_data)
        
        return encoded

    def process(self, rndata: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        rndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        H, W = self.H, self.W
        
        # --- Check the input shape ---
        assert n_regional_nodes == H * W, \
            f"n_regional_nodes ({n_regional_nodes}) is not equal to H ({H}) * W ({W})"
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"H({H}) and W({W}) must be divisible by P({P})"

        # --- Reshape the input data ---
        num_patches_H = H // P
        num_patches_W = W // P
        num_patches = num_patches_H * num_patches_W
        ##  Reshape to patches
        rndata = rndata.view(batch_size, H, W, C)
        rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, num_patches_H * num_patches_W, P * P * C)
        
        ## --- Apply Vision Transformer processor ---
        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)  # shape [num_patches, 2]

        if self.positional_embedding_name == 'absolute':
            pos_emb = self._compute_absolute_embeddings(pos, self.patch_size * self.patch_size * self.node_latent_size)
            rndata = rndata + pos_emb
            relative_positions = None
    
        elif self.positional_embedding_name == 'rope':
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        ## --- Reshape back to the original shape ---
        rndata = rndata.view(batch_size, num_patches_H, num_patches_W, P, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, H * W, C)

        return rndata

    def decode(self, latent_tokens_coord: torch.Tensor, 
               rndata: torch.Tensor, 
               query_coord: torch.Tensor, 
               decoder_nbrs: list) -> torch.Tensor:
        
        if self.use_encoder_decoder:
            # Apply MAGNO decoder
            decoded = self.decoder(
                latent_tokens_coord = latent_tokens_coord,
                rndata = rndata, 
                query_coord = query_coord,
                decoder_nbrs = decoder_nbrs)
        else:
            # For ablation: project and unsort from grid
            batch_size = rndata.shape[0]
            
            # Apply output projection
            projected = self.output_projection(rndata)
            # Reshape to grid format and unsort
            grid_data = projected.view(batch_size, self.H, self.W, -1)
            decoded = self._unsort_from_grid(grid_data, self.grid_indices)
        
        return decoded

    def forward(self,
                latent_tokens_coord: torch.Tensor,
                xcoord: torch.Tensor,
                pndata: torch.Tensor,
                query_coord: Optional[torch.Tensor] = None,
                encoder_nbrs: Optional[list] = None,
                decoder_nbrs: Optional[list] = None,
                condition: Optional[float] = None,
                ) -> torch.Tensor:
        """
        Forward pass with ablation study support.

        Parameters
        ----------
        latent_tokens_coord: torch.Tensor
            ND Tensor of shape [n_regional_nodes, n_dim]
        xcoord: Optional[torch.Tensor]
            ND Tensor of shape [n_physical_nodes, n_dim]
        pndata: Optional[torch.Tensor]
            ND Tensor of shape [batch_size, n_physical_nodes, input_size]
        query_coord: Optional[torch.Tensor]
            ND Tensor of shape [n_physical_nodes, n_dim]
        condition: Optional[float]
            The condition of the model
        encoder_nbrs: Optional[list]
            List of neighbors for the encoder (only used if use_encoder_decoder=True)
        decoder_nbrs: Optional[list]
            List of neighbors for the decoder (only used if use_encoder_decoder=True)

        Returns
        -------
        torch.Tensor
            The output tensor of shape [batch_size, n_physical_nodes, output_size]
        """
        # Encode: Map physical nodes to regional nodes using GNO Encoder (or direct projection)
        rndata = self.encode(
            x_coord = xcoord, 
            pndata = pndata,
            latent_tokens_coord = latent_tokens_coord,
            encoder_nbrs = encoder_nbrs)

        # Process: Apply Vision Transformer on the regional nodes
        rndata = self.process(
            rndata = rndata, 
            condition = condition)

        # Decode: Map regional nodes back to physical nodes using GNO Decoder (or direct projection)
        if query_coord is None:
            query_coord = xcoord
        output = self.decode(
            latent_tokens_coord = latent_tokens_coord,
            rndata = rndata, 
            query_coord = query_coord,
            decoder_nbrs = decoder_nbrs)

        return output 
