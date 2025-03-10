"""
GNN Architecture for two-dimensional airfoil simulation

This module provides the Graph Neural Network (GNN) architecture used for the computational fluid dynamics simulation.

Third-party libraries for the project:

- PyTorch
- PyTorch-Geometric
- PyTorch-Lightning

User-defined libraries:

- lib.data_utils_rotated_airfoil: Utility functions for data loading and normalization

Hyper-parameters:

- BATCH_SIZE_PER_GPU
- EDGE_CONV_MLP_LAYERS
- EDGE_CONV_MLP_SIZE
- DECODER_LAYER_SIZE_LIST
- USE_DECAY_LR
- FIXED_LEARNING_RATE
- INITIAL_LEARNING_RATE
- DECAY_FACTOR

Class Definitions:

1. GNNDataModule(pl.LightningDataModule):
   Data module for Graph Neural Network.

2. EdgeConvMLP(torch.nn.Module):
   Multi-Layer Perceptron (MLP) module for the GNN-EdgeConv.

3. LightningEdgeConvModel(pl.LightningModule):
   Lightning module for the GNN using EdgeConvolution.

    - training_step(self, train_batch, batch_idx) -> Tensor: Training step.
    - validation_step(self, val_batch, batch_idx) -> Tensor: Validation step.
    - test_step(self, test_batch, batch_idx) -> Tensor: Test step.
    - predict_step(self, pred_batch, batch_idx) -> Tensor: Prediction step.

    - configure_optimizers() -> list: Configures the optimizer and learning rate scheduler.

Author: Sankalp Jena
Date: 29 July 2023
"""

# Third-party libraries for the project

# PyTorch
import torch
# Set the default tensor type to float
torch.set_default_dtype(torch.float32)
from torch.utils.data import random_split
from torch.nn import Sequential, Linear, ReLU, Tanh, MSELoss, BatchNorm1d

# PyTorch-Geometric
import torch_geometric
from torch_geometric.nn.models import GraphUNet
from torch_geometric.loader import DataLoader

# PyTorch-Lightning
import pytorch_lightning as pl

# User-defined libraries
from src import data_utils_rotated_airfoil

# Dataloader
num_cpus = 8 #2  # Number of available CPUs, on Taurus NUMA core CPUs, == '--cpus-per-task=8'

class GNNDataModule(pl.LightningDataModule):
    """Data module for Graph Neural Network."""
    def __init__(self, pyg_graph_dict, manual_seed, noise_std=0, batch_size_per_gpu=32):
        """
        Initialize the GNNDataModule.

        Args:
            pyg_graph_dict (dict): Dictionary containing PyTorch Geometric graph data.
            noise_std (float): Standard deviation of the Gaussian noise to be added.
        """
        super().__init__()
        self.pyg_graph_dict = pyg_graph_dict
        self.train_dataset_dict = None
        self.val_dataset_dict = None 
        self.test_dataset_dict = None
        self.manual_seed = int(manual_seed)
        self.noise_std = noise_std
        self.batch_size_per_gpu = batch_size_per_gpu

    def setup(self, stage):
        """
        Setup the dataset splits and perform normalization.

        Args:
            stage (str): Stage of training (e.g., 'fit', 'validate', 'test').
        """
        self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict = data_utils_rotated_airfoil.random_split_dataset(self.pyg_graph_dict, train_ratio=0.8, val_ratio=0.1, seed=self.manual_seed)
        
        norm_stats = data_utils_rotated_airfoil.get_minmax_normalization_stats(self.train_dataset_dict, save_path='./split_dataset_info' + '_seed_' + f'{self.manual_seed}')
        
        self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict = data_utils_rotated_airfoil.minmax_normalize_dataset(norm_stats, self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict, save_path='./split_dataset_info' + '_seed_' + f'{self.manual_seed}')

        if (stage == 'fit') and (self.noise_std!=0):
            # Add Gaussian noise to training data
            self.add_noise_to_dataset(self.train_dataset_dict)
    
    def add_noise_to_dataset(self, dataset_dict):
        """
        Add Gaussian noise to the dataset.

        Args:
            dataset_dict (dict): Dictionary containing the dataset to which noise is to be added.
        """
        for element in dataset_dict.values():
            # dict has [pyg_graph, Ub]
            # print("Before noise: ", element[0].y)
            element.y = element.y + torch.randn_like(element.y) * self.noise_std
            # print("After noise: ", element[0].y, '\n')
            # print('Shape: ', element[0].y.shape)
            # break
    
    def train_dataloader(self):
        """
        Return the data loader for the training dataset.

        Returns:
            DataLoader: Data loader for the training dataset.
        """
        if self.train_dataset_dict is None:
            raise RuntimeError("You must call `setup()` before accessing the dataloader.")
        train_dataset_list = [data for data in self.train_dataset_dict.values()]
        train_batch_size = self.batch_size_per_gpu
        train_loader = DataLoader(train_dataset_list, batch_size=train_batch_size, shuffle=True, persistent_workers=True, num_workers=num_cpus,pin_memory=True)
        print(f'Train set divided in batches of {train_batch_size}')
        return train_loader

    def val_dataloader(self):
        """
        Return the data loader for the validation dataset.

        Returns:
            DataLoader: Data loader for the validation dataset.
        """
        if self.val_dataset_dict is None:
            raise RuntimeError("You must call `setup()` before accessing the dataloader.")
        val_dataset_list = [data for data in self.val_dataset_dict.values()]
        batch_size = self.batch_size_per_gpu
        val_loader = DataLoader(val_dataset_list, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_cpus,pin_memory=True)
        return val_loader

    def test_dataloader(self):
        """
        Return the data loader for the test dataset.

        Returns:
            DataLoader: Data loader for the test dataset.
        """
        if self.test_dataset_dict is None:
            raise RuntimeError("You must call `setup()` before accessing the dataloader.")
        test_dataset_list = [data for data in self.test_dataset_dict.values()]
        batch_size = self.batch_size_per_gpu
        test_loader = DataLoader(test_dataset_list, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=num_cpus,pin_memory=True)
        return test_loader

class LightningGraphUNet(pl.LightningModule):
    """Lightning module for the GraphUNet using GCN."""
    def __init__(self, in_channels, out_channels, hidden_channels, depth, pool_ratios, lr_initial=2e-3, gamma_decay=2e-3, use_optimizer="adam", lr=1e-2, activation_function='elu'):
        """
        Initialize the LightningEdgeConvModel.

        Args:
            node_features (int): Number of node features.
            node_labels (int): Number of node labels.
        """
        super(LightningGraphUNet, self).__init__()

        # Model-parameters
        self.in_channels=in_channels
        self.out_channels=out_channels

        # Hyper-parameters
        self.hidden_channels=hidden_channels
        self.depth=depth
        self.pool_ratios=pool_ratios
        
        # Optimizer & Learning Rate
        self.use_optimizer = use_optimizer
        self.LR = lr
        self.lr_initial = lr_initial
        self.gamma_decay = gamma_decay

        # Activation Function
        self.activation_function = activation_function

        # Initialize the GraphUNet model
        self.model = GraphUNet(
            in_channels=self.in_channels,           # Input features per node (2)
            hidden_channels=self.hidden_channels,      # Hidden layer size (arbitrary, can be tuned)
            out_channels=self.out_channels,          # Output features per node (1)
            depth=self.depth,                 # Pooling and unpooling 3 times
            pool_ratios=self.pool_ratios,          # Keep 50% of nodes during pooling
            act=self.activation_function
        )

        self.loss = MSELoss(reduction='mean')

    def forward(self, x, edge_index):
        """
        Perform forward pass through the GNN.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Graph connectivity.

        Returns:
            Tensor: Predicted output.
        """
      
        # Pass the concatenated features through the decoder
        output = self.model(x, edge_index)
        
        return output
    
    def training_step(self, train_batch, batch_idx):
        """
        Perform a training step.

        Args:
            train_batch (Batch): Batch of training data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Training loss.
        """
        out = self.forward(train_batch.x, train_batch.edge_index)
        loss = self.loss(out, train_batch.y)
        # Log the training loss to TensorBoard
        self.log('train_loss', loss, batch_size=train_batch.x.shape[0], sync_dist=True, prog_bar=True, on_epoch=True, logger=True) #on_step=True
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Perform a validation step.

        Args:
            val_batch (Batch): Batch of validation data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Validation loss.
        """
        out = self.forward(val_batch.x, val_batch.edge_index)
        loss = self.loss(out, val_batch.y)
        self.log('val_loss', loss, batch_size=val_batch.x.shape[0], sync_dist=True, prog_bar=True, on_epoch=True, logger=True) #on_step=True
        # print(f"Test Loss: {loss.item():.6f}")
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Performs a test step in the training loop.

        Args:
            test_batch (torch.Tensor): Test batch containing input features.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the test step.
        """
        out = self.forward(test_batch.x, test_batch.edge_index)
        loss = self.loss(out, test_batch.y)
        self.log('test_loss', loss, batch_size=test_batch.x.shape[0], sync_dist=True, prog_bar=True)
        print(f"Test Loss: {loss.item():.8f}")
        return loss


    def predict_step(self, pred_batch, batch_idx):
        """
        Performs a prediction step in the training loop.

        Args:
            pred_batch (torch.Tensor): Batch containing input features for prediction.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        out = self.forward(pred_batch.x, pred_batch.edge_index)
        return out


    def configure_optimizers(self):
        """
        This function is used to configure the optimizer for the model's parameters. It uses the Adam 
        optimizer, a method for efficient stochastic optimization that is well suited for problems that 
        are large in terms of data and/or parameters.
        
        The function can operate in two modes depending on the value of `USE_DECAY_LR`. If `USE_DECAY_LR` 
        is set to True, a learning rate scheduler is used that decays the learning rate over time according 
        to the formula provided in the lambda function. This is typically useful for problems where the 
        loss landscape changes over time or where we want the optimizer to make large updates in the early 
        stages of training and smaller updates later on.
        
        If `USE_DECAY_LR` is set to False, a fixed learning rate specified by `FIXED_LEARNING_RATE` is used 
        for the Adam optimizer. This is a simpler approach and can be effective for many problems, 
        particularly if the learning rate is tuned carefully.
        
        The function returns the optimizer, and the learning rate scheduler if `USE_DECAY_LR` is set to True.
        
        Returns:
            list: List containing the optimizer and potentially a learning rate scheduler.
        """
        if self.use_optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        
        if self.use_optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.LR, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        
        if self.use_optimizer == "adam_decay":
            # From Chen et. al. (2021)
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_initial)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.lr_initial / (1 + self.gamma_decay * epoch))
            return [optimizer], [lr_scheduler]
            
        return optimizer