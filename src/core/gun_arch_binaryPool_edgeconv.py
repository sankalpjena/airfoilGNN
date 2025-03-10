# Third-party libraries for the project

# PyTorch
import torch
from torchinfo import summary

# Set the default tensor type to float
torch.set_default_dtype(torch.float32)
from torch.utils.data import random_split
from torch.nn import Sequential, Linear, ReLU, Tanh, MSELoss, BatchNorm1d
from torch_geometric.nn.resolver import activation_resolver

# PyTorch-Geometric
import torch_geometric
from torch_geometric.nn import EdgeConv
from torch_geometric.loader import DataLoader

# PyTorch-Lightning
import lightning as pl

# User-defined libraries
import sys
root_path = '../../'
sys.path.append(root_path)
from src.core import data_utils_rotated_airfoil
from src.core.graph_unet_edgeconv import GraphUNetEdgeConv

# Dataloader
num_cpus = 2 #2  # Number of available CPUs, on Taurus NUMA core CPUs, == '--cpus-per-task=8'

class GNNDataModule(pl.LightningDataModule):
    """Data module for Graph Neural Network."""
    def __init__(self, pyg_graph_dict_train, pyg_graph_dict_test, manual_seed, batch_size_per_gpu=32):
        """
        Initialize the GNNDataModule.

        Args:
            pyg_graph_dict (dict): Dictionary containing PyTorch Geometric graph data.
            noise_std (float): Standard deviation of the Gaussian noise to be added.
        """
        super().__init__()
        self.pyg_graph_dict_train = pyg_graph_dict_train
        self.pyg_graph_dict_test = pyg_graph_dict_test
        self.train_dataset_dict = None
        self.val_dataset_dict = None 
        self.test_dataset_dict = pyg_graph_dict_test
        self.manual_seed = int(manual_seed)
        self.batch_size_per_gpu = batch_size_per_gpu

    def setup(self, stage):
        """
        Setup the dataset splits and perform normalization.

        Args:
            stage (str): Stage of training (e.g., 'fit', 'validate', 'test').
        """
        self.train_dataset_dict, self.val_dataset_dict = data_utils_rotated_airfoil.random_split_dataset(self.pyg_graph_dict_train, self.pyg_graph_dict_test, train_ratio=0.9, seed=self.manual_seed)
        
        norm_stats = data_utils_rotated_airfoil.get_minmax_normalization_stats(self.train_dataset_dict, save_path='./split_dataset_info' + '_seed_' + f'{self.manual_seed}')
        
        self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict = data_utils_rotated_airfoil.minmax_normalize_dataset(norm_stats, self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict, save_path='./split_dataset_info' + '_seed_' + f'{self.manual_seed}')
    
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


class LightningGraphUNetEdgeConv(pl.LightningModule):
    """Lightning module for the GraphUNet using GCN."""
    def __init__(self, in_channels, out_channels, hidden_channels, depth, ec_mlp_width, ec_mlp_layer, v_cycles, lr_initial=2e-3, gamma_decay=2e-3, use_optimizer="adam", lr=1e-2, weight_decay=0.0, activation_function='elu'):
        """
        Initialize the LightningEdgeConvModel.

        Args:
            node_features (int): Number of node features.
            node_labels (int): Number of node labels.
        """
        super(LightningGraphUNetEdgeConv, self).__init__()

        # Model-parameters
        self.in_channels=in_channels
        self.out_channels=out_channels

        # Hyper-parameters
        self.hidden_channels=hidden_channels # node embeddings
        self.depth=depth
        self.ec_mlp_width=ec_mlp_width
        self.ec_mlp_layer=ec_mlp_layer
        
        # Optimizer & Learning Rate
        self.use_optimizer = use_optimizer
        self.LR = lr
        self.weight_decay = weight_decay
        self.lr_initial = lr_initial
        self.gamma_decay = gamma_decay

        # Activation Function
        self.activation_function = activation_resolver(activation_function)

        # Initialize the GraphUNet model
        # Init v_cycles
        self.v_cycle_list = torch.nn.ModuleList()
        self.v_cycles = v_cycles

        # first v_cycle has input_channels = self.hidden_channels
        self.v_cycle_list.append(GraphUNetEdgeConv(
            in_channels=self.in_channels,                       # Input features per node (self.hidden_channels)
            hidden_channels=self.hidden_channels,               # Hidden layer size of EdgeConv (arbitrary, can be tuned)
            ec_mlp_width=self.ec_mlp_width,
            ec_mlp_layer=self.ec_mlp_layer,
            out_channels=self.out_channels,                     # Output features per node (1)
            depth=self.depth,                                   # Pooling and unpooling
            act=self.activation_function
        ))

        for idx in range(self.v_cycles-1):
            self.v_cycle_list.append(GraphUNetEdgeConv(
            in_channels=self.hidden_channels, # Input features per node (8) in the 2nd cycle
            hidden_channels=self.hidden_channels,               # Hidden layer size of EdgeConv (arbitrary, can be tuned)
            ec_mlp_width=self.ec_mlp_width,
            ec_mlp_layer=self.ec_mlp_layer,
            out_channels=self.out_channels,                     # Output features per node (1)
            depth=self.depth,                                   # Pooling and unpooling
            act=self.activation_function
        ))

        # MLP Decoder for the final output graph:
        # # Initialize a list to hold the layers
        # layers = []

        # # Define the input channel of mlp_output
        # # if output from top-level upool
        # mlp_decoder_dim = self.hidden_channels
        
        # # Concatenating input [2], first conv (encode) [8], top-most (unpool) [8]
        # # mlp_decoder_dim = self.in_channels + ( (self.v_cycles*2) * self.hidden_channels)

        # # Add the first layer
        # layers.append(torch.nn.Linear(mlp_decoder_dim, mlp_decoder_dim))
        # layers.append(BatchNorm1d(mlp_decoder_dim))  # Add BatchNorm layer
        # layers.append(self.activation_function)

        # # Add intermediate layers
        # for i in range(self.mlp_decoder_layers - 1):
        #     layers.append(torch.nn.Linear(mlp_decoder_dim, mlp_decoder_dim))
        #     layers.append(BatchNorm1d(mlp_decoder_dim))  # Add BatchNorm layer
        #     layers.append(self.activation_function)

        # # Add the final layer
        # layers.append(torch.nn.Linear(mlp_decoder_dim, self.out_channels))
        # layers.append(BatchNorm1d(self.out_channels))

        # decoder_layers list intialization
        decoder_layers = []

        # Define the input channel of mlp_encoder
        mlp_decoder_dim_list = [self.hidden_channels,self.hidden_channels,self.out_channels]

        # Create the decoder layers
        for dec_layer_idx in range(len(mlp_decoder_dim_list)-1):
            decoder_layers.append(torch.nn.Linear(mlp_decoder_dim_list[dec_layer_idx], mlp_decoder_dim_list[dec_layer_idx+1]))
            # Don't add activation, batchnorm to final output layer
            if dec_layer_idx !=1:
                decoder_layers.append(BatchNorm1d(mlp_decoder_dim_list[dec_layer_idx+1]))
                decoder_layers.append(self.activation_function)
        
        # Create the sequential model
        self.mlp_decoder = torch.nn.Sequential(*decoder_layers)

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
        # Pass through 1st v_cycle
        output = self.v_cycle_list[0](x, edge_index)
        # print(f"V-cycle: 1")
        # print(f"Output dim: {output.shape}")
        
        for v in range(self.v_cycles-1):
            # print(f"V-cycle: {v+2}")
            output = self.v_cycle_list[v+1](output, edge_index)
            # print(f"Output dim: {output.shape}")
        
        # pass the feature matrix from latest v-cycle through mlp_decoder
        output = self.mlp_decoder(output)
        # print(f'MLP_DECODER: {summary(self.mlp_decoder)}')
        # print(f'MLP_DECODER Output dim: {output.shape}')

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
        
        # # Log gradients (only for TensorBoardLogger)
        # if isinstance(self.log, TensorBoardLogger):
        #     for name, param in self.named_parameters():
        #         if param.requires_grad and param.grad is not None:
        #             self.logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.global_step)

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
        print(f'Output Dim: {out.shape}')
        print(f'Label Dim: {test_batch.y.shape}')
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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.LR, weight_decay=self.weight_decay)
        
        if self.use_optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.LR, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        
        if self.use_optimizer == "adam_decay":
            # From Chen et. al. (2021)
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_initial)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.lr_initial / (1 + self.gamma_decay * epoch))
            return [optimizer], [lr_scheduler]
            
        return optimizer