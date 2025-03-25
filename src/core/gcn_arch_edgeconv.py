
# Third-party libraries for the project

# PyTorch
import torch
from torchinfo import summary
# Set the default tensor type to float
torch.set_default_dtype(torch.float32)
from torch.utils.data import random_split
from torch.nn import Sequential, Linear, ELU, MSELoss, BatchNorm1d
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

class EdgeConvMLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) module for the GNN-EdgeConv."""
    def __init__(self, in_channels, out_channels, ec_mlp_width, ec_mlp_layer, activation_function=ELU()):
        """
        Initialize the MLP.

        Args:
            in_channels (int): Number of input channels = Node Features
            out_channels (int): Number of output channels = Embedded Space of the Node Features
            hidden_neurons (int): Number of hidden neurons = intermediate Embedded Space of the Node Features
            hidden_layers (int): Number of MLP layers, including the output layer
        """
        super().__init__()
        
        self.activation_function = activation_resolver(activation_function)
        
        layers = []
        mlp_layer_width = ec_mlp_width
        
        mlp_layers_dim_list = [mlp_layer_width] * int(ec_mlp_layer)

        # hidden_layer_1
        layers.append(Linear(in_channels * 2, mlp_layer_width))
        layers.append(BatchNorm1d(mlp_layer_width))  # Add BatchNorm layer after Linear, and before Activation
        layers.append(activation_function)
        # can improve by adding nn.Dropout(drop_prob=0.4), after the Activation

        # Define rest hidden_layers
        for layer_idx in range(len(mlp_layers_dim_list)-1):
            layers.append(Linear(mlp_layers_dim_list[layer_idx], mlp_layers_dim_list[layer_idx+1]))
            layers.append(BatchNorm1d(mlp_layers_dim_list[layer_idx+1]))  # Add BatchNorm layer
            layers.append(activation_function)

        # final output_layer without activation
        layers.append(Linear(mlp_layer_width, out_channels))

        self.mlp = Sequential(*layers)

    def forward(self, x):
        """
        Perform forward pass through the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # print(f'EC_MLP {summary(self.mlp)}')
        return self.mlp(x)

class LightningEdgeConvModel(pl.LightningModule):
    """Lightning module for the GNN using EdgeConvolution.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, num_edge_conv, ec_mlp_width, ec_mlp_layer, lr_initial=2e-3, gamma_decay=2e-3, use_optimizer="adam", lr=1e-2, weight_decay=0.0, activation_function=ELU(), skip_connection_type="concat"):
        
        """
        Initialize the LightningEdgeConvModel.

        Args:
            in_channels (int): Dimension of input node feature
            hidden_channels (int): Dimension of embedded node features
            out_channels (int): Dimension of node predictions

            num_edge_conv (int): Number of edge convolution layers

            activation_function: Non-linear activation function

            lr*: Learning rates
        """
        super(LightningEdgeConvModel, self).__init__()

        # Hyper-parameters
        # EdgeConv layers
        self.L = num_edge_conv # Number of EdgeConv
        self.hidden_neurons=hidden_channels # number of neurons in EdgeConvMLP
        self.ec_mlp_width = ec_mlp_width
        self.ec_mlp_layer = ec_mlp_layer
        self.in_channels = in_channels
        # Optimizer & Learning Rate
        self.use_optimizer = use_optimizer
        self.weight_decay = weight_decay
        self.LR = lr
        self.lr_initial = lr_initial
        self.gamma_decay = gamma_decay

        # Activation Function
        self.activation_function = activation_resolver(activation_function)

        # Skip-connection type
        self.skip_connection_type = skip_connection_type
        
        if self.skip_connection_type == "concat":
            self.decoder_layers= [int(self.L) * self.hidden_neurons] #DECODER_LAYER_SIZE_LIST #[1024, 512, 256] ... note this needs to be a list
            decoder_in_dim = int(self.L) * self.hidden_neurons

        # Define L convolutional layers sequentially
        self.convs = torch.nn.ModuleList()
        
        for i in range(self.L):
            ec_mlp_in_channels = self.in_channels if i == 0 else self.hidden_neurons
            self.convs.append(EdgeConv(nn=EdgeConvMLP(in_channels=ec_mlp_in_channels, out_channels=self.hidden_neurons, ec_mlp_width=self.ec_mlp_width, ec_mlp_layer=self.ec_mlp_layer, activation_function=self.activation_function), aggr="max"))
        
        # Initialize a list to hold the layers
        layers = []

        # Add the first layer
        layers.append(torch.nn.Linear(decoder_in_dim, self.decoder_layers[0]))
        layers.append(BatchNorm1d(self.decoder_layers[0]))
        layers.append(self.activation_function)

        # Add the final layer
        layers.append(torch.nn.Linear(self.decoder_layers[-1], out_channels))

        # Create the Sequential model
        self.decoder = torch.nn.Sequential(*layers)

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
        
        if self.skip_connection_type=="concat":
            # Define local embeddings
            local_features_list = []
            for conv in self.convs:
                x = conv(x, edge_index)
                local_features_list.append(x)
            
            # Concatenate EdgeConv's embedded spaces
            # local_features = torch.cat((h1, h2, h3), dim=1) 
            local_features_concat = torch.cat([x for _, x in enumerate(local_features_list)], dim=1) # Size: [N_v x (L*EDGE_CONV_MLP_SIZE = 320)]
            # print('\n')
            # print('local_features shape: ', local_features.shape)
        
            # Pass the concatenated features through the decoder
            # print(f"Total decoder layers: {(self.decoder)}")
            # print(f'PRESSURE_DECODER {summary(self.decoder)}')
            output = self.decoder(local_features_concat)

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
        out = self.forward(train_batch.x.float(), train_batch.edge_index)
        loss = self.loss(out.float(), train_batch.y)
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
        out = self.forward(val_batch.x.float(), val_batch.edge_index)
        loss = self.loss(out.float(), val_batch.y)
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
        out = self.forward(pred_batch.x.float(), pred_batch.edge_index)
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