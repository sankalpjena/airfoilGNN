from typing import Callable, Union

import torch
from torch import Tensor
from torchinfo import summary

from torch_geometric.data import Data

from torch_geometric.nn import EdgeConv
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor

# Pooling
import math
from torch_geometric.nn.pool import avg_pool

# EdgeConv
from torch.nn import Sequential, Linear, ELU, BatchNorm1d

# Functions
def binary_fusion_cluster_club_last(x):
    """
    PyG pooling requires clustering.
    If there are 6 nodes in a cycle graph, and we want graph with half nodes, the cluster is
        cluster = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    
    If there are odd nodes, then the last node will be clubbed with the previous cluster.
    Eg. N=3, cluster = [0,0,0] instead of [0,0,1]
    """
    
    # Calculate the length of the tensor
    length = math.ceil((x.shape[0]) / 2)
    # print(f'Length: {length}')
    
    # Generate the tensor with the pattern [0, 0, 1, 1, 2, 2, ..., length]
    pattern_tensor = torch.repeat_interleave(torch.arange(length), 2)

    # If the number of nodes is odd, append the last cluster index
    if x.shape[0] % 2 != 0:
        pattern_tensor = pattern_tensor[:-2] # remove the last two indices corresponding to extra cluster
        pattern_tensor = torch.cat((pattern_tensor, pattern_tensor[-1].unsqueeze(0))) # append the previous cluster, thus forming a single cluster for the last 3 nodes
        
        # other approach is to use the length of tensor
        # pattern_tensor = torch.cat((pattern_tensor, torch.tensor([length - 2], dtype=torch.long)))
    
    return pattern_tensor

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

        layers = []
        
        # mlp_layers_dim_list = hidden_layers
        # mlp_layers_dim_list = [hidden_neurons, 2*hidden_neurons, 4*hidden_neurons, 2*hidden_neurons, hidden_neurons]
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


class GraphUNetEdgeConv(torch.nn.Module):
    r"""Graph-U-Net with Binary Pooling/Unpooling and EdgeConv

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        ec_mlp_width:int,
        ec_mlp_layer:int,
        out_channels: int,
        depth: int,
        act: Union[str, Callable] = 'relu',
    ):
        super().__init__()
        assert depth >= 1 # Depth 1 is the first conv, so if we want D0 then its depth=1, if we want D6 then depth=7
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.ec_mlp_width=ec_mlp_width
        self.ec_mlp_layer=ec_mlp_layer
        self.act = activation_resolver(act)

        channels = hidden_channels
        
        # Create the functions (nn.ModuleList) to apply at every EdgeConv
        self.down_convs = torch.nn.ModuleList()

        # First graph edge convolution during downsampling/pooling:
        self.down_convs.append(EdgeConv(nn=EdgeConvMLP(in_channels=self.in_channels, out_channels=self.hidden_channels, ec_mlp_width=self.ec_mlp_width, ec_mlp_layer=self.ec_mlp_layer, activation_function=self.act), aggr='max'))

        # Rest of graph edge convolutions
        for i in range(depth):
            self.down_convs.append(EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_channels, out_channels=self.hidden_channels, ec_mlp_width=self.ec_mlp_width, ec_mlp_layer=self.ec_mlp_layer, activation_function=self.act), aggr='max'))

        # Define the dimension of node features during unpooling
        # Define the unpooling/up-convolutions
        self.up_convs = torch.nn.ModuleList()
        
        # Unpooling from bottom to top-1 level
        for i in range(depth - 1):
            self.up_convs.append(EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_channels, out_channels=self.hidden_channels, ec_mlp_width=self.ec_mlp_width, ec_mlp_layer=self.ec_mlp_layer, activation_function=self.act), aggr='max'))
        
        # Final output graph: can make the node features = output or a higher dimension to be decoded by an mlp_output
        # self.up_convs.append(EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_channels, out_channels=self.out_channels, hidden_neurons=self.hidden_channels, hidden_layers=self.edge_conv_mlp_layers, activation_function=self.act), aggr='max'))

        # # MLP Decoder for the final output graph:
        #         # Initialize a list to hold the layers
        # layers = []

        # # Add the first layer
        # layers.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        # layers.append(self.act)

        # # # Add intermediate layers
        # # for i in range(len(self.decoder_layers) - 1):
        # #     layers.append(torch.nn.Linear(self.decoder_layers[i], self.decoder_layers[i + 1]))
        # #     layers.append(self.act)

        # # Add the final layer
        # layers.append(torch.nn.Linear(self.hidden_channels, self.out_channels))

        # # Create the Sequential model
        # self.mlp_decoder = torch.nn.Sequential(*layers)


    def forward(self, x: Tensor, edge_index: Tensor,
                batch: OptTensor = None) -> Tensor:
        
        device = x.device  # Get the device of the input tensor
        if self.depth >1:
            # First apply EdgeConv; others do that on the input graph
            x = self.down_convs[0](x, edge_index)
            # x = self.act(x)

            # Store the graph info at each stage of pooling and use during unpooling
            # xs = [x]
            # edge_indices = [edge_index]
            clusters = [] # Nodes used to cluster: currently [0, 0, 1, 1, ..., num_nodes/2] i.e assign every pair to one node in the downsampled level
            
            pooled_graph = Data(x=x, edge_index=edge_index)
            pooled_graph_list = [pooled_graph] # First element is the convolved graph

            for i in range(1, self.depth + 1): # start range from 1 i.e after first EdgeConv applied on the input graph, until index 'depth'
                
                cluster = binary_fusion_cluster_club_last(pooled_graph.x).to(device)  # Ensure cluster is on the same device
                # print(f"cluster size at depth={i-1} with N={pooled_graph.x.shape[0]}: {cluster.shape}")
                
                # Perform average pooling
                # print(f"{pooled_graph}")
                pooled_graph = avg_pool(cluster, pooled_graph)

                # apply convolution on the pooled graph
                x = self.down_convs[i](pooled_graph.x, pooled_graph.edge_index)
                # print(self.act)
                # x = self.act(x)
                
                # Store the node feature matrix, edge_indices until the bottom-most level
                if i < self.depth:
                    # print(f"Storing pooled graph at depth={i}")
                    pooled_graph_list.append(pooled_graph)
                    # xs += [x]
                    # edge_indices += [edge_index]
                    
                # clusters += [cluster] # contains the nodes indices of the pooled graph
            
            # print(f"Length of pooled_graph_list: {len(pooled_graph_list)}\n")
            # print(f"last pooled_graph_list: {pooled_graph_list[self.depth-1]}\n")
            # print(f"last pooled_graph_list: {pooled_graph_list[-1]}\n")
            # print(f"last x shape: {x.shape}\n")
            # print(f"last x: {x}\n")
            
            # list of node embeddings in unpooling/upscaling:
            # x_list = []
            unpool_x_list = []
            unpool_x_list.append(x)
            for i in range(self.depth-1):
                j = self.depth - 1 - i # j ... unpool layer
                
                # print(f"\nUnpooling layer: {j}")
                
                # res = xs[j] # residual to add from pool_layer to unpool_layer
                # duplicate_node_embeddings = torch.zeros_like(res)
                # print(f"Dimension of duplicate node embeddings: {res.shape}")
                
                pooled_graph = pooled_graph_list[j-1]
                pooled_residual_x = pooled_graph.x # residual to add from pool_layer to unpool_layer
                
                # edge_index = edge_indices[j]
                # cluster = clusters[j]
                # print(f"Cluster tensor: {cluster}")
                # print(f"Cluster tensor shape: {cluster.shape[0]}")

                # up = torch.zeros_like(res) # zero element - node feature matrix of unpooled graph
                # up[cluster] = x # copy the node feature values from bottom-most-level to location of nodes selected by TopKPool

                duplicate_node_embeddings = torch.zeros_like(pooled_residual_x) # zero element - node feature matrix of unpooled graph
                # print(f"\nDimension of residual node embeddings: {duplicate_node_embeddings.shape}")
                prev_layer_len = unpool_x_list[-1].shape[0]
                duplicate_node_embeddings[:2*prev_layer_len] = unpool_x_list[-1].repeat_interleave(2, dim=0) # copy the node feature values from bottom-level to upper-level
                
                if duplicate_node_embeddings.shape[0] != 2*prev_layer_len:
                    # Fill the last row with the previous row
                    duplicate_node_embeddings[-1] = duplicate_node_embeddings[-2]

                # Add the residual
                x = duplicate_node_embeddings + pooled_residual_x
                # print(f"Node embedding at depth={j}: {unpool_x_list[-1]}")

                # Apply convolution on the unpooled graph
                x = self.up_convs[i](x, pooled_graph.edge_index)
                unpool_x_list.append(x)
                # x_list.append(x)
                # print(f"Node embedding dimension at depth={j}: {unpool_x_list[-1].shape}")
                # x = self.act(x) if i < self.depth - 1 else x

            # Final convolution for decoder
            # x = self.up_convs[i+1](x, pooled_graph.edge_index) # edge_conv decoder
            # x = self.mlp_decoder(x)
            # print(f"\nNode embedding dimension at output: {x.shape}")
        
        elif self.depth==1: # depth=1, means no down-up sampling
            x = self.down_convs[0](x, edge_index)

        return x #, pooled_graph_list[0].x # for concatentating top-level, return the first conv as well
