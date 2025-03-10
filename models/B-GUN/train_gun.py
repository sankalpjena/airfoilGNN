import torch
import pandas as pd
# Set the default tensor type to float
torch.set_default_dtype(torch.float32)
import torch_geometric
from torchinfo import summary

import sys
root_path = '../../'
sys.path.append(root_path)
from src.core import data_utils_rotated_airfoil
from src.core import gun_arch_binaryPool_edgeconv
# Import the post processing callback
from src.utils.training_callbacks import PostProcessTraining # NOTE: Implemented for GBF-B-GUN only

# Lightning modules
import lightning as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# function call sets the float32 precision for matrix multiplications to 'high'
# which enables Tensor Core utilization and trades off precision for performance
# works only with A100 GPUs

# Check if CUDA is available before attempting to set the precision
if torch.cuda.is_available():
    if torch.cuda.get_device_name(0):
        torch.set_float32_matmul_precision('high')  # 'medium' or 'high'

# Native modules
import time
import argparse
import csv

def load_model_weights(gnn_model, checkpoint_path):
    """
    Load model weights from a checkpoint file.

    Args:
        gnn_model (pytorch_lightning.LightningModule): The GNN model to load the weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        pytorch_lightning.LightningModule: The GNN model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path)
    gnn_model.load_state_dict(checkpoint["state_dict"])
    print(f"Model weights loaded from {checkpoint_path}")
    return gnn_model

def main():
    """
    main() to train the graph neural network for airfoil surface pressure prediction.
    """
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train Graph-U-Net for airfoil surface pressure prediction")
    parser.add_argument('--device', type=str, required=True, help='Device: "cpu' or "gpu")
    parser.add_argument('--max_epochs', type=int, required=True, help='Maximum number of epochs for training')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size of training set')
    parser.add_argument('--in_channels', type=int, required=True, help='Input features per node')
    parser.add_argument('--out_channels', type=int, required=True, help='Output features per node')
    parser.add_argument('--hidden_channels', type=int, required=True, help='Hidden layer size (arbitrary, can be tuned)')
    parser.add_argument('--depth', type=int, required=True, help='Pooling and unpooling --depth times')
    parser.add_argument('--v_cycles', type=int)
    parser.add_argument('--ec_mlp_width', type=int)

    # Activation
    parser.add_argument('--activation_function', type=str, required=True, help='Activation function: relu, tanh, elu')
    
    # Optimizer
    parser.add_argument('--use_optimizer', type=str, required=True, help='"adam" or "adam_decay" or "lbfgs"')
    
    # Learning rates
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--lr_fixed', type=float, required=True, help='Fixed learning rate for Adam')
    parser.add_argument('--lr_initial', type=float, required=False, help='Initial LR')
    parser.add_argument('--lr_gamma_decay', type=float, required=False, help='Decay Rate LR')

    # Model checkpoint
    parser.add_argument('--ckpt_path', type=str, required=False, help='Path to model checkpoint')

    # GNN model name
    parser.add_argument('--model_name', type=str, required=True, help='Name of the GNN model')

    # Data split seed
    parser.add_argument('--manual_seed', type=int, required=True, help='Seed used in the splitting the dataset into train, val, and test sets')

    # Path to dataset
    parser.add_argument('--pyg_graph_path_train', type=str, required=True, help='Path to the PyGeometric graph dataset')
    parser.add_argument('--pyg_graph_path_test', type=str, required=True, help='Path to the PyGeometric graph dataset')

    parser.add_argument('--ec_mlp_layer', type=int)

    # Parse arguments
    args = parser.parse_args()
    MAX_EPOCHS = args.max_epochs
    BATCH_SIZE = args.batch_size
    IN_CHANNELS = args.in_channels
    OUT_CHANNELS = args.out_channels
    HIDDEN_CHANNELS = args.hidden_channels
    DEPTH = args.depth
    V_CYCLES = args.v_cycles
    EC_MLP_WIDTH = args.ec_mlp_width
    EC_MLP_LAYER = args.ec_mlp_layer

    CKPT_PATH = args.ckpt_path
    DEVICE = args.device

    # Activation function
    ACTIVATION_FUNCTION = args.activation_function

    # Optimizer
    USE_OPTIMIZER = args.use_optimizer
    
    # Learning rate
    WEIGHT_DECAY = args.weight_decay
    LR_FIXED = args.lr_fixed
    LR_INITIAL = args.lr_initial
    LR_GAMMA_DECAY = args.lr_gamma_decay

    # Model name
    GNN_MODEL_NAME = args.model_name
    
    # Manual seed
    MANUAL_SEED = args.manual_seed

    # Path to PyG graph dict
    PYG_GRAPH_PATH_TRAIN = args.pyg_graph_path_train
    PYG_GRAPH_PATH_TEST = args.pyg_graph_path_test

    # Set the seed for reproducibility
    pl.seed_everything(MANUAL_SEED)

    pyg_graph_dict_train = torch.load(PYG_GRAPH_PATH_TRAIN)
    pyg_graph_dict_test = torch.load(PYG_GRAPH_PATH_TEST)

    # Print details of a sample graph
    key_tuple = list(pyg_graph_dict_train.keys())[0]
    sample_graph_data = pyg_graph_dict_train[key_tuple]
    print(f"Data: {sample_graph_data}")
    # data_utils.plot_pygraph(sample_graph_data)

    # Load dataset using DataModule
    gnn_datamodule= gun_arch_binaryPool_edgeconv.GNNDataModule(pyg_graph_dict_train=pyg_graph_dict_train, pyg_graph_dict_test=pyg_graph_dict_test, manual_seed=MANUAL_SEED, batch_size_per_gpu=BATCH_SIZE)
    
    # Setup model
    gnn_model = gun_arch_binaryPool_edgeconv.LightningGraphUNetEdgeConv(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, depth=DEPTH, ec_mlp_width=EC_MLP_WIDTH, ec_mlp_layer=EC_MLP_LAYER, v_cycles=V_CYCLES, lr=LR_FIXED, weight_decay=WEIGHT_DECAY, lr_initial=LR_INITIAL, gamma_decay=LR_GAMMA_DECAY, use_optimizer=USE_OPTIMIZER, activation_function=ACTIVATION_FUNCTION)
    
    # Get the model parameters
    model_summary = summary(gnn_model)
    model_params = model_summary.total_params

    # If a previous model, with same weights exists, use it for initializing weights => Transfer Learning
    if CKPT_PATH:
        gnn_model = load_model_weights(gnn_model, CKPT_PATH)

    # Create TensorBoard and CSV loggers
    csv_logger = CSVLogger(save_dir=f'{GNN_MODEL_NAME}_csv_logs/', name='csv_logs')
    # Initialize the TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir=f'{GNN_MODEL_NAME}_tb_logs/', name="tb_logs")
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{GNN_MODEL_NAME}_checkpoints',
        filename='%s-{epoch:05d}-{train_loss:.5e}-{val_loss:.5e}' % (GNN_MODEL_NAME),
        monitor='val_loss',
        mode='min',
        save_top_k=10,
        save_last=True,
        save_weights_only=False
    )

    # Create the post-process callback - turned off by default
    post_process_callback = PostProcessTraining(
    interval=100,  # Run every 10 epochs
    pyg_graph_dict_test=pyg_graph_dict_test,
    dataset_info_path=f"./split_dataset_info_seed_{MANUAL_SEED}/",  # Pass the path instead of dataframes
    model_name=GNN_MODEL_NAME,
    output_dir=f"./training_snapshots_{GNN_MODEL_NAME}/",
    sigma=10
    )
    
    # If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`.
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=0, # or, match with the training batches
        logger=[tb_logger, csv_logger],
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback] # Optional: [post_process_callback]
    )
    
    # Record the start time
    start_time = time.time()
    # Run training
    trainer.fit(model=gnn_model, datamodule=gnn_datamodule)

    # Record the end time
    end_time = time.time()

    time_taken = end_time - start_time

    # Convert time taken to hours
    time_taken_minutes = time_taken / 60

    # Print the time taken in hours
    print(f"Time taken for training: {time_taken_minutes:.2f} minutes")

    # Save the training time and model info as a CSV file
    with open(f'training_info_{GNN_MODEL_NAME}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Model Params', 'Time Taken (minutes)'])
        writer.writerow([GNN_MODEL_NAME, f'{model_params}', f"{time_taken_minutes:.2f}"])

if __name__ == '__main__':
    main()