"""
1. Plot the train and val loss
2. Create the test loss
3. Create sample inference for all test data
3. Store the min, max, near mean cases
4. Save figures
"""

import torch
from torchinfo import summary

import lightning as pl
import pandas as pd
import numpy as np

import seaborn as sns
sns.set_theme(context='talk') # sets theme for plots
sns.set_style("white") # sets background to white

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import your custom modules
import sys
root_path = '../../'
sys.path.append(root_path)
from src.core import gun_arch_binaryPool_edgeconv

# Import visualisation parameters
from src.utils.visualisation_parameters import *

# Native modules
from pathlib import Path
import ast
import copy
import argparse
import os

# Function to plot train and validation loss
def plot_train_val_loss(csv_path, output_path, title_name, sigma):
    
    data = pd.read_csv(csv_path)

    # Extract epoch, train and val losses
    epoch_np = data.iloc[::2]['epoch'].values
    train_loss_np = data['train_loss_epoch'].dropna().values
    val_loss_np = data['val_loss'].dropna().values

    train_loss_smth = gaussian_filter1d(train_loss_np, sigma=sigma)
    val_loss_smth = gaussian_filter1d(val_loss_np, sigma=sigma)

    # Find the size of epoch_np and train_loss_smth
    epoch_size = epoch_np.shape[0]
    train_size = train_loss_smth.shape[0]
    val_size = val_loss_smth.shape[0]
    # print(epoch_size, train_size, val_size)

    # get common size
    min_size = min(epoch_size, train_size, val_size)

    plt.figure() # figsize=(10, 6)
    plt.plot(epoch_np[:min_size], train_loss_smth[:min_size], label=r"$\mathcal{L}_{\mathrm{train}}$")
    plt.plot(epoch_np[:min_size], val_loss_smth[:min_size], label=r"$\mathcal{L}_{\mathrm{val}}$")
    plt.xlabel('Epoch')
    plt.ylabel('MSE (norm) Loss')
    plt.legend(loc="best")
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True, which='both')  # Show grid
    plt.title(f'{title_name}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def minmax_normalize_sample(IN_CHANNELS, minmax_df, sample_graph_dict):
    """
    Normalizes the sample graph using the provided normalization statistics.

    Args:
        norm_stats (list): List containing the normalization statistics.
        sample_graph_dict (dict): Dictionary containing the sample graph data.

    Returns:
        sample_graph_dict (dict): Dictionary containing the normalized sample graph data
    """
    
    # Unpack the normalization statistics
    # Convert string to list
    if IN_CHANNELS == 3:
        min_x = torch.tensor(ast.literal_eval(minmax_df['min_x[pos_x,pos_y,re]'].values[0]))
        max_x = torch.tensor(ast.literal_eval(minmax_df['max_x[pos_x,pos_y,re]'].values[0]))

    elif IN_CHANNELS == 4:
        min_x = torch.tensor(ast.literal_eval(minmax_df['min_x[pos_x,pos_y,re_stag,cp_inv]'].values[0]))
        max_x = torch.tensor(ast.literal_eval(minmax_df['max_x[pos_x,pos_y,re_stag,cp_inv]'].values[0]))

    min_y = torch.tensor(ast.literal_eval(minmax_df['min_y[cp]'].values[0]))
    max_y = torch.tensor(ast.literal_eval(minmax_df['max_y[cp]'].values[0]))
    
    # Make copies of the dictionaries
    sample_graph_dict_copy = copy.deepcopy(sample_graph_dict)
    
    # Normalize feature vector (Data.x)
    sample_graph_dict_copy.x = (sample_graph_dict_copy.x - min_x) / (max_x - min_x)
    sample_graph_dict_copy.x = sample_graph_dict_copy.x.float()
    sample_graph_dict_copy.y = (sample_graph_dict_copy.y - min_y) / (max_y - min_y)
    sample_graph_dict_copy.y = sample_graph_dict_copy.y.float()
    
    return sample_graph_dict_copy, min_x.float().numpy(), max_x.float().numpy(), min_y.float().numpy(), max_y.float().numpy()

# Function to create test loss
def create_model_stats(model, datamodule, train_time_path, output_path, csv_path):
    
    # Run the inference on test set
    trainer = pl.Trainer(accelerator="cpu")
    test_results = trainer.test(model=model, datamodule=datamodule)
    test_loss = test_results[0]['test_loss']

    # Also on validation set
    data = pd.read_csv(csv_path)

    # Extract epoch, train and val losses
    val_loss_np = data['val_loss'].dropna().values
    val_loss = val_loss_np[-1]
    
    # Model train time
    train_df = pd.read_csv(train_time_path)
    train_time = train_df[train_df.columns[1]].to_numpy()[0]

    # Model params
    model_summary = summary(model)
    model_params = model_summary.total_params
    
    # Create a DataFrame of the model stats
    # Ensure each array is 2D with shape (n, 1) i.e have a single column
    test_loss = np.array(test_loss).reshape(-1, 1)
    val_loss = np.array(val_loss).reshape(-1,1)
    train_time = np.array(train_time).reshape(-1, 1)
    model_params = np.array(model_params).reshape(-1, 1)

    # Stack the arrays horizontally: create n(=1) row and 3 columns
    data_ = np.hstack([test_loss, val_loss, train_time, model_params])

    df = pd.DataFrame(data=data_, columns=['test_loss', 'val_loss', 'train_time', 'params'])
    df.to_csv(output_path, index=False)

def d2f_dx2_fd(x, fx):
    """
    Computes the second-order finite difference second derivative
    Handles irregular spacing.
    At the left endpoint, forward difference is used.
    At the right endpoint, backward difference is used.
    
    Parameters:
        x (array): Array of x values (irregular spacing allowed).
        f_x (function): Function that takes x and returns f(x).
        
    Returns:
        derivative (array): Second derivatives of f(x) at x.
    """
    n = len(x)
    derivative = np.zeros(n)

    # Forward difference for left endpoint using the three-point formula
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]

    a0 = 2 / (h1 * (h1 + h2))
    a1 = -2 / (h1 * h2)
    a2 = 2 / (h2 * (h1 + h2))

    derivative[0] = (
        a0 * fx[0] +
        a1 * fx[1] +
        a2 * fx[2]
    )
    
    # Interior points using central differences
    for i in range(1, n - 1):
        h_i = x[i + 1] - x[i]
        h_im1 = x[i] - x[i - 1]

        # Apply the formula from the image
        term1 = fx[i + 1] + (h_i / h_im1) * fx[i - 1]
        term2 = (1 + h_i / h_im1) * fx[i]
        denominator = h_i * h_im1 * (1 + h_i / h_im1)

        # Print values for debugging
        # print(f"i: {i}, h_i: {h_i}, h_im1: {h_im1}, denominator: {denominator}")

        # Check for invalid values
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            print(f"Invalid value encountered at index {i}")
            continue

        derivative[i] = 2 * (term1 - term2) / denominator

    # Backward difference for right endpoint using the three-point formula
    h1 = x[-1] - x[-2]
    h2 = x[-2] - x[-3]

    b0 = 2 / (h1 * (h1 + h2))
    b1 = -2 / (h1 * h2)
    b2 = 2 / (h2 * (h1 + h2))

    derivative[-1] = (
        b0 * fx[-3] +
        b1 * fx[-2] +
        b2 * fx[-1]
    )

    return derivative

# Function to create sample inference for all test data
def create_sample_inference(model, IN_CHANNELS, test_data_df, pyg_graph_dict, minmax_df, output_path):
    sample_inference = {}
    for idx, row in test_data_df.iterrows():
        test_data_key = (row['u_inlet'], row['aoa'], row['naca_param'])
        test_graph_sample = pyg_graph_dict[test_data_key]
        test_graph_sample_norm, min_x, max_x, min_y, max_y = minmax_normalize_sample(IN_CHANNELS, minmax_df, test_graph_sample)

        # Data
        label_real = test_graph_sample.y[:, 0].numpy().flatten()

        model.eval()
        out_ = model.predict_step(test_graph_sample_norm, batch_idx=530)
        pred_real = (out_.detach().numpy().flatten() * (max_y - min_y)) + min_y
        
        # Metrics
        mse_ = mean_squared_error(label_real, pred_real)
        rmse_= np.sqrt(mse_)
        mae_ = mean_absolute_error(label_real, pred_real)
        mape_ = np.mean(np.abs((label_real - pred_real) / label_real)) * 100

        sample_inference[test_data_key] = {
            'input_real': test_graph_sample.x.numpy(),
            'label_real': label_real,
            'pred_real': pred_real.flatten(),
            'mse': mse_,
            'rmse': rmse_,
            'mae': mae_,
            'mape': mape_
        }

    torch.save(sample_inference, output_path)

# Function to store min, max, near mean cases
def store_cases(sample_inference, output_path):
    mse_list = [mean_squared_error(v['label_real'], v['pred_real']) for v in sample_inference.values()]
    mse_array = np.array(mse_list)
    min_idx = np.argmin(mse_array)
    max_idx = np.argmax(mse_array)
    mean_idx = np.argmin(np.abs(mse_array - np.mean(mse_array)))

    cases = {
        'min_case': list(sample_inference.keys())[min_idx],
        'max_case': list(sample_inference.keys())[max_idx],
        'mean_case': list(sample_inference.keys())[mean_idx]
    }

    torch.save(cases, output_path)

# Function to plot the min, max, near mean cases
def plot_cases(cases, sample_inference, output_path):
    
    fig, axs = plt.subplots(3,2, figsize=(15,18)) 
    
    for idx, key in enumerate(list(cases.keys())):
        
        case_key = cases[key]
        input_real = sample_inference[case_key]['input_real']
        label_real = sample_inference[case_key]['label_real']
        pred_real = sample_inference[case_key]['pred_real']
        rmse_ = sample_inference[case_key]['rmse']

        # Case data
        U_infty = float(case_key[0])
        aoa = float(case_key[1])
        re_ = U_infty/1.56e-5

        # Plot the airfoil profile
        axs[idx][0].plot(input_real[:,0], input_real[:,1], 'k-', label='XFOIL', linewidth=2)
        axs[idx][0].fill_between(input_real[:,0], input_real[:,1], color='skyblue', alpha=0.5)
        axs[idx][0].set_xlabel(r"$x/c$")
        axs[idx][0].set_ylabel(r"$y/c$")
        axs[idx][0].set_xlim(-0.5,1.5)
        axs[idx][0].set_ylim(-1.5,1.5)
        axs[idx][0].set_title(r"$Re$ = " + f"{re_:.2e}, " + r"$\alpha$ = " + f"{aoa:.1f} deg, ")
        axs[idx][0].grid('both')

        # Plot the cp profile
        rotated_airfoil_profile = np.array([input_real[:,0], input_real[:,1]]).T
        airfoil_profile = rotate_airfoil_coordinates(rotated_airfoil_profile, -aoa)
        axs[idx][1].plot(airfoil_profile[:,0], label_real, 'k-', label='XFOIL', linewidth=4)
        axs[idx][1].plot(airfoil_profile[:,0], pred_real, '-', label=f'RMSE:{rmse_:.2f}')
        axs[idx][1].set_xlabel(r"$x/c$")
        axs[idx][1].set_ylabel(r"$c_p$")
        axs[idx][1].legend(bbox_to_anchor=(1,1))
        axs[idx][1].set_title(f"{key}")
        axs[idx][1].grid(True, which='both')  # Show grid
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def rotate_airfoil_coordinates(airfoil_coordinates, angle_of_attack):
    angle_of_attack = angle_of_attack * (np.pi/180)
    rotated_airfoil_coordinates = np.zeros((airfoil_coordinates.shape[0], airfoil_coordinates.shape[1]))
    rotated_airfoil_coordinates[:,0] = airfoil_coordinates[:,0] * np.cos(angle_of_attack) + airfoil_coordinates[:,1] * np.sin(angle_of_attack)
    rotated_airfoil_coordinates[:,1] = - airfoil_coordinates[:,0] * np.sin(angle_of_attack) + airfoil_coordinates[:,1] * np.cos(angle_of_attack)
    return rotated_airfoil_coordinates

# Main function to orchestrate the tasks
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=False)
    parser.add_argument('--seed', type=str, required=False)
    parser.add_argument('--pyg_graph_path_train', type=str, required=False)
    parser.add_argument('--pyg_graph_path_test', type=str, required=False)
    parser.add_argument('--in_channels', type=int, required=False)
    parser.add_argument('--out_channels', type=int, required=False)
    parser.add_argument('--h_value', type=int, required=False)
    parser.add_argument('--w_value', type=int, required=False)
    parser.add_argument('--depth', type=int, required=False)
    parser.add_argument('--v_cycles', type=int, required=False)
    parser.add_argument('--l_value', type=int, required=False)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--lr_fixed', type=float, required=False)
    
    args = parser.parse_args()
    
    MODEL_NAME = args.model_name
    SEED = args.seed

    IN_CHANNELS = args.in_channels
    OUT_CHANNELS = args.out_channels
    DEPTH = args.depth
    V_CYCLES = args.v_cycles
    H_VALUE = args.h_value
    W_VALUE = args.w_value
    L_VALUE = args.l_value
    WEIGHT_DECAY = args.weight_decay
    LR_FIXED = args.lr_fixed
    
    pyg_graph_dict_path_train = args.pyg_graph_path_train
    pyg_graph_dict_path_test = args.pyg_graph_path_test
    
    train_time_path = f"training_info_{MODEL_NAME}.csv"

    mse_loss_csv_path = f"./{MODEL_NAME}_csv_logs/csv_logs/version_0/metrics.csv"
    
    # pyg_graph_dict_path = "/Users/sankalpjena/simulations/xfoil-sim/naca4-dataset/experiments/gun_viscous/dataset/pyg_dataset/pyg_graph_dict_maxnodes_160_rotated_repane_viscous_cp_with_stag.pt"
    
    dataset_info_path = f"./split_dataset_info_seed_{SEED}/"
    ckpt_folder_path = f"./{MODEL_NAME}_checkpoints/"

    # Create the folder to save post train data
    # Figures path
    figures_path = "./post_train_data/figures/"
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    
    # Paths and parameters
    train_val_loss_output_path = f'./post_train_data/figures/train_val_loss_{MODEL_NAME}.png'
    test_loss_output_path = f'./post_train_data/test_loss_{MODEL_NAME}.csv'

    minmax_stats_path = dataset_info_path + "minmax_normalization_stats.csv"
    test_data_path = dataset_info_path + "test_dataset.csv"
    
    file = "last.ckpt"
    model_checkpoint_path = ckpt_folder_path + file

    sample_inference_output_path = f'./post_train_data/sample_inference_{MODEL_NAME}.pt'
    min_max_mean_cases_output_path = f'./post_train_data/min_max_mean_cases_{MODEL_NAME}.pt'
    plot_cases_path = f'./post_train_data/figures/cases_{MODEL_NAME}.png'

    # Load data
    pyg_graph_dict_train = torch.load(pyg_graph_dict_path_train)
    pyg_graph_dict_test = torch.load(pyg_graph_dict_path_test)
    
    minmax_df = pd.read_csv(minmax_stats_path)
    test_data_df = pd.read_csv(test_data_path, dtype=str)

    # Load model
    gnn_model = gun_arch_binaryPool_edgeconv.LightningGraphUNetEdgeConv(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_channels=H_VALUE, depth=DEPTH, ec_mlp_width=W_VALUE, ec_mlp_layer=L_VALUE, v_cycles=V_CYCLES, lr=1e-4, weight_decay=WEIGHT_DECAY, use_optimizer="adam", activation_function="elu")
       
    checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
    gnn_model.load_state_dict(checkpoint["state_dict"])

    # disable randomness, dropout, etc...
    gnn_model.eval()

    # Create datamodule
    datamodule = gun_arch_binaryPool_edgeconv.GNNDataModule(pyg_graph_dict_train=pyg_graph_dict_train, pyg_graph_dict_test=pyg_graph_dict_test, manual_seed=SEED,batch_size_per_gpu=500)

    # Plot train and validation loss
    plot_train_val_loss(mse_loss_csv_path, train_val_loss_output_path, title_name=f"Loss curve - {DEPTH}", sigma=10)

    # Create test loss
    if os.path.exists(train_time_path):
        create_model_stats(gnn_model, datamodule, train_time_path, test_loss_output_path, mse_loss_csv_path)

    # Create sample inference for all test data
    create_sample_inference(gnn_model, IN_CHANNELS, test_data_df, pyg_graph_dict_test, minmax_df, sample_inference_output_path)

    # Store min, max, near mean cases
    sample_inference = torch.load(sample_inference_output_path)
    store_cases(sample_inference, min_max_mean_cases_output_path)

    # Plot min, max, near mean cases
    min_max_mean_cases = torch.load(min_max_mean_cases_output_path)
    plot_cases(min_max_mean_cases, sample_inference, plot_cases_path)

if __name__ == '__main__':
    
    main()