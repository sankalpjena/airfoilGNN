"""
Interactive airfoil morphing along with pressure distribution using surroagate
model: PBF-B-GUN
"""

# Imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import torch
from torch_geometric.data import Data
from math import comb

# Imports for the B-GUN model
import sys
sys.path.append('../')
from src.core import gun_arch_binaryPool_edgeconv
import ast
import copy

# Visualisation imports
# from src.utils import visualisation_parameters
import matplotlib.pyplot as plt

# Imports for the Vortex Panel Method (VPM)
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from vortexpanel import VortexPanel as vp
from scipy.interpolate import interp1d

# === Functions ===

def load_model(HYPER_PARAMS, MODEL_PARAMS, TRAINED_MODEL_PATH, MODEL_NAME):
    """
    Loads the trained B-GUN model
    """
    # Unpack hyperparameters
    H_VALUE = HYPER_PARAMS['H_VALUE']  # Dimension of the feature embeddings
    DEPTH = HYPER_PARAMS['DEPTH']  # Number of coarsening levels
    W_VALUE = HYPER_PARAMS['W_VALUE']  # Width of the edge convolution MLP
    L_VALUE = HYPER_PARAMS['L_VALUE']  # Number of layers in the edge convolution MLP
    WEIGHT_DECAY = HYPER_PARAMS['WEIGHT_DECAY']  # Weight decay for the optimizer; to deal with noisy predictions
    LEARNING_RATE = HYPER_PARAMS['LEARNING_RATE']  # Learning rate for the optimizer
    
    # Unpack model parameters
    IN_CHANNELS = MODEL_PARAMS['IN_CHANNELS']  # Number of input features at each node
    OUT_CHANNELS = MODEL_PARAMS['OUT_CHANNELS']  # Number of output features at each node
    
    # Initialize the model
    gun_model_viscous = gun_arch_binaryPool_edgeconv.LightningGraphUNetEdgeConv(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_channels=H_VALUE, depth=DEPTH, v_cycles=1, ec_mlp_width=W_VALUE, ec_mlp_layer=L_VALUE, weight_decay=WEIGHT_DECAY, lr=LEARNING_RATE)

    checkpoint = torch.load(TRAINED_MODEL_PATH + MODEL_NAME, map_location=torch.device('cpu'), weights_only=False)
    gun_model_viscous.load_state_dict(checkpoint["state_dict"])

    # disable randomness, dropout, etc...
    gun_model_viscous.eval()

    return gun_model_viscous

def load_dataset_stats(TRAINED_MODEL_PATH):
    """
    Loads test data set and training data stats
    """

    dataset_info_path = f"{TRAINED_MODEL_PATH}split_dataset_info_seed_888/"
    minmax_stats_path = dataset_info_path + "minmax_normalization_stats.csv"
    minmax_df = pd.read_csv(minmax_stats_path)
    PYG_GRAPH_PATH_TEST_PBF = "../data/airfrans_test_data_PBF.pt"
    pyg_graph_dict_test = torch.load(PYG_GRAPH_PATH_TEST_PBF, weights_only=False)

    return minmax_df, pyg_graph_dict_test

def compute_stagnation_point_max_cp(df):
    """
    Find the stagnation point based on maximum Cp.
    
    Args:
        df: DataFrame with columns 'x', 'y', 'cp_inv'
        
    Returns:
        tuple: (x_stag, y_stag) coordinates of the stagnation point
    """
    # Find row where cp_inv is maximum
    max_row = df['cp_inv'].idxmax()
    max_row_data = df.iloc[max_row]
    
    # Get the data
    x_stag = max_row_data['x']
    y_stag = max_row_data['y']
    
    return x_stag, y_stag

def make_airfoil(airfoil_coordinates):
    x = airfoil_coordinates[:, 0]
    y = airfoil_coordinates[:, 1]
    return vp.panelize(x,y)

#===================VPM functions=====================
def separate_monotonic_surfaces(df):
    """
    Separate the panels into monotonic surfaces meaning that the x-coordinates run from TE to LE in decreasing order
    """
    keys = list(df.keys())
    key_ = keys[0]  # First key is 'xc' or 'x' for the center and node DataFrames
    # 1. Find index of leading edge (minimum x)
    le_index = df[key_].idxmin()

    # 2. Split into upper and lower surface
    upper_surface = df.iloc[0:le_index+1]   # from TE to LE
    lower_surface = df.iloc[le_index+1:]     # from LE to TE
    return upper_surface, lower_surface

# Get panel node coordinates
def get_panel_node_coordinates(airfoil_panels):
    """
    Get the coordinates of the panel nodes.
    Arguments:
        airfoil_panels: VortexPanel object
    Returns:
        x: x-coordinates of the panel nodes
        y: y-coordinates of the panel nodes
    """
    x = airfoil_panels.get_array('x')[:,0]  # x-coordinates of the panels
    y = airfoil_panels.get_array('y')[:,0]  # y-coordinates of the panels

    # Create a DataFrame with the panel node coordinates
    panel_nodes_df = pd.DataFrame({
        'x': x,
        'y': y
    })
    return panel_nodes_df

def get_panel_centers_df(airfoil_panels):
    """
    Create a DataFrame with the x-coordinates and pressure coefficients of the panels.
    Arguments:
        airfoil_panels: VortexPanel object
    Returns:
        df: DataFrame with columns 'xc' (x-coordinates) and 'cp_inviscid' (pressure coefficients) in anti-clockwise order
    """
    xc = airfoil_panels.get_array('xc') # Panel center x-coordinates
    yc = airfoil_panels.get_array('yc') # Panel center y-coordinates
    cp_inviscid = 1- airfoil_panels.get_array('gamma')**2 # Pressure coefficient
    df = pd.DataFrame(
        {'xc': xc,
        'yc': yc,
        'cp_inviscid': cp_inviscid}
    )
    df = df.iloc[::-1].reset_index(drop=True)  # Reverse the order of panels to resemble the orderinig in XFOIL
    return df

from scipy.interpolate import interp1d

# Interpolate the pressure coefficients to the panel nodes
def interpolate_pressure_to_panel_nodes(panel_centers_df, panel_nodes_df):
    
    # Split the panel centers and nodes into upper and lower surfaces
    panel_center_upper, panel_center_lower = separate_monotonic_surfaces(panel_centers_df)
    panel_node_upper, panel_node_lower = separate_monotonic_surfaces(panel_nodes_df)

    # Create interpolating functions using panel centers
    # If new coordinate is more than max, use the first value which is the TE value
    # If new coordinate is less than min, use the last value which is the LE value
    # interp_upper = interp1d(panel_center_upper['xc'], panel_center_upper['cp_inviscid'], kind='linear', bounds_error=False, fill_value="extrapolate")
    upper_fill_near_te = panel_center_upper['cp_inviscid'].iloc[0]
    upper_fill_near_le = panel_center_upper['cp_inviscid'].iloc[-1]
    interp_upper = interp1d(panel_center_upper['xc'], panel_center_upper['cp_inviscid'], kind='linear', bounds_error=False, fill_value=(upper_fill_near_le,upper_fill_near_te))
    # Lower surface runs from LE to TE, so the fill values are swapped
    lower_fill_near_le = panel_center_lower['cp_inviscid'].iloc[0]
    lower_fill_near_te = panel_center_lower['cp_inviscid'].iloc[-1]
    interp_lower = interp1d(panel_center_lower['xc'], panel_center_lower['cp_inviscid'], kind='linear', bounds_error=False, fill_value=(lower_fill_near_le, lower_fill_near_te))
    cp_node_upper = interp_upper(panel_node_upper['x'])
    cp_node_lower = interp_lower(panel_node_lower['x'])

    # Stack the results into a single DataFrame
    cp_node = np.hstack((cp_node_upper, cp_node_lower))
    x_node = np.hstack((panel_node_upper['x'], panel_node_lower['x']))
    y_node = np.hstack((panel_node_upper['y'], panel_node_lower['y']))
    panel_nodes_df = pd.DataFrame({
        'x': x_node,
        'y': y_node,
        'cp_inviscid': cp_node
    })
    return panel_nodes_df
#===================VPM functions end=====================

def get_vpm_features(test_graph_sample, make_airfoil, aoa, re):
    """
    Get VPM-based features (cp_inv and re_stag).
    
    Args:
        test_graph_sample: PyG Data object with the test case
        make_airfoil: Function to create airfoil panels
        aoa: Angle of attack
        re: Reynolds number
        
    Returns:
        Data: PyG Data object with VPM-based features
    """
    # Extract and rotate coordinates
    rotated_airfoil_coordinates = test_graph_sample.x[:,0:2].numpy()
    
    # Get cp_invisicd from VPM
    # Reverse the order of points to make it clockwise
    rotated_airfoil_coordinates_clockwise = rotated_airfoil_coordinates[-1::-1, :]  
    airfoil_panels = make_airfoil(rotated_airfoil_coordinates_clockwise)  # make the shape
    # airfoil_panels.solve_gamma_O2(alpha=0*np.pi/180, kutta=[(0,-1)])  # solve for gamma
    airfoil_panels.solve_gamma(alpha=0*np.pi/180, kutta=[(0,-1)])  # solve for gamma
    
    # New code:
    panel_centers_df = get_panel_centers_df(airfoil_panels)
    panel_nodes_df_ = get_panel_node_coordinates(airfoil_panels)
    panel_nodes_df = interpolate_pressure_to_panel_nodes(panel_centers_df, panel_nodes_df_)

    # Create dataframe with VPM data
    airfoil_data_df = pd.DataFrame({
        'x': panel_nodes_df['x'],
        'y': panel_nodes_df['y'],
        'cp_inv': panel_nodes_df['cp_inviscid']
    })
    
    # Find stagnation point
    x_stag, y_stag = compute_stagnation_point_max_cp(airfoil_data_df)
    
    # Compute distance to stagnation point and re_stag
    x = airfoil_data_df['x'].values
    xr_to_stag = np.abs(x - x_stag)
    re_stag = xr_to_stag * re
    
    # Update dataframe with re_stag
    airfoil_data_df['re_stag'] = re_stag
    
    # Reorder columns to match expected input format
    airfoil_data_df = airfoil_data_df[['x', 'y', 're_stag', 'cp_inv']]
    
    # Create PyG Data object
    node_features = airfoil_data_df.values
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge index for a bidirectional cycle graph
    num_nodes = len(airfoil_data_df)
    indices = np.arange(num_nodes)
    edge_index = np.vstack((indices, (indices + 1) % num_nodes)).reshape(2, -1)
    edge_index = np.hstack((edge_index, edge_index[::-1]))  # Add bidirectional edges
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Create PyG Data object
    data_vpm = Data(x=node_features, edge_index=edge_index)
    
    return data_vpm

def minmax_normalize_sample(minmax_df, sample_graph_dict, type="train"):
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
    min_x = torch.tensor(ast.literal_eval(minmax_df['min_x[pos_x,pos_y,re_stag,cp_inv]'].values[0]))
    max_x = torch.tensor(ast.literal_eval(minmax_df['max_x[pos_x,pos_y,re_stag,cp_inv]'].values[0]))
    min_y = torch.tensor(ast.literal_eval(minmax_df['min_y[cp]'].values[0]))
    max_y = torch.tensor(ast.literal_eval(minmax_df['max_y[cp]'].values[0]))
    
    # Make copies of the dictionaries
    sample_graph_dict_copy = copy.deepcopy(sample_graph_dict)
    
    # Normalize feature vector (Data.x)
    sample_graph_dict_copy.x = (sample_graph_dict_copy.x - min_x) / (max_x - min_x)
    sample_graph_dict_copy.x = sample_graph_dict_copy.x.float()
    if type == "train":
        sample_graph_dict_copy.y = (sample_graph_dict_copy.y - min_y) / (max_y - min_y)
        sample_graph_dict_copy.y = sample_graph_dict_copy.y.float()
        
    return sample_graph_dict_copy, min_x.float().numpy(), max_x.float().numpy(), min_y.float().numpy(), max_y.float().numpy()

def cosine_distribution(n_points, x_start=0, x_end=1, min_le_value=0, max_te_value=0.999): # min_le_value=5e-5, max_te_value=0.999
    """
    Generates a non-uniform distribution of points, similar to XFOIL's repane method.
    Uses cosine spacing to cluster points near the leading and trailing edges.
    
    Args:
        n_points: Number of points to generate
        x_start: Start of the range (usually 0 for leading edge)
        x_end: End of the range (usually 1 for trailing edge)
        min_le_value: Minimum allowable value for the leading edge (x_LE).
        max_te_value: Maximum allowable value for the trailing edge (x_TE).
    
    Returns:
        Array of x-coordinates with clustered distribution.
    """
    # Adjust the start and end of the range
    adjusted_x_start = max(x_start, min_le_value)
    adjusted_x_end = min(x_end, max_te_value)
    
    # Generate cosine-spaced points using beta
    # beta = np.pi * np.linspace(1, 0, n_points + 1) # from airfrans
    beta = np.pi * np.linspace(adjusted_x_end, adjusted_x_start, n_points)
    cosine_points = (1 - np.cos(beta)) / 2  # Normalized to range [0, 1]
    
    # Scale and shift to the adjusted range [adjusted_x_start, adjusted_x_end]
    x = cosine_points * (adjusted_x_end - adjusted_x_start) + adjusted_x_start
    
    # # for airfoils going below 0 shift (translate) them to origin
    # if np.min(x)<0:
    #     x_min = np.abs(np.min(x))
    #     x = x + x_min

    # Ensure the final values are strictly within [adjusted_x_start, adjusted_x_end]
    x = np.clip(x, adjusted_x_start, adjusted_x_end)
    
    return x

def get_vpm_df(data_vpm):
    x_coord = data_vpm.x[:,0].numpy()
    y_coord = data_vpm.x[:,1].numpy()
    cp_vpm = data_vpm.x[:,3].numpy()
    vpm_df = pd.DataFrame({
        'x': x_coord,
        'y': y_coord,
        'cp_vpm': cp_vpm
    })
    return vpm_df

def get_inferred_aoa(vpm_df):
    """
    Get the inferred angle of attack based on the leading and trailing edge coordinates.
    NOTE: Not tested for airfoils with weird cambers
    """
    # Find the index of the maximum x-coordinate
    max_x_index = vpm_df['x'].idxmax()
    min_x_index = vpm_df['x'].idxmin()
    # Calculate the angle of attack based on the y-coordinates at the leading and trailing edges
    # By design, TE is at index 0
    x_te = vpm_df['x'].iloc[0]
    y_te = vpm_df['y'].iloc[0]

    # Leading edge based on farthest distance from the trailing edge ~ same as the minimum x-coordinate
    distances = np.sqrt((vpm_df['x'] - x_te)**2 + (vpm_df['y'] - y_te)**2)
    farthest_idx = distances.idxmax()
    print(f"Farthest point from TE: {farthest_idx}, x: {vpm_df['x'].iloc[farthest_idx]}, y: {vpm_df['y'].iloc[farthest_idx]}")
    # x_le = vpm_df['x'].iloc[farthest_idx]
    # y_le = vpm_df['y'].iloc[farthest_idx]
    x_le = vpm_df['x'].iloc[min_x_index]
    y_le = vpm_df['y'].iloc[min_x_index]
    aoa_inferred = np.arctan2(y_te - y_le, x_te - x_le)  # Assuming x-coordinates are normalized to [0, 1]
    aoa_inferred = np.degrees(aoa_inferred)  # Convert to degrees
    aoa_inferred *= -1  # By convention, the aoa is in the opposite direction

    return aoa_inferred

def get_downsampled_data_vpm(vpm_df, re):

    # 1. Split the monotonic upper and lower surfaces
    upper_surface, lower_surface = separate_monotonic_surfaces(vpm_df)

    # 2. Create interpolating functions of cp and y with the upper and lower surfaces
    # Create interpolating functions for cp and y using the upper and lower surfaces; near TE/LE previous values are used to fill
    cp_upper_fill_near_te = upper_surface['cp_vpm'].iloc[0] 
    cp_upper_fill_near_le = upper_surface['cp_vpm'].iloc[-1]
    cp_upper_interp = interp1d(upper_surface['x'], upper_surface['cp_vpm'], kind='linear', bounds_error=False, fill_value=(cp_upper_fill_near_le,cp_upper_fill_near_te))

    y_upper_fill_near_te = upper_surface['y'].iloc[0]
    y_upper_fill_near_le = upper_surface['y'].iloc[-1]
    y_upper_interp = interp1d(upper_surface['x'], upper_surface['y'], kind='linear', bounds_error=False, fill_value=(y_upper_fill_near_le,y_upper_fill_near_te))

    cp_lower_fill_near_te = lower_surface['cp_vpm'].iloc[-1]
    cp_lower_fill_near_le = lower_surface['cp_vpm'].iloc[0]
    cp_lower_interp = interp1d(lower_surface['x'], lower_surface['cp_vpm'], kind='linear', bounds_error=False, fill_value=(cp_lower_fill_near_le, cp_lower_fill_near_te))

    y_lower_fill_near_te = lower_surface['y'].iloc[-1]
    y_lower_fill_near_le = lower_surface['y'].iloc[0]
    y_lower_interp = interp1d(lower_surface['x'], lower_surface['y'], kind='linear', bounds_error=False, fill_value=(y_lower_fill_near_le,y_lower_fill_near_te))

    # 3. Get coarser x coordinates with cosine distribution
    x_cosine = cosine_distribution(n_points=80)
    x_cosine_airfoil = np.hstack((x_cosine, x_cosine[::-1]))
    y_cosine = np.zeros_like(x_cosine)  # y-coordinates are zero for the cosine distribution
    cos_distribution = np.column_stack((x_cosine, y_cosine))
    aoa_inferred = get_inferred_aoa(vpm_df)
    cos_distribution_rotated = rotate_airfoil_coordinates(cos_distribution, aoa_inferred)
    
    # 4. Get cp and y at the x_cosine distribution
    cp_upper = cp_upper_interp(cos_distribution_rotated[:,0])
    cp_lower = cp_lower_interp(cos_distribution_rotated[:,0])
    y_upper = y_upper_interp(cos_distribution_rotated[:,0])
    y_lower = y_lower_interp(cos_distribution_rotated[:,0])

    # Stack the results into a single DataFrame
    cp_node = np.hstack((cp_upper, cp_lower[::-1]))
    x_node = np.hstack((cos_distribution_rotated[:,0], cos_distribution_rotated[:,0][::-1]))
    y_node = np.hstack((y_upper, y_lower[::-1]))
    downsampled_df = pd.DataFrame({
        'x': x_node,
        'y': y_node,
        'cp_inv': cp_node
    })

    # 5. Create a df and apply re_stag on it
    # Find stagnation point
    x_stag, y_stag = compute_stagnation_point_max_cp(downsampled_df)
    
    # Compute distance to stagnation point and re_stag
    x = downsampled_df['x'].values
    xr_to_stag = np.abs(x - x_stag)
    re_stag = xr_to_stag * re
    
    # Update dataframe with re_stag
    downsampled_df['re_stag'] = re_stag
    
    # Reorder columns to match expected input format
    downsampled_df = downsampled_df[['x', 'y', 're_stag', 'cp_inv']]
    
    # 6. Create a new graph
    # Create PyG Data object
    node_features = downsampled_df.values
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge index for a bidirectional cycle graph
    num_nodes = len(downsampled_df)
    indices = np.arange(num_nodes)
    edge_index = np.vstack((indices, (indices + 1) % num_nodes)).reshape(2, -1)
    edge_index = np.hstack((edge_index, edge_index[::-1]))  # Add bidirectional edges
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Create PyG Data object
    downsampled_data_vpm = Data(x=node_features, edge_index=edge_index)

    return downsampled_data_vpm, x_cosine_airfoil

def predict_pressure(model, data_vpm, minmax_df, re):
    """
    Predict pressure distribution VPM features.
    
    Args:
        model: Neural network model for prediction
        data_vpm: PyG Data object with VPM features
        minmax_df: DataFrame with normalization parameters
        
    Returns:
        array:  pred_real_vpm
    """
    # Downsample the graph to N=160 nodes
    
    # Create a df of the original N=1024 graph
    vpm_df = get_vpm_df(data_vpm)

    # Get downsampled data
    downsampled_data_vpm, x_cosine_airfoil = get_downsampled_data_vpm(vpm_df, re)
    
    # Normalize input data
    test_graph_sample_norm_vpm, min_x, max_x, min_y, max_y = minmax_normalize_sample(
        minmax_df, downsampled_data_vpm, type='inference')
    
    # Predict using VPM features
    out_vpm = model.predict_step(test_graph_sample_norm_vpm, batch_idx=530)
    pred_real_vpm = (out_vpm.detach().numpy().flatten() * (max_y - min_y)) + min_y
    
    return pred_real_vpm, downsampled_data_vpm, x_cosine_airfoil

# === Bernstein, FFD, and Rotation ===
def bernstein_poly(i, n, t):
    """
    Compute the i-th Bernstein polynomial of degree n at t:
    B_{i,n}(t) = C(n,i) * t^i * (1 - t)^(n - i)
    """
    return comb(n, i) * (t**i) * ((1 - t)**(n - i))

def generate_initial_control_points(xmin, xmax, ymin, ymax, m, n):
    """
    Generate the initial (undeformed) control points in a uniform grid.
    Returns an array of shape (m+1, n+1, 2).
    """
    control_points = np.zeros((m+1, n+1, 2), dtype=float)
    
    for i in range(m+1):
        for j in range(n+1):
            # Equation (23) in: Masters, D.A., Taylor, N.J., Rendall, T., Allen, C.B., Poole, D.J., 2016. A Geometric Comparison of Aerofoil Shape Parameterisation Methods, in: 54th AIAA Aerospace Sciences Meeting. Presented at the 54th AIAA Aerospace Sciences Meeting, American Institute of Aeronautics and Astronautics, San Diego, California, USA. https://doi.org/10.2514/6.2016-0558

            x_ij = xmin + (xmax - xmin) * i / m
            y_ij = ymin + (ymax - ymin) * j / n
            control_points[i, j, 0] = x_ij
            control_points[i, j, 1] = y_ij

    return control_points

def apply_bezier_ffd(baseline_airfoil, control_points, airfoil_graph, make_airfoil, aoa, re, gun_model_viscous, minmax_df):
    """
    Apply 2D Bezier FFD to the given airfoil points using the provided control_points.
    
    baseline_airfoil: (N, 2) array of airfoil coordinates
    control_points:   (m+1, n+1, 2) array of control points
    
    Returns new_airfoil: (N, 2) array of deformed coordinates.
    """
    # Determine bounding box from the control points
    m = control_points.shape[0] - 1  # # of segments in x-direction
    n = control_points.shape[1] - 1  # # of segments in y-direction

    # Deduce bounding box from generate_initial_control_points:
    x_min = control_points[0, 0, 0]
    x_max = control_points[m, 0, 0]
    y_min = control_points[0, 0, 1]
    y_max = control_points[0, n, 1]

    # Prepare the deformed airfoil array
    new_airfoil = np.zeros_like(baseline_airfoil)

    for k, (xk, yk) in enumerate(baseline_airfoil):
        # 1) Compute the local (u, v) parameters \xi, \eta in [0,1]
        #    Here u <-> x, v <-> y
        u = (xk - x_min) / (x_max - x_min) if x_max != x_min else 0.0
        v = (yk - y_min) / (y_max - y_min) if y_max != y_min else 0.0

        # 2) Bézier surface mapping:
        #    X(u, v) = sum_{i=0}^{m} sum_{j=0}^{n} B_{i,m}(u) * B_{j,n}(v) * P_{i,j}
        X_deformed = 0.0
        Y_deformed = 0.0
        for i in range(m+1):
            B_i = bernstein_poly(i, m, u)
            for j in range(n+1):
                B_j = bernstein_poly(j, n, v)
                X_deformed += B_i * B_j * control_points[i, j, 0]
                Y_deformed += B_i * B_j * control_points[i, j, 1]

        new_airfoil[k, 0] = X_deformed
        new_airfoil[k, 1] = Y_deformed

    # Create a new airfoil graph with the deformed coordinates
    new_airfoil_graph = copy.deepcopy(airfoil_graph)
    
    new_airfoil_graph.x[:,0] = torch.tensor(new_airfoil[:, 0])
    new_airfoil_graph.x[:,1] = torch.tensor(new_airfoil[:, 1])

    # Generate airfoil graph using VPM features
    data_vpm = get_vpm_features(new_airfoil_graph, make_airfoil, aoa, re)
    
    # Predict pressure
    pred_real_vpm, downsampled_data_vpm, x_cosine_airfoil = predict_pressure(gun_model_viscous, data_vpm, minmax_df, re)
    
    cl_predicted = np.trapezoid(pred_real_vpm, x_cosine_airfoil)
    return new_airfoil, pred_real_vpm, cl_predicted, downsampled_data_vpm, x_cosine_airfoil

def rotate_airfoil_coordinates(airfoil_coordinates, angle_of_attack):
    angle_of_attack = angle_of_attack * (np.pi/180)
    rotated_airfoil_coordinates = np.zeros((airfoil_coordinates.shape[0], airfoil_coordinates.shape[1]))
    rotated_airfoil_coordinates[:,0] = airfoil_coordinates[:,0] * np.cos(angle_of_attack) + airfoil_coordinates[:,1] * np.sin(angle_of_attack)
    rotated_airfoil_coordinates[:,1] = - airfoil_coordinates[:,0] * np.sin(angle_of_attack) + airfoil_coordinates[:,1] * np.cos(angle_of_attack)
    return rotated_airfoil_coordinates

#============NACA Generator==============
# Source: https://airfrans.readthedocs.io/en/latest/_modules/airfrans/naca_generator.html#naca_generator

def thickness_dist(t, x, CTE = True):
    """
    Standard NACA profile to warp with the help of a camber line to define all the
    4 and 5 digits profiles.

    Args:
        t (float): Thickness of the airfoil in percentage of the chord length.
        x (np.ndarray): Abscissas in chord unit.
        CTE (bool, optional): If ``True`` the profile will be closed at the trailing edge. Default: ``True``
    """
    # CTE for close trailing edge
    if CTE:
        a = -0.1036
    else:
        a = -0.1015
    return 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 + a*x**4)

def camber_line(params, x):
    """
    Camber line definition for the NACA 4 and 5 digits series.

    Args:
        params (np.ndarray): Parameters of the NACA 4 or 5 digits profile (ndarray of shape `(3)` or `(4)`).
        x (np.ndarray): Abscissas in chord unit.
    """
    y_c = np.zeros_like(x)
    dy_c = np.zeros_like(x)

    if len(params) == 2:
        m = params[0]/100
        p = params[1]/10

        if p == 0:
            dy_c = -2*m*x
            return y_c, dy_c
        elif p == 1:
            dy_c = 2*m*(1 - x)
            return y_c, dy_c

        mask1 = (x < p)
        mask2 = (x >= p)
        y_c[mask1] = (m/p**2)*(2*p*x[mask1] - x[mask1]**2)
        dy_c[mask1] = (2*m/p**2)*(p - x[mask1])
        y_c[mask2] = (m/(1 - p)**2)*((1 - 2*p) + 2*p*x[mask2] - x[mask2]**2)
        dy_c[mask2] = (2*m/(1 - p)**2)*(p - x[mask2])

    elif len(params) == 3:
        l, p, q = params
        c_l, x_f = 3/20*l, p/20

        f = lambda x: x*(1 - np.sqrt(x/3)) - x_f
        df = lambda x: 1 - 3*np.sqrt(x/3)/2
        old_m = np.array(0.5)
        cond = True
        while cond:
            new_m = np.max(np.array([old_m - f(old_m)/df(old_m), 0]))
            cond = (np.abs(old_m - new_m) > 1e-15)
            old_m = new_m        
        m = old_m
        r = (3*m - 7*m**2 + 8*m**3 - 4*m**4)/np.sqrt(m*(1 - m)) - 3/2*(1 - 2*m)*(np.pi/2 - np.arcsin(1 - 2*m))
        k_1 = c_l/r

        mask1 = (x <= m)
        mask2 = (x > m)
        if q == 0:            
            y_c[mask1] = k_1*((x[mask1]**3 - 3*m*x[mask1]**2 + m**2*(3 - m)*x[mask1]))
            dy_c[mask1] = k_1*(3*x[mask1]**2 - 6*m*x[mask1] + m**2*(3 - m))
            y_c[mask2] = k_1*m**3*(1 - x[mask2])
            dy_c[mask2] = -k_1*m**3*np.ones_like(dy_c[mask2])

        elif q == 1:
            k = (3*(m - x_f)**2 - m**3)/(1 - m)**3
            y_c[mask1] = k_1*((x[mask1] - m)**3 - k*(1 - m)**3*x[mask1] - m**3*x[mask1] + m**3)
            dy_c[mask1] = k_1*(3*(x[mask1] - m)**2 - k*(1 - m)**3 - m**3)
            y_c[mask2] = k_1*(k*(x[mask2] - m)**3 - k*(1 - m)**3*x[mask2] - m**3*x[mask2] + m**3)
            dy_c[mask2] = k_1*(3*k*(x[mask2] - m)**2 - k*(1 - m)**3 - m**3)

        else:
            raise ValueError('Q must be 0 for normal camber or 1 for reflex camber.')

    else:
        raise ValueError('The first input must be a tuple of the 2 or 3 digits that represent the camber line.')   

    return y_c, dy_c

def naca_generator(params, nb_samples = 400, scale = 1, origin = (0, 0), cosine_spacing = True, verbose = True, CTE = True):
    """
    Definition of a complete profile from the NACA 4 and 5 digits series.

    Args:
        params (np.ndarray): Parameters of the NACA 4 or 5 digits profile (ndarray of shape `(3)` or `(4)`).
        nb_samples (int, optional): Number of points to define the profile. Default: 400
        scale (float, optional): Chord length in meters. Default: 1
        origine (tuple, optional): Absolute position of the leading edge. Default: `(0, 0)`
        cosine_spacing (bool, optional): If ``True``, points are sampled via a cosine distance instead of uniformly. Default: ``True``
        verbose (bool, optional): Comments on the generation process. Default: ``True``
        CTE (bool, optional): If ``True`` the profile will be closed at the trailing edge. Default: ``True``
    """
    if len(params) == 3:
        params_c = params[:2]
        t = params[2]/100
        if verbose:
            print(f'Generating naca M = {params_c[0]}, P = {params_c[1]}, XX = {t*100}')
    elif len(params) == 4:
        params_c = params[:3]
        t = params[3]/100
        if verbose:
            print(f'Generating naca L = {params_c[0]}, P = {params_c[1]}, Q = {params_c[2]}, XX = {t*100}')
    else:
        raise ValueError('The first argument must be a tuple of the 4 or 5 digits of the airfoil.')    

    if cosine_spacing:
        beta = np.pi*np.linspace(1, 0, nb_samples + 1)
        x = (1 - np.cos(beta))/2
    else:
        x = np.linspace(1, 0, nb_samples + 1)

    y_c, dy_c = camber_line(params_c, x)
    y_t = thickness_dist(t, x, CTE)
    theta = np.arctan(dy_c)
    x_u = x - y_t*np.sin(theta)
    x_l = x + y_t*np.sin(theta)
    y_u = y_c + y_t*np.cos(theta)
    y_l = y_c - y_t*np.cos(theta)
    x = np.concatenate([x_u, x_l[:-1][::-1]], axis = 0)
    y = np.concatenate([y_u, y_l[:-1][::-1]], axis = 0)
    pos = np.stack([
            x*scale + origin[0],
            y*scale + origin[1]
        ], axis = -1
    )
    pos[0], pos[-1] = np.array([1, 0]), np.array([1, 0])
    return pos
#========================================
# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("Airfoil morping using Bezier FFD with PBF-B-GUN")

with st.sidebar:
    st.header("Initial Airfoil Geometry")
    airfrans_test_index = st.text_input("airfRANS Test Index", "99")
    st.header("FFD Controls")
    # aoa = st.slider("Angle of Attack (deg)", -5.0, 15.0, 0.0, 0.5)
    # u_inlet = st.slider("Free Stream Velocity (m/s)", 10.0, 100.0, 40.0, 1.0)
    # re = u_inlet / 1.56e-5
    # m = st.slider("FFD Grid X (m)", 1, 4, 3)
    # n = st.slider("FFD Grid Y (n)", 1, 2, 1)

xmin, xmax, ymin, ymax = 0.0, 1.0, -0.3, 0.3
m, n = 2,1
if "control_points" not in st.session_state:
    st.session_state.control_points = generate_initial_control_points(xmin, xmax, ymin, ymax, m, n)
control_points = st.session_state.control_points

st.sidebar.subheader("Move Control Point (y only)")
all_idx = [(i, j) for i in range(m+1) for j in range(n+1)]
labels = [f"P[{i},{j}]" for i,j in all_idx]
sel_label = st.sidebar.selectbox("Select", labels)
i_sel, j_sel = all_idx[labels.index(sel_label)]
dy = st.sidebar.slider("Δy", -0.2, 0.2, 0.0, 0.005)
if dy != 0.0:
    control_points[i_sel, j_sel, 1] += dy # any change in control point gets added
    control_points[i_sel, j_sel, 1] += dy # any change in control point gets added

# === Airfoil + FFD  + B-GUN ===
# B-GUN model hyperparameters
H_VALUE = 8 # Dimension of the feature embeddings
DEPTH = 6 # Number of coarsening levels
W_VALUE = 128 # Width of the edge convolution MLP
L_VALUE = 2 # Number of layers in the edge convolution MLP
WEIGHT_DECAY = 0.0 # Weight decay for the optimizer; to deal with noisy predictions
LEARNING_RATE = 1e-4 # Learning rate for the optimizer

HYPER_PARAMS = {
    "H_VALUE": H_VALUE,
    "DEPTH": DEPTH,
    "W_VALUE": W_VALUE,
    "L_VALUE": L_VALUE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "LEARNING_RATE": LEARNING_RATE
}

# Model parameters
IN_CHANNELS = 4 # Number of input features at each node
OUT_CHANNELS = 1 # Number of output features at each node

MODEL_PARAMS = {
    "IN_CHANNELS": IN_CHANNELS,
    "OUT_CHANNELS": OUT_CHANNELS
}

# Path to model
MODEL_NAME = "pbf-bgun-epoch=24060-train_loss=3.23646e-07-val_loss=1.24484e-06.ckpt"
TRAINED_MODEL_PATH = "../models/trained_models/PBF-B-GUN/"

# Load the model
gun_model_viscous = load_model(HYPER_PARAMS, MODEL_PARAMS, TRAINED_MODEL_PATH, MODEL_NAME)

# Load test dataset and normalisation stats
minmax_df, test_data = load_dataset_stats(TRAINED_MODEL_PATH)
test_keys = list(test_data.keys())
key_ = test_keys[int(airfrans_test_index)]
# Flow conditions and NACA params from the key
U_infty = float(key_[0])
aoa = float(key_[1])
re = U_infty / 1.56e-5  # Reynolds number based on the free stream velocity and kinematic viscosity of air
naca_params = [float(p) for p in key_[2].split('_')] # Can be NACA4/5

# Generate the airfoil from the NACA parameters
nodes_on_airfoil = 1024 # with these many points, all to all communication doesn't happen in the B-GUN architecture with 6 graph coarsening levels
naca_coords = naca_generator(naca_params, nb_samples=int(nodes_on_airfoil/2), CTE=True, cosine_spacing=True)
rotated_airfoil_coordinates = rotate_airfoil_coordinates(naca_coords, aoa)

# Create an airfoil graph from the airfoil coordinates
# Create edge index for a bidirectional cycle graph using NumPy
num_nodes = rotated_airfoil_coordinates.shape[0]
indices = np.arange(num_nodes)
edge_index = np.vstack((indices, (indices + 1) % num_nodes)).reshape(2, -1)
edge_index = np.hstack((edge_index, edge_index[::-1]))  # Add bidirectional edges
edge_index = torch.tensor(edge_index, dtype=torch.long)

# Create PyG Data Graph object with node coordinates as features
airfoil_graph = Data(x=torch.tensor(rotated_airfoil_coordinates), edge_index=edge_index)

# airfoil_graph = test_data[key_] # Airfoil as graph created from RANS simulation data
# rotated_airfoil_coordinates = airfoil_graph.x[:,0:2].numpy()
airfoil_coordinates = rotate_airfoil_coordinates(rotated_airfoil_coordinates, -aoa)
airfoil = copy.deepcopy(rotated_airfoil_coordinates)

# Generate airfoil graph using VPM features
data_vpm = get_vpm_features(airfoil_graph, make_airfoil, aoa, re)

# Predict pressure
# pred_real_vpm = predict_pressure(gun_model_viscous, data_vpm, minmax_df)
pred_real_vpm, downsampled_data_vpm, x_cosine_airfoil = predict_pressure(gun_model_viscous, data_vpm, minmax_df, re)

# === pressure ===
cp_original = pred_real_vpm #airfoil_graph.y.numpy().flatten()# =label

# cp_predicted = pred_real_vpm*0.7
cl_original = np.trapezoid(pred_real_vpm, x_cosine_airfoil)

# deformed, cp_predicted, cl_predicted = apply_bezier_ffd(airfoil, control_points, airfoil_graph, make_airfoil, aoa, re, gun_model_viscous, minmax_df)
deformed, cp_predicted, cl_predicted, downsampled_data_vpm, x_cosine_airfoil = apply_bezier_ffd(airfoil, control_points, airfoil_graph, make_airfoil, aoa, re, gun_model_viscous, minmax_df)

# === Plot ===
# V1
# Create two columns with a 50:50 width ratio
col1, col2 = st.columns(2)

# Define font sizes
TITLE_SIZE = 24
AXIS_TITLE_SIZE = 20
TICK_SIZE = 20
LEGEND_SIZE = 20

with col1:
    # === Airfoil Plot ===
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=airfoil[:,0], y=airfoil[:,1], mode='lines', name='Original', line=dict(color='black', width=2)))
    fig1.add_trace(go.Scatter(x=deformed[:,0], y=deformed[:,1], mode='lines', name='Deformed', line=dict(color='blue', width=2)))
    fig1.add_trace(go.Scatter(x=control_points[:,:,0].flatten(), y=control_points[:,:,1].flatten(),
                             mode='markers+text', text=labels,
                             textposition="top center", name='Control Points'))
    fig1.update_layout(
        height=500, 
        width=500,
        # Set title font
        title=dict(
            text=f"Original case: AoA = {aoa}°; U_infty = {U_infty} m/s, Re={U_infty/1.56e-5:.2e}",
            font=dict(size=TITLE_SIZE)
        ),
        # Set global font
        font=dict(size=TICK_SIZE),
        # Set axis properties
        xaxis=dict(
            title=dict(text='x', font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE)
        ),
        yaxis=dict(
            scaleanchor="x", 
            scaleratio=1,
            title=dict(text='y', font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE)
        ),
        # Set legend properties
        legend=dict(font=dict(size=LEGEND_SIZE)),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # === Pressure Distribution Plot ===
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_cosine_airfoil, y=cp_original,
                             mode='lines', name='Original, ' + f'cl={cl_original:.2f}' ,
                             line=dict(color='black', width=2)))
    fig2.add_trace(go.Scatter(x=x_cosine_airfoil, y=cp_predicted,
                             mode='lines', name='Deformed, ' + f'cl={cl_predicted:.2f}' ,
                             line=dict(color='blue', width=2)))
    fig2.update_layout(
        height=500,
        width=500,
        # Set title font
        title=dict(
            text="Pressure Coefficient (cp)",
            font=dict(size=TITLE_SIZE)
        ),
        # Set global font
        font=dict(size=TICK_SIZE),
        # Set axis properties
        xaxis=dict(
            title=dict(text='x/c', font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE)
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=0.3,
            autorange="reversed",
            title=dict(text='cp', font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE)
        ),
        # Set legend properties
        legend=dict(font=dict(size=LEGEND_SIZE)),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)