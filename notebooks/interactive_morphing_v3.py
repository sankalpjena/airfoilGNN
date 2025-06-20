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
from src.utils import visualisation_parameters
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

    checkpoint = torch.load(TRAINED_MODEL_PATH + MODEL_NAME, map_location=torch.device('cpu'))
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
    pyg_graph_dict_test = torch.load(PYG_GRAPH_PATH_TEST_PBF)

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
    airfoil_panels.solve_gamma_O2(alpha=0*np.pi/180, kutta=[(0,-1)])  # solve for gamma

    # Replace this
    # cp_inv_vpm = get_cp_on_nodes(airfoil_panels, rotated_airfoil_coordinates_clockwise)
    
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
    # min_x = np.array(ast.literal_eval(minmax_df['min_x[pos_x,pos_y,re/cp_inv]'].values[0]))
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

def predict_pressure(model, data_vpm, minmax_df):
    """
    Predict pressure distribution VPM features.
    
    Args:
        model: Neural network model for prediction
        data_vpm: PyG Data object with VPM features
        minmax_df: DataFrame with normalization parameters
        
    Returns:
        array:  pred_real_vpm
    """
    # Normalize input data
    test_graph_sample_norm_vpm, min_x, max_x, min_y, max_y = minmax_normalize_sample(
        minmax_df, data_vpm, type='inference')
    
    # Predict using VPM features
    out_vpm = model.predict_step(test_graph_sample_norm_vpm, batch_idx=530)
    pred_real_vpm = (out_vpm.detach().numpy().flatten() * (max_y - min_y)) + min_y
    
    return pred_real_vpm

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
    pred_real_vpm = predict_pressure(gun_model_viscous, data_vpm, minmax_df)
    
    cl_predicted = np.trapezoid(pred_real_vpm, data_vpm.x[:,0])
    return new_airfoil, pred_real_vpm, cl_predicted

def rotate_airfoil_coordinates(airfoil_coordinates, angle_of_attack):
    angle_of_attack = angle_of_attack * (np.pi/180)
    rotated_airfoil_coordinates = np.zeros((airfoil_coordinates.shape[0], airfoil_coordinates.shape[1]))
    rotated_airfoil_coordinates[:,0] = airfoil_coordinates[:,0] * np.cos(angle_of_attack) + airfoil_coordinates[:,1] * np.sin(angle_of_attack)
    rotated_airfoil_coordinates[:,1] = - airfoil_coordinates[:,0] * np.sin(angle_of_attack) + airfoil_coordinates[:,1] * np.cos(angle_of_attack)
    return rotated_airfoil_coordinates

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
    m = st.slider("FFD Grid X (m)", 1, 4, 3)
    n = st.slider("FFD Grid Y (n)", 1, 2, 1)

xmin, xmax, ymin, ymax = 0.0, 1.0, -0.3, 0.3
if "control_points" not in st.session_state:
    st.session_state.control_points = generate_initial_control_points(xmin, xmax, ymin, ymax, m, n)
control_points = st.session_state.control_points

st.sidebar.subheader("Move Control Point (Y only)")
all_idx = [(i, j) for i in range(m+1) for j in range(n+1)]
labels = [f"P[{i},{j}]" for i,j in all_idx]
sel_label = st.sidebar.selectbox("Select", labels)
i_sel, j_sel = all_idx[labels.index(sel_label)]
dy = st.sidebar.slider("ΔY", -0.2, 0.2, 0.0, 0.005)
if dy != 0.0:
    control_points[i_sel, j_sel, 1] += dy # any change in control point gets added
    control_points[i_sel, j_sel, 1] += dy # any change in control point gets added

# === Airfoil + FFD  + B-GUN ===
# The model hyperparameters
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

# Load dataset stats
minmax_df, test_data = load_dataset_stats(TRAINED_MODEL_PATH)
test_keys = list(test_data.keys())
# airfrans_test_index = 99  # Change this index to select different test samples
key_ = test_keys[int(airfrans_test_index)]
U_infty = float(key_[0])
aoa = float(key_[1])
re = U_infty / 1.56e-5  # Reynolds number based on the free stream velocity and kinematic viscosity of air
airfoil_graph = test_data[key_]
rotated_airfoil_coordinates = airfoil_graph.x[:,0:2].numpy()
airfoil_coordinates = rotate_airfoil_coordinates(rotated_airfoil_coordinates, -aoa)
airfoil = copy.deepcopy(rotated_airfoil_coordinates)

# Generate airfoil graph using VPM features: modify from v1.ipynb
data_vpm = get_vpm_features(airfoil_graph, make_airfoil, aoa, re)
# Predict pressure
pred_real_vpm = predict_pressure(gun_model_viscous, data_vpm, minmax_df)
# === pressure ===
cp_original = pred_real_vpm #airfoil_graph.y.numpy().flatten()# =label
# cp_predicted = pred_real_vpm*0.7
cl_original = np.trapezoid(pred_real_vpm, data_vpm.x[:,0])

deformed, cp_predicted, cl_predicted = apply_bezier_ffd(airfoil, control_points, airfoil_graph, make_airfoil, aoa, re, gun_model_viscous, minmax_df)

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
    fig2.add_trace(go.Scatter(x=airfoil_coordinates[:, 0], y=cp_original,
                             mode='lines+markers', name='Original, ' + f'cl={cl_original:.2f}' ,
                             line=dict(color='black', width=2)))
    fig2.add_trace(go.Scatter(x=airfoil_coordinates[:, 0], y=cp_predicted,
                             mode='lines+markers', name='Deformed, ' + f'cl={cl_predicted:.2f}' ,
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
            title=dict(text='x', font=dict(size=AXIS_TITLE_SIZE)),
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