import torch
# Set the default tensor type to float
torch.set_default_dtype(torch.float32)
from torch_geometric.utils import to_networkx

import matplotlib.pyplot as plt

import networkx as nx
import csv
import copy
import os
import sklearn

def plot_pygraph(data):
    # Convert PyG graph to NetworkX graph
    G = to_networkx(data, node_attrs=['x'])

    # Extract positions for plotting
    pos = {i: (data.x[i, 0].item(), data.x[i, 1].item()) for i in range(data.x.shape[0])}

    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_size=5, node_color='skyblue', edge_color='gray')
    # plt.xlim(0.8,1.2)
    plt.title("Airfoil as Cycle Graph")
    plt.gca().set_aspect('equal')
    plt.show()

def write_csv_file(uinlet, aoa, naca_param, filename):
    """
    Writes case details to a CSV file.

    Args:
    uinlet (list): List of inlet velocities.
    aoa (list): List of angles of attack.
    naca_param (list): List of NACA4/5 parameters.
    filename (str): Name of the CSV file to write.

    Returns:
    None
    """
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["u_inlet", "aoa", "naca_param"])  # Write header row
        for uinlet, aoa, naca_param in zip(uinlet, aoa, naca_param):
            writer.writerow([uinlet, aoa, naca_param])

def get_dataset_keys(dataset):
    uinlet_list = []
    aoa_list = []
    naca_param_list = []
    for key in list(dataset.keys()):
        uinlet_list.append(float(key[0]))
        aoa_list.append(float(key[1]))
        naca_param_list.append(key[2])
    return uinlet_list, aoa_list, naca_param_list

def random_split_dataset(pyg_graph_dict, pyg_graph_dict_test, train_ratio=0.9, seed=None, save_path='./split_dataset_info'):
    """
    Splits the PyG graph dictionary into train and validation based on the provided ratios.
    Also writes the inlet velocities, angles of attack and NACA parmaeter values of each dataset to separate CSV and .pt files.

    Args:
        pyg_graph_dict_train (dict): Dictionary containing PyG graphs as values and case numbers as keys.
        train_ratio (float, optional): Ratio of train dataset size. Defaults to 0.8.
        val_ratio (float, optional): Ratio of validation dataset size. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        save_path (str, optional): Path to save dataset and seed files. Defaults to './dataset_info'.

    Returns:
        tuple: Tuple containing train, validation, and test dataset dictionaries.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        # torch.manual_seed(seed)
        seed_generator = torch.Generator().manual_seed(seed)

    # Convert pyg_graph_dict keys into a list
    keys = list(pyg_graph_dict.keys())
    total_size = len(keys)

    # Shuffle the keys to ensure randomness before splitting
    shuffled_keys = sklearn.utils.shuffle(keys, random_state = seed)

    # Calculate the sizes of the splits
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # Use torch.utils.data.random_split to split the dataset. Note: random_split() doesn't shuffle before split
    train_keys, val_keys = torch.utils.data.random_split(
        shuffled_keys, [train_size, val_size], generator=seed_generator)

    # Extract corresponding subsets of the PyG graph dictionary
    train_dataset_dict = {key: pyg_graph_dict[key] for key in train_keys}
    val_dataset_dict = {key: pyg_graph_dict[key] for key in val_keys}

    # Create the dataset_info folder if it doesn't exist
    save_path = save_path + '_seed_' + f'{seed}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Save the datasets along with the seed as .pt files
    torch.save({
        'train_dataset': train_dataset_dict,
        'val_dataset': val_dataset_dict,
        'test_dataset': pyg_graph_dict_test,
        'seed': seed
    }, os.path.join(save_path, f'random_split_dataset_{seed}.pt'))

    # Extract the individual components (uinlet, aoa, mpxx) and write to CSV
    write_csv_file([key[0] for key in train_keys],
                   [key[1] for key in train_keys],
                   [key[2] for key in train_keys],
                   os.path.join(save_path, 'train_dataset.csv'))
    
    write_csv_file([key[0] for key in val_keys],
                   [key[1] for key in val_keys],
                   [key[2] for key in val_keys],
                   os.path.join(save_path, 'val_dataset.csv'))
    
    # Also write Test Keys
    test_key_list = list(pyg_graph_dict_test.keys())
    write_csv_file([key[0] for key in test_key_list],
                   [key[1] for key in test_key_list],
                   [key[2] for key in test_key_list],
                   os.path.join(save_path, 'test_dataset.csv'))

    print(f'Data set is split into {train_size} training set and {val_size} validation set!\n')
    print(f'Splits and seed saved in {save_path}')

    # Save a figure of the parameter space
    split_data = torch.load(f'./split_dataset_info_seed_{int(seed)}/random_split_dataset_{int(seed)}.pt')
    train_dataset = split_data['train_dataset']
    val_dataset = split_data['val_dataset']
    test_dataset = split_data['test_dataset']
    seed_used = split_data['seed']
    
    # Train set
    plt.figure() # Plot the parameter space
    uinlet_list, aoa_list, mpxx_list = get_dataset_keys(train_dataset)
    plt.scatter(uinlet_list, aoa_list, label='Train')
    
    # Val set
    uinlet_list, aoa_list, mpxx_list = get_dataset_keys(val_dataset)
    plt.scatter(uinlet_list, aoa_list, marker='D', label='Validation')

    # Val set
    uinlet_list, aoa_list, mpxx_list = get_dataset_keys(test_dataset)
    plt.scatter(uinlet_list, aoa_list, marker='s', label='Validation')
    
    plt.xlabel('Inlet velocity')
    plt.ylabel('Angle of Attack')
    plt.title(f'Seed: {seed}')
    plt.legend(ncol=1, loc="upper right", bbox_to_anchor=(1.4, 0.5))
    plt.tight_layout()
    plt.savefig(f'./split_dataset_info_seed_{int(seed)}/' + f'param_space_seed_{seed_used}.png', dpi=300)

    return train_dataset_dict, val_dataset_dict


def calculate_range(tensor):
    """
    Calculates the minimum and maximum values along each dimension of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        tuple: Tuple containing the minimum and maximum values along each dimension.
    """
    min_value = tensor.min(dim=0).values.numpy()
    max_value = tensor.max(dim=0).values.numpy()
    return min_value, max_value

def print_dataset_range(dataset_dict, name='default'):
    """
    Prints the range of values in the dataset.

    Args:
        dataset_dict (dict): Dictionary containing the dataset.
        name (str): Name of the dataset.

    Returns:
        None
    """
    x_dataset = torch.cat([element.x for element in dataset_dict.values()], dim=0)
    y_dataset = torch.cat([element.y for element in dataset_dict.values()], dim=0)

    min_value_x, max_value_x = calculate_range(x_dataset)
    min_value_y, max_value_y = calculate_range(y_dataset)

    if min_value_x.shape[0] == 2:
        print(f'Dataset: {name}')
        print(f'(pos_x, pos_y) : [{min_value_x[0]}, {max_value_x[0]}], [{min_value_x[1]}, {max_value_x[1]}]!')
        print(f'(Cp) : [{min_value_y[0]}, {max_value_y[0]}]!\n')    
    
    elif min_value_x.shape[0] == 3:
        print(f'Dataset: {name}')
        print(f'(pos_x, pos_y, re) : [{min_value_x[0]}, {max_value_x[0]}], [{min_value_x[1]}, {max_value_x[1]}], [{min_value_x[2]}, {max_value_x[2]}]!')
        print(f'(Cp) : [{min_value_y[0]}, {max_value_y[0]}]!\n')    
    
    elif min_value_x.shape[0] == 4:
        print(f'Dataset: {name}')
        print(f'(pos_x, pos_y, re_stag, cp_inv) : [{min_value_x[0]}, {max_value_x[0]}], [{min_value_x[1]}, {max_value_x[1]}], [{min_value_x[2]}, {max_value_x[2]}], [{min_value_x[3]}, {max_value_x[3]}]!')
        print(f'(Cp) : [{min_value_y[0]}, {max_value_y[0]}]!\n')    
        
def get_minmax_normalization_stats(train_dataset_dict, save_path='./split_dataset_info'):
    """
    Calculates the minimum and maximum values for each feature dimension in the training dataset
    and exports the normalization statistics to a CSV file.

    Args:
        train_dataset_dict (dict): Dictionary containing training datasets with PyG graphs.

    Returns:
        list: List containing the minimum and maximum values for each feature dimension.
    """
    # Calculate the minimum and maximum values of each feature dimension in Data.x
    x_train = [element.x for element in train_dataset_dict.values()]  # Extract Data.x from training dataset
    x_train = torch.cat(x_train, dim=0)  # Concatenate all Data.x tensors
    
    min_x = x_train.min(dim=0).values
    max_x = x_train.max(dim=0).values
    
    # Calculate the minimum and maximum values of each feature dimension in Data.y
    y_train = [element.y for element in train_dataset_dict.values()]  # Extract Data.y from training dataset
    y_train = torch.cat(y_train, dim=0)  # Concatenate all Data.y tensors
    min_y = y_train.min(dim=0).values
    max_y = y_train.max(dim=0).values
    
    if min_x.shape[0] == 2:
        # Export normalization statistics to a CSV file
        stats = {
                'min_x[pos_x,pos_y]': min_x.tolist(),
                'max_x[pos_x,pos_y]': max_x.tolist(),
                'min_y[cp]': min_y.tolist(),
                'max_y[cp]': max_y.tolist()
            }
    
    elif min_x.shape[0] == 3:
        # Export normalization statistics to a CSV file
        stats = {
                'min_x[pos_x,pos_y,re]': min_x.tolist(),
                'max_x[pos_x,pos_y,re]': max_x.tolist(),
                'min_y[cp]': min_y.tolist(),
                'max_y[cp]': max_y.tolist()
            }
    
    elif min_x.shape[0] == 4:
        # Export normalization statistics to a CSV file
        stats = {
                'min_x[pos_x,pos_y,re_stag,cp_inv]': min_x.tolist(),
                'max_x[pos_x,pos_y,re_stag,cp_inv]': max_x.tolist(),
                'min_y[cp]': min_y.tolist(),
                'max_y[cp]': max_y.tolist()
            }
    
    csv_file_path = f'./{save_path}/minmax_normalization_stats.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
    
    norm_stats = [min_x, max_x, min_y, max_y]
    return norm_stats

def minmax_normalize_dataset(norm_stats, train_dataset_dict, val_dataset_dict, test_dataset_dict, save_path):
    """
    Normalizes the dataset using the provided normalization statistics.

    Args:
        norm_stats (list): List containing the normalization statistics.
        train_dataset_dict (dict): Dictionary containing the training dataset.
        val_dataset_dict (dict): Dictionary containing the validation dataset.
        test_dataset_dict (dict): Dictionary containing the test dataset.

    Returns:
        tuple: Tuple containing the normalized training, validation, and test datasets.
    """
    # Unpack the normalization statistics
    min_x, max_x, min_y, max_y = norm_stats
    
    if min_x.shape[0] == 2:
        print('Normalization stats are:\n')
        print(f'min_x[pos_x,pos_y]:{min_x}')
        print(f'max_x[pos_x,pos_y]: {max_x}')
        print(f'min_y[cp]: {min_y}')
        print(f'max_y[cp]: {max_y}')

    elif min_x.shape[0] == 3:
        print('Normalization stats are:\n')
        print(f'min_x[pos_x,pos_y,re]:{min_x}')
        print(f'max_x[pos_x,pos_y,re]: {max_x}')
        print(f'min_y[cp]: {min_y}')
        print(f'max_y[cp]: {max_y}')

    elif min_x.shape[0] == 4:
        print('Normalization stats are:\n')
        print(f'min_x[pos_x,pos_y,re_stag,cp_inv]:{min_x}')
        print(f'max_x[pos_x,pos_y,re_stag,cp_inv]: {max_x}')
        print(f'min_y[cp]: {min_y}')
        print(f'max_y[cp]: {max_y}')

    # Make copies of the dictionaries
    train_dataset_dict_copy = copy.deepcopy(train_dataset_dict)
    val_dataset_dict_copy = copy.deepcopy(val_dataset_dict)
    test_dataset_dict_copy = copy.deepcopy(test_dataset_dict)

    # Range of values before normalization (Data.x)
    print('BEFORE:\n')
    print_dataset_range(train_dataset_dict_copy, name='Training Set')
    print_dataset_range(val_dataset_dict_copy, name='Validation Set')
    print_dataset_range(test_dataset_dict_copy, name='Test Set')

    # Normalize feature and label vectors (Data.x, Data.y)
    for element in train_dataset_dict_copy.values():
        element.x = (element.x - min_x) / (max_x - min_x)
        element.y = (element.y - min_y) / (max_y - min_y)

    for element in val_dataset_dict_copy.values():
        element.x = (element.x - min_x) / (max_x - min_x)
        element.y = (element.y - min_y) / (max_y - min_y)

    for element in test_dataset_dict_copy.values():
        element.x = (element.x - min_x) / (max_x - min_x)
        element.y = (element.y - min_y) / (max_y - min_y)

    # Output the range of values after normalization
    print('AFTER:\n')
    print_dataset_range(train_dataset_dict_copy, name='Training Set')
    print_dataset_range(val_dataset_dict_copy, name='Validation Set')
    print_dataset_range(test_dataset_dict_copy, name='Test Set')

    print(f'Dataset normalized using Training Set statistics in {save_path}')

    return train_dataset_dict_copy, val_dataset_dict_copy, test_dataset_dict_copy