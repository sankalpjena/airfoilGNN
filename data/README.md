# Dataset for GNN

This folder contains preprocessed (NOT normalised) PyTorch `.pt` datasets for training Graph Neural Network (GNN) models.

### Description

The datasets are divided into training, testing, and extrapolation sets. Each dataset is available in two different feature representations:

1. **Geometry-based features (GBF)**: The node features consist of spatial coordinates and the global Reynolds number:

   $$
   \mathbf{x}_i = [x, y, \mathrm{Re}]^\mathrm{T}
   $$

2. **Physics-based features (PBF)**: The node features include spatial coordinates, local Reynolds number, and the inviscid pressure coefficient:

   ```math
   \mathbf{x}_i = [x, y, \mathrm{Re}_x, c_{p,\mathrm{inviscid}}]^\mathrm{T}
   ```

### File naming convention

Each dataset follows the naming convention:

```
 airfrans_[train/test/extrapolation]_data_[GBF/PBF].pt
```

where:

`train, test, and extrapolation` indicate the dataset split.

`GBF` and `PBF` specify the feature representation type.

### Notes

- The extrapolation dataset contains $\texttt{S809/27}$ airfoils beyond the training distribution, which consists of $\texttt{NACA4/5}$ airfoils.

- Each dataset is stored as a dictionary, which can be loaded using `torch.load('filename.pt')`. The keys represent the airfoil parameters along with flow conditions, and the corresponding values are PyTorch Geometric (PyG) graph objects.

- An example from the training set with GBF input is:

  ```
  ('36.622', '11.319', '3.941_5.424_1.0_16.283'): Data(x=[150, 3], edge_index=[2, 300], y=[150, 1])
  ```

  Here, the key `('36.622', '11.319', '3.941_5.424_1.0_16.283')` represents:

  - Freestream velocity: $U_{\infty} = 36.622 \ \mathrm{[m/s]}$
  - Angle of attack: $\alpha = 11.319 ^ {\circ}$
  - NACA airfoil parameters: $\texttt{NACA(3.941,5.424,1.0,16.283)}$

  The corresponding value is a PyG `Data` object with:

  - $N = 150$ nodes and $e = 300$ edges.
  - `Data.x`: Input tensor with 3 features per node.
  - `Data.y`: Output tensor with 1 feature per node.
  - `Data.edge_index`: Connectivity matrix in Coordinate (COO) format with shape $[2, e]$.

- The airfoil is represented as a ring graph with bidirectional edges.
![Alt Image Text](/images/example_airfoil_graph.png "Airfoil Graph")

- The airfoil and its pressure distribution is:
![Alt Image Text](/images/example_input_output.png "Airfoil and pressure distribution")


For further details on dataset preprocessing, refer to [link to root/preprocessing/README.md]
