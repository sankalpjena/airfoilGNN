This repository contains the [data](./data/) and [models](./models/) used in the work [Predicting airfoil pressure distribution using boundary graph neural networks](https://arxiv.org/abs/2503.18638)


### Dependencies

- Python 3.10
- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- NumPy
- Matplotlib
- Pandas
- SciPy

### Install the dependencies using `conda`

```bash
conda env create -f airfoilGNN_environment.yml 
```

This will create a `conda` environment named `torch-lightning`. The `name` can be changed in [airfoilGNN_environment.yml](airfoilGNN_environment.yml)

Modify the line: `name: torch-lightning` -> `name: your-desired-env-name`

### Usage

#### Training a Model

[GNN Model Training](models/README.md)