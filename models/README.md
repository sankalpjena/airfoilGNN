# GNN Model Training

There are two architectures:
1. **B-GCN**: This is a single-level model. Each convolution aggregates information from 1-hop neighbors.
2. **B-GUN**: This is a multi-level model. The U-Net architecture ensures aggregation from multiple distant nodes in a single convolution operation.

Architecture descriptions can be found in [*Predicting airfoil pressure distribution using boundary graph neural networks*](https://arxiv.org/abs/2503.18638) (right-click to open in a new tab).

Each model has bash script `config.sh` to define the model parameters, input feature type `GBF/PBF`, model hyperparameters, and computing device `cpu/gpu`.

To train a model,
1. Define `config.sh`
2. From terminal, 
   ```
   # Provide execution rights to bash script
   chmod +x run_train_[gun/gcn].sh
   
   # Run the model
   ./run_train_[gun/gcn]
   ```

By default, 
- B-GUN coarsens the graph to full depth of $D=5$ 
- B-GCN has $K=20$ recursive convolutions
- Feature type is `GBF`.

Optionally, `clean_training.sh` can be used to delete checkpoints, and logs.

### Notes:

Inside `B-GUN/B-GCN` following files exist:
1. `config.sh` **:** to define the model parameters, hyperparameters, input type, computing device.
2. `train_[gcn/gun].py` **:** Python script to train the model
3. `post_train_[gcn/gun].py` **:** Python script to post-process trained model
4. `run_train_[gcn/gun].sh` **:** Bash script to load the arguments for `[train/post_train]_[gcn/gun].py` from `config.sh`. It first trains the model and runs the post-processing.
5.  `clean_training.sh` **:** Bash script to delete checkpoints, and logs.


The training process generates:

- Model checkpoints in `{MODEL_NAME}_checkpoints/`
- TensorBoard logs in `{MODEL_NAME}_tb_logs/`
- CSV logs in `{MODEL_NAME}_csv_logs/`
- Training snapshots in `training_snapshots_{MODEL_NAME}/`; NOTE: Implemented for GBF-B-GUN only
- Final post-processing results in `post_train_data/`
