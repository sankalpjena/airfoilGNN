#!/bin/bash

# Ensure correct python environment before running
# conda activate torch-lightning

# Load the python arguments
chmod +x config.sh # Provide execution rights to config.sh bash script
source config.sh

# Name the model
GNN_MODEL_NAME="gun_${FEATURE_TYPE}_V_${V_CYCLES}_D_${DEPTH}_L_${EC_MLP_LAYER}_H_${HIDDEN_CHANNELS}_W_${EC_MLP_WIDTH}_WD_${WEIGHT_DECAY}_S_${MANUAL_SEED}"

python train_gun.py --device=$DEVICE --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --in_channels=$IN_CHANNELS --out_channels=$OUT_CHANNELS --hidden_channels=$HIDDEN_CHANNELS --ec_mlp_width=$EC_MLP_WIDTH --ec_mlp_layer=$EC_MLP_LAYER --depth=$DEPTH --v_cycles=$V_CYCLES --lr_fixed=$LR_FIXED --weight_decay=$WEIGHT_DECAY --use_optimizer=$USE_OPTIMIZER --activation_function=$ACTIVATION_FUNCTION --model_name=$GNN_MODEL_NAME --manual_seed=$MANUAL_SEED --pyg_graph_path_train=$PYG_GRAPH_PATH_TRAIN --pyg_graph_path_test=$PYG_GRAPH_PATH_TEST

# Optional: Run post processing
python post_train_gun.py --in_channels=$IN_CHANNELS --out_channels=$OUT_CHANNELS --w_value=$EC_MLP_WIDTH --l_value=$EC_MLP_LAYER --depth=$DEPTH --v_cycles=$V_CYCLES --h_value=$HIDDEN_CHANNELS --model_name=$GNN_MODEL_NAME --seed=$MANUAL_SEED --pyg_graph_path_train=$PYG_GRAPH_PATH_TRAIN --pyg_graph_path_test=$PYG_GRAPH_PATH_TEST --lr_fixed=$LR_FIXED --weight_decay=$WEIGHT_DECAY

# Removes logs created during post processing
rm -r lightning_logs 