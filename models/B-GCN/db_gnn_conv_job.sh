#!/bin/bash
source config.sh

NUM_EC_CONV_LAYERS=1

# model name
GNN_MODEL_NAME="gnn_${FLOW_TYPE}_L_${NUM_EC_CONV_LAYERS}_H_${HIDDEN_CHANNELS}_W_${EC_MLP_WIDTH}_L2_${EC_MLP_LAYER}_WD_${WEIGHT_DECAY}_S_${MANUAL_SEED}"

python train_gnn.py --device=$DEVICE --max_epochs=$MAX_EPOCHS --batch_size=$BATCH_SIZE --in_channels=$IN_CHANNELS --out_channels=$OUT_CHANNELS --hidden_channels=$HIDDEN_CHANNELS --ec_mlp_width=$EC_MLP_WIDTH --ec_mlp_layer=$EC_MLP_LAYER --num_ec_conv_layers=$NUM_EC_CONV_LAYERS --lr_fixed=$LR_FIXED --weight_decay=$WEIGHT_DECAY  --use_optimizer=$USE_OPTIMIZER --activation_function=$ACTIVATION_FUNCTION --model_name=$GNN_MODEL_NAME --manual_seed=$MANUAL_SEED --pyg_graph_path_train=$PYG_GRAPH_PATH_TRAIN --pyg_graph_path_test=$PYG_GRAPH_PATH_TEST