# Model parameters
FEATURE_TYPE="GBF" # "GBF" or "PBF"

if [ "$FEATURE_TYPE" = "GBF" ]; then
    IN_CHANNELS=3 # GBF input has 3 features per node
    # Dataset
    PYG_GRAPH_PATH_TRAIN="../../data/airfrans_train_data_GBF.pt" # GBF input
    PYG_GRAPH_PATH_TEST="../../data/airfrans_test_data_GBF.pt" # 3 input model
else
    IN_CHANNELS=4 # PBF input has 4 features per node
    # Dataset
    PYG_GRAPH_PATH_TRAIN="../../data/airfrans_train_data_PBF.pt" # GBF input
    PYG_GRAPH_PATH_TEST="../../data/airfrans_test_data_PBF.pt" # 3 input model
fi

# Model parameters
V_CYCLES=1
DEPTH=6 # Number of graph pooling; NOTE: D=(DEPTH-1). For example, for full depth at D=5, use DEPTH=6.
OUT_CHANNELS=1 # Prediction dimension per node

# Hyperparameters
HIDDEN_CHANNELS=8
EC_MLP_WIDTH=128
EC_MLP_LAYER=2

# Other
DEVICE="cpu"
MAX_EPOCHS=2
BATCH_SIZE=32

# Learning Rate
WEIGHT_DECAY=0.0
LR_FIXED=1e-4
LR_INITIAL=1e-4
LR_GAMMA_DECAY=5e-3
USE_OPTIMIZER="adam"
ACTIVATION_FUNCTION="elu"

# Manual seed for the train-val split
MANUAL_SEED=888