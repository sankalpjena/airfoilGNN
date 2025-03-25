#!/bin/bash

for convs in  5 10 20 40 80 160 320 640 1280 2560 5120 10240 20480 40960 81920 163840 327680 655360 1310720 2621440; do

    FILE=db_gnn_L_${convs}.sh

    cp db_gnn_conv_job.sh $FILE

    # Modify the FILE
    sed -i '' 's/^#SBATCH --job-name=.*/#SBATCH --job-name="'L"${convs}"'"/' $FILE
    sed -i '' "s/^#SBATCH --output=.*/#SBATCH --output=L${convs}.%j.out/" $FILE
    sed -i '' "s/^#SBATCH --error=.*/#SBATCH --error=L${convs}.%j.err/" $FILE
    sed -i '' "s/^#SBATCH --time=.*/#SBATCH --time=04:00:00/" $FILE

    # Replace number of edge convolutions
    sed -i '' "s/^NUM_EC_CONV_LAYERS=.*/NUM_EC_CONV_LAYERS=${convs}/" $FILE
    
    # Provide executable rights
    chmod +x $FILE
done