#! /bin/bash
echo "configs: $1"
echo "DATASET.DATAROOT: $2"
python train.py --config $1 DATASET.DATAROOT $2 DATASET.MAP_FOLDER $2