#! /bin/bash
echo "configs: $1"
echo "DATASET.DATAROOT: $2"
echo "PRETRAINED.PATH: $3"
python train.py --config $1 DATASET.DATAROOT $2 DATASET.MAP_FOLDER $2 PRETRAINED.PATH $3