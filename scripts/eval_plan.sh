#! /bin/bash
echo "checkpoint: $1"
echo "dataroot: $2"
python evaluate.py --checkpoint $1 --dataroot $2