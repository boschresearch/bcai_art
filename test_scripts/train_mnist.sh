#!/bin/bash -e
DEVICE_QTY=$1

[ "$DEVICE_QTY" != "" ] || { echo "Specify the # of GPU devices to use! (1st arg)" ; exit 1 ; }

PORT=$2

[ "$PORT" != "" ] || { echo "Specify the master port for multiprocessing! (2d arg)" ; exit 1 ; }

# Standard model training
bcai_art_run.py sample_configs/mnist/mnist_normal.json \
         --general.device.device_qty $DEVICE_QTY --general.master_port $PORT

# PGD training
bcai_art_run.py sample_configs/mnist/mnist_pgd.json \
         --general.device.device_qty $DEVICE_QTY --general.master_port $PORT

# Free adv. training
bcai_art_run.py sample_configs/mnist/mnist_free.json \
         --general.device.device_qty $DEVICE_QTY --general.master_port $PORT

# Universal perturbation training
bcai_art_run.py sample_configs/mnist/mnist_univ_perturb.json \
         --general.device.device_qty $DEVICE_QTY --general.master_port $PORT

# Patch-attack training
bcai_art_run.py sample_configs/mnist/mnist_patch.json \
         --general.device.device_qty $DEVICE_QTY --general.master_port $PORT
