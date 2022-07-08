#!/bin/bash -e
DEVICE_QTY=$1

[ "$DEVICE_QTY" != "" ] || { echo "Specify the # of GPU devices to use! (1st arg)" ; exit 1 ; }

PORT=$2

[ "$PORT" != "" ] || { echo "Specify the master port for multiprocessing! (2d arg)" ; exit 1 ; }

DATA_ROOT=$3

[ "$DATA_ROOT" != "" ] || { echo "Specify the dataset root (3d arg)" ; exit 1 ; }

EPOCH_QTY=$4
EPOCH_QTY_ARG=""
if [ "$EPOCH_QTY" != "" ] ; then
  EPOCH_QTY_ARG="--epoch_qty $EPOCH_QTY"
fi

echo "Argument for the # of epochs: $EPOCH_QTY_ARG"

# Standard model training
bcai_art_run.py sample_configs/ucf101/ucf101_normal.json \
         $EPOCH_QTY_ARG \
         --dataset.root "$DATA_ROOT"  \
         --general.device.device_qty $DEVICE_QTY --general.master_port $PORT
# Universal perturbation training
bcai_art_run.py sample_configs/ucf101/ucf101_pgd.json \
         $EPOCH_QTY_ARG \
         --dataset.root "$DATA_ROOT"  \
         --general.device.device_qty $DEVICE_QTY --general.master_port $PORT

