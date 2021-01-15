#!/usr/bin/env bash
set -x
DATAPATH=YOUR_DATAPATH
LOGDIR=DIR_TO_SAVE_TRAINING_LOG
CUDA_VISIBLE_DEVICES=0 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti_train.txt --testlist ./filenames/shuffled_kitti12_val.txt \
    --logdir $LOGDIR \
    --ckpt_start_epoch 4713 --summary_freq 1000 \
    --epochs 4713 --lrepochs "4598,4690:4,2.5" \
    --batch_size 4 --test_batch_size 8 \
    --lr 0.0004 \
    --maxdisp 256
