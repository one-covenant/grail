#!/bin/bash

echo "Starting GRAIL pipeline (parallel)..."

echo "Starting Training (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 nohup grail -vv train > grail_train.log 2>&1 &
TRAIN_PID=$!
echo "  Training PID: $TRAIN_PID"
sleep 450


echo "Starting Validating (GPU 4)..."
CUDA_VISIBLE_DEVICES=4 nohup grail -vv validate > grail_validate.log 2>&1 &
VALIDATE_PID=$!
echo "  Validation PID: $VALIDATE_PID"

echo "Starting Mining (GPU 7)..."
CUDA_VISIBLE_DEVICES=7 nohup grail -vv mine > grail_mine.log 2>&1 &
MINE_PID=$!
echo "  Mining PID: $MINE_PID"

echo "All processes started in parallel!"

