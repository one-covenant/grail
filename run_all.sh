#!/bin/bash

# Start trainer on GPU 6 in test mode (trains only on TRAINER_UID data)
CUDA_VISIBLE_DEVICES=6 nohup grail -vv train --test-mode > train.log 2>&1 &

# Wait 240 seconds for trainer to initialize
sleep 320

# Start miner on GPU 7
CUDA_VISIBLE_DEVICES=7 nohup grail -vv mine > mine.log 2>&1 &

# Start validator on GPU 5 in test mode (validates only TRAINER_UID)
CUDA_VISIBLE_DEVICES=5 nohup grail -vv validate --test-mode > validate.log 2>&1 &

