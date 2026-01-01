## TRL GSM8K Quickstart

### 1. Launch the vLLM server
```
source tools/vllm-server/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

### 2. Start GRPO training
```
CUDA_VISIBLE_DEVICES=7 nohup python scripts/train_trl_gsm8k.py > scripts/train_trl_gsm8k.log 2>&1 &
```

Training logs stream to `scripts/train_trl_gsm8k.log`.
