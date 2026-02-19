#!/bin/bash
# VeRL GRPO Training Runner Script
# This script provides easy-to-use commands for training with VeRL

set -e

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
DATASET="${DATASET:-gsm8k}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-$HOME/data}"
SEED="${SEED:-42}"

# Training hyperparameters (matching TRL GRPO config)
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-256}"
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
LR="${LR:-3e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
TOTAL_STEPS="${TOTAL_STEPS:-400}"
ROLLOUTS="${ROLLOUTS:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

# GRPO/DAPO specific
CLIP_LOW="${CLIP_LOW:-0.2}"
CLIP_HIGH="${CLIP_HIGH:-0.28}"
KL_COEF="${KL_COEF:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"

# ════════════════════════════════════════════════════════════════════════════
# HELP TEXT
# ════════════════════════════════════════════════════════════════════════════
show_help() {
    cat << EOF
VeRL GRPO Training Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    prepare     Prepare dataset (generate parquet files)
    train       Run VeRL training
    all         Prepare data and train (default)

Environment Variables:
    DATASET             Dataset to use: gsm8k, math, mbpp (default: gsm8k)
    MODEL               Model ID (default: Qwen/Qwen2.5-1.5B-Instruct)
    DATA_DIR            Directory for data files (default: ~/data)
    SEED                Random seed (default: 42)

    BATCH_SIZE          Per-device batch size (default: 2)
    GRAD_ACCUM          Gradient accumulation steps (default: 256)
    LR                  Learning rate (default: 3e-6)
    WARMUP_STEPS        Warmup steps (default: 20)
    TOTAL_STEPS         Total training steps (default: 400)
    ROLLOUTS            Rollouts per problem (default: 16)

    CLIP_LOW            PPO clip lower bound (default: 0.2)
    CLIP_HIGH           DAPO clip upper bound (default: 0.28)
    KL_COEF             KL divergence coefficient (default: 0.0)
    TEMPERATURE         Sampling temperature (default: 0.7)

Examples:
    # Basic training on GSM8K
    $0 train

    # Train on MATH with different model
    MODEL=Qwen/Qwen2.5-7B-Instruct DATASET=math $0 train

    # Just prepare data
    DATASET=mbpp $0 prepare

    # Full pipeline with custom settings
    BATCH_SIZE=4 LR=1e-6 TOTAL_STEPS=200 $0 all

EOF
}

# ════════════════════════════════════════════════════════════════════════════
# PREPARE DATA
# ════════════════════════════════════════════════════════════════════════════
prepare_data() {
    echo "════════════════════════════════════════════════════════════════════"
    echo "Preparing $DATASET dataset..."
    echo "════════════════════════════════════════════════════════════════════"

    cd "$PROJECT_ROOT"
    python research/verl/train_verl_grpo.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --prepare-data-only \
        --generate-config

    echo "Data preparation complete!"
}

# ════════════════════════════════════════════════════════════════════════════
# RUN TRAINING
# ════════════════════════════════════════════════════════════════════════════
run_training() {
    echo "════════════════════════════════════════════════════════════════════"
    echo "Starting VeRL GRPO training..."
    echo "════════════════════════════════════════════════════════════════════"
    echo "Dataset: $DATASET"
    echo "Model: $MODEL"
    echo "Effective batch size: $EFFECTIVE_BATCH"
    echo "Rollouts per problem: $ROLLOUTS"
    echo "Learning rate: $LR"
    echo "Total steps: $TOTAL_STEPS"
    echo "════════════════════════════════════════════════════════════════════"

    TRAIN_FILE="$DATA_DIR/grail_$DATASET/train.parquet"
    VAL_FILE="$DATA_DIR/grail_$DATASET/test.parquet"
    REWARD_MODULE="$SCRIPT_DIR/reward_functions.py"

    # Check files exist
    if [[ ! -f "$TRAIN_FILE" ]]; then
        echo "Error: Training file not found: $TRAIN_FILE"
        echo "Run '$0 prepare' first to generate data files."
        exit 1
    fi

    cd "$PROJECT_ROOT"

    # Run VeRL training
    python -m verl.trainer.main_ppo \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$VAL_FILE" \
        data.train_batch_size=$EFFECTIVE_BATCH \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.shuffle=True \
        data.seed=$SEED \
        actor_rollout_ref.model.path="$MODEL" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$EFFECTIVE_BATCH \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$BATCH_SIZE \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
        actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.shuffle=True \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=$ROLLOUTS \
        actor_rollout_ref.rollout.temperature=$TEMPERATURE \
        actor_rollout_ref.rollout.top_p=$TOP_P \
        actor_rollout_ref.rollout.top_k=$TOP_K \
        actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.do_sample=True \
        actor_rollout_ref.optim.lr=$LR \
        actor_rollout_ref.optim.lr_warmup_steps=$WARMUP_STEPS \
        actor_rollout_ref.optim.lr_scheduler_type=constant \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.type=fixed \
        algorithm.kl_ctrl.kl_coef=$KL_COEF \
        reward_model.enable=False \
        trainer.total_epochs=$TOTAL_STEPS \
        trainer.project_name=grail-verl \
        trainer.experiment_name="${DATASET}_grpo_dapo" \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=5 \
        trainer.save_freq=50 \
        trainer.test_freq=40 \
        trainer.val_before_train=True \
        +custom_reward_function.path="$REWARD_MODULE" \
        +custom_reward_function.name=compute_score
}

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
COMMAND="${1:-all}"

case "$COMMAND" in
    prepare)
        prepare_data
        ;;
    train)
        run_training
        ;;
    all)
        prepare_data
        run_training
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
