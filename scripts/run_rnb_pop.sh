#!/bin/bash
#SBATCH --job-name=rnb_pop
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --exclusive
#SBATCH --output=slurm_log/%j_%x.slurm.out

# ============================================================
# R&B-POP: Full 2-stage pipeline (SLURM version)
# ============================================================
# This script runs the full R&B-POP pipeline:
#   Stage 1: Rank & Build (close-range frames, 0-40m)
#   Stage 2: Self-training (all frames, 0-90m)
#
# Each stage:
#   1. Refine initial labels with the box ranker (C2F sampling)
#   2. Filter labels by ranker score
#   3. Train ego detector on pseudo-labels
#   4. (Between stages) Generate new pseudo-labels with trained detector
# ============================================================

module load cuda/11.8.0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rnb-pop

# ---------- Distributed training setup (SLURM) ----------
master_port=12345
export MASTER_PORT=${master_port}
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr}
dist_url="tcp://${master_addr}:${master_port}"

echo "Nodes: ${SLURM_JOB_NUM_NODES}, GPUs/node: ${SLURM_GPUS_PER_NODE}, dist_url: ${dist_url}"
# --------------------------------------------------------

# ==================== USER CONFIGURATION ====================
# Dataset root and output directories
root_dir="/path/to/v2v4real"
save_dir="exp"

# Initial reference car pseudo-labels
# To use the preprocessed labels included in the release:
#   tar -xzf exp/refcar_predictions_preprocessed.tar.gz -C exp/
# then set initial_label_path="exp/refcar_predictions_preprocessed"
initial_label_path="exp/refcar_predictions_preprocessed"

# Pretrained ranker checkpoint
ranker_path="/path/to/ranker_checkpoint.pth"

# Training hyperparameters
num_stages=2
declare -a max_distance=(40 90)   # distance curriculum per stage (meters)
num_epochs_per_stage=60
warmup_epochs=6
batch_size=8
lr=0.002
warmup_lr=2e-4
lr_min=2e-5

# Ranker & filtering settings
sampling_method="coarse_to_fine"
num_samples=512
ranker_score_threshold=0.5

# Dynamic score threshold for self-training (Tc + a/distance)
fixed_score_threshold=0.2
a=1
b=0
c=0
# ============================================================

set -e
set -x

for ((stage = 1; stage <= num_stages; stage++)); do

    echo "=== Stage ${stage} ==="

    # Step 1: Refine labels with box ranker + coarse-to-fine sampling
    echo "[Stage ${stage}] Refining labels..."
    srun python -u tools/refine_labels.py \
        --model_dir configs/rnb_pop_v2v4real.yaml \
        --model_path ${ranker_path} \
        --sampling_method ${sampling_method} \
        --num_samples ${num_samples} \
        --batch_size_ranker ${num_samples} \
        --initial_label_path ${initial_label_path} \
        --npy_label_idx pred.npy \
        --save_dir ${save_dir}/stage_${stage}_1_refined \
        --ranker_save_prefix ranker \
        --data_split train \
        --use_offset --adjust_with_estimated_offset

    # Step 2: Filter labels by ranker score
    echo "[Stage ${stage}] Filtering labels..."
    srun python -u tools/filter_labels.py \
        --root_dir ${root_dir} \
        --ranker_path ${ranker_path} \
        --combine_method current_only \
        --filtering_method threshold \
        --ranker_score_threshold ${ranker_score_threshold} \
        --prev_label_dir ${save_dir}/stage_${stage}_1_refined \
        --prev_label_prefix ranker_ranker \
        --cur_label_dir ${save_dir}/stage_${stage}_1_refined \
        --cur_label_prefix ranker_ranker \
        --save_dir ${save_dir}/stage_${stage}_2_filtered \
        --use_offset

    # Step 3: Evaluate label quality (informational)
    echo "[Stage ${stage}] Evaluating label quality..."
    srun python -u eval_label_quality.py \
        --root_dir ${root_dir} \
        --pseudo_label_path ${save_dir}/stage_${stage}_2_filtered \
        --pseudo_label_idx pred

    # Step 4: Train ego detector on filtered pseudo-labels
    echo "[Stage ${stage}] Training detector..."
    srun python -u train.py \
        --hypes_yaml configs/rnb_pop_v2v4real.yaml \
        --save_path ${save_dir}/checkpoints \
        --load_npy_label \
        --npy_label_path ${save_dir}/stage_${stage}_2_filtered \
        --npy_label_idx pred.npy \
        --n_epoches ${num_epochs_per_stage} \
        --batch_size ${batch_size} \
        --warmup_epoches ${warmup_epochs} \
        --lr ${lr} --warmup_lr ${warmup_lr} --lr_min ${lr_min} \
        --stage ${stage} \
        --distance_filtering \
        --min_distance 0 \
        --max_distance ${max_distance[$stage-1]}

    model_path="${save_dir}/checkpoints/stage_${stage}/net_epoch$(printf '%03d' ${num_epochs_per_stage}).pth"

    # Step 5: (Between stages) Generate new pseudo-labels with trained detector
    if [[ ${stage} -ne ${num_stages} ]]; then
        echo "[Stage ${stage}] Generating pseudo-labels for next stage..."
        srun python -u test.py \
            --model_dir ${save_dir}/checkpoints \
            --strict_model_path ${model_path} \
            --data_split train \
            --fixed_score_threshold ${fixed_score_threshold} \
            --use_dynamic_score_threshold --a ${a} --b ${b} --c ${c} \
            --save_npy --save_npy_n 5000 \
            --save_path ${save_dir}/stage_${stage}_3_trained

        initial_label_path="${save_dir}/stage_${stage}_3_trained/npy"
    fi

    # Step 6: Evaluate on test set
    echo "[Stage ${stage}] Evaluating on test set..."
    srun python -u test.py \
        --model_dir ${save_dir}/checkpoints \
        --strict_model_path ${model_path} \
        --data_split test

    echo "=== Stage ${stage} done ==="

done

echo "R&B-POP pipeline complete."
