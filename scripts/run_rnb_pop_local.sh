#!/bin/bash
# ============================================================
# R&B-POP: Full 2-stage pipeline (local / multi-GPU version)
# ============================================================
# Runs the full pipeline on a single machine with multiple GPUs.
# Set ngpus to the number of GPUs available on your machine.
# ============================================================

# ==================== USER CONFIGURATION ====================
ngpus=2   # number of GPUs to use for training

# root_dir="/path/to/v2v4real"
root_dir="/fs/scratch/PAS2099/dataset/v2v4real"

save_dir="exp"

# Initial reference car pseudo-labels
# To use the preprocessed labels included in the release:
#   tar -xzf exp/refcar_predictions_preprocessed.tar.gz -C exp/
# then set initial_label_path="exp/refcar_predictions_preprocessed"

initial_label_path="exp/refcar_predictions_preprocessed"
# initial_label_path="${root_dir}/new_postprocessed_refcar_pred_pointpillar_32beam"

ranker_path="pretrained_models/ranker.pth"
# ranker_path="/path/to/ranker_checkpoint.pth"

num_stages=2
declare -a max_distance=(40 90)
num_epochs_per_stage=60
warmup_epochs=6
batch_size=8
lr=0.002
warmup_lr=2e-4
lr_min=2e-5

sampling_method="coarse_to_fine"
num_samples=512
ranker_score_threshold=0.5
fixed_score_threshold=0.2
a=1
b=0
c=0
# ============================================================

set -e
set -x

for ((stage = 1; stage <= num_stages; stage++)); do

    echo "=== Stage ${stage} ==="

    echo "[Stage ${stage}] Refining labels..."
    python -u tools/refine_labels.py \
        --no_dist \
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

    echo "[Stage ${stage}] Filtering labels..."
    python -u tools/filter_labels.py \
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

    echo "[Stage ${stage}] Evaluating label quality..."
    python -u eval_label_quality.py \
        --root_dir ${root_dir} \
        --pseudo_label_path ${save_dir}/stage_${stage}_2_filtered \
        --pseudo_label_idx pred

    echo "[Stage ${stage}] Training detector..."
    torchrun --nproc_per_node=${ngpus} train.py \
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

    if [[ ${stage} -ne ${num_stages} ]]; then
        echo "[Stage ${stage}] Generating pseudo-labels for next stage..."
        python -u test.py \
            --model_dir ${save_dir}/checkpoints \
            --strict_model_path ${model_path} \
            --data_split train \
            --fixed_score_threshold ${fixed_score_threshold} \
            --use_dynamic_score_threshold --a ${a} --b ${b} --c ${c} \
            --save_npy --save_npy_n 5000 \
            --save_path ${save_dir}/stage_${stage}_3_trained

        initial_label_path="${save_dir}/stage_${stage}_3_trained/npy"
    fi

    echo "[Stage ${stage}] Evaluating on test set..."
    python -u test.py \
        --model_dir ${save_dir}/checkpoints \
        --strict_model_path ${model_path} \
        --data_split test

    echo "=== Stage ${stage} done ==="

done

echo "R&B-POP pipeline complete."
