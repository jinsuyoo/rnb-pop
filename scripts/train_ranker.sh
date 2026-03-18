#!/bin/bash
# ============================================================
# Train the PointNet-based box ranker
# ============================================================
# Before running this script, generate ranker training data with:
#   python ranker/generate_data/generate_ranker_data.py \
#       --root_dir /path/to/v2v4real \
#       --num_annotate_frames 8 \
#       --save_dir exp/ranker_training_data
# ============================================================

# ==================== USER CONFIGURATION ====================
root_dir="/path/to/v2v4real"
train_data_dir="exp/ranker_training_data"
save_dir="ranker_experiment"            # output directory (under ./results/)

num_annotate_frames=2   # must match value used in generate_ranker_data.py
batch_size=256
epoch=100
learning_rate=0.001
# ============================================================

set -e
set -x

python -u ranker/train_ranker.py \
    --root_dir ${root_dir} \
    --train_data_dir ${train_data_dir} \
    --num_annotate_frames ${num_annotate_frames} \
    --batch_size ${batch_size} \
    --epoch ${epoch} \
    --learning_rate ${learning_rate} \
    --save_dir ${save_dir} \
    --use_offset \
    --random_drop_points \
    --no_dist

echo "Ranker training complete. Checkpoints saved to: ./results/${save_dir}/checkpoints/"
