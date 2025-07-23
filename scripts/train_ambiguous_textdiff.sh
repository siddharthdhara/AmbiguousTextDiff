#!/bin/bash

# Set up paths and hyperparameters
DATA_DIR="/workspace/AmbiguousTextDiff/data/training"
TEXT_ANNOTATIONS="/workspace/AmbiguousTextDiff/data/training/lidc_annotations_text_augmented_train.csv"
OUTPUT_DIR="results/ambiguous_textdiff"

# Hyperparameters
TOTAL_STEPS=50000 
BATCH_SIZE=4
LR=1e-4
SAVE_INTERVAL=5000 # Save every 5000 steps

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training script
python scripts/train_ambiguous_textdiff.py \
    --data_dir $DATA_DIR \
    --text_annotations_file $TEXT_ANNOTATIONS \
    --log_dir $OUTPUT_DIR \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --diffusion_steps 1000 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 128 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --image_size 128 \
    --image_actual_channels 1 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_interval 5000 \
    --lr_anneal_steps $TOTAL_STEPS \
    --freeze_text_encoder True \
    --text_model_name "emilyalsentzer/Bio_ClinicalBERT" \
    --max_text_length 128 \
    --text_dim 768