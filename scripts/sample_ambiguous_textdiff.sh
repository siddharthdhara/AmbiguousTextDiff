#!/bin/bash

# Set up paths and hyperparameters

MODEL_PATH="results/ambiguous_textdiff/model050000.pt"
DATA_DIR="/workspace/AmbiguousTextDiff/data/testing"
TEST_LIST="${DATA_DIR}/test_list_500.txt"
OUTPUT_DIR="results/ambiguous_textdiff/samples_50k_16s_500img"
TEXT_ANNOTATIONS="/workspace/AmbiguousTextDiff/data/testing/lidc_annotations_text_augmented_test.csv"
NUM_SAMPLES=16

# Create output directory
mkdir -p $OUTPUT_DIR

# Run sampling script
python scripts/sample_ambiguous_textdiff.py \
    --model_path $MODEL_PATH \
    --test_list $TEST_LIST \
    --output_dir $OUTPUT_DIR \
    --text_annotations_file $TEXT_ANNOTATIONS \
    --num_samples $NUM_SAMPLES \
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
    --clip_denoised True \
    --threshold 0.5 \
    --save_ensemble True