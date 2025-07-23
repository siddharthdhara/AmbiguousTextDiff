#!/bin/bash

# Directory where your generated samples are saved
SAMPLES_DIR="results/ambiguous_textdiff/samples_50k_16s_500img"

# Directory where your ground truth test data is
GT_DIR="/workspace/AmbiguousTextDiff/data/testing"

# The output CSV file for the metrics
RESULTS_FILE="${SAMPLES_DIR}/evaluation_metrics_500img_16s.csv"

echo "Evaluating samples in: ${SAMPLES_DIR}"
echo "Comparing against GTs in: ${GT_DIR}"

python scripts/evaluate_ambiguous_textdiff.py \
    --samples_dir $SAMPLES_DIR \
    --gt_dir $GT_DIR \
    --results_file $RESULTS_FILE