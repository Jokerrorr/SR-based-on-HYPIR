#!/bin/bash
# ============================================================
# Alignment Module Ablation Test Script
# ============================================================
# HYPIR is always the baseline, RM and alignment are ablated.
#
# Modes:
#   baseline  : HYPIR only (no RM, no alignment)   - control
#   rm        : RM + HYPIR (no alignment)           - RM contribution
#   full      : RM + alignment + HYPIR               - complete pipeline
#
# Usage:
#   bash test_alignment.sh full bid
#   bash test_alignment.sh baseline bsr data/test/RealSRVal_crop128/test_LR
#   bash test_alignment.sh full bsr data/test/RealSRVal_crop128/test_LR results/RealSRVal_crop128 4.0 1.0
# ============================================================

set -e

# ---- Arguments ----
# Usage: test_alignment.sh [mode] [task] [input_dir] [output_dir] [rm_upscale] [hypir_upscale]
MODE="${1:-full}"
TASK="${2:-bid}"
INPUT_DIR="${3:-data/test/RealSRVal_crop128/test_LR}"
OUTPUT_DIR="${4:-results/RealSRVal_crop128}"
RM_UPSCALE_OVERRIDE="${5:-1.0}"
HYPIR_UPSCALE_OVERRIDE="${6:-4.0}"

# ---- Model paths ----
BASE_MODEL_PATH="checkpoints/sd2"
HYPIR_WEIGHT_PATH="checkpoints/HYPIR/HYPIR_sd2.pth"
ALIGNMENT_WEIGHT_PATH="checkpoints/fusion/state_dict-LSDIR50-ALIGN10K-HYPIR5K.pth"

RM_BID_MODEL="checkpoints/DiffBIR/scunet_color_real_psnr.pth"
RM_BFR_MODEL="checkpoints/DiffBIR/face_swinir_v1.ckpt"
RM_BSR_MODEL="checkpoints/DiffBIR/BSRNet.pth"

# ---- HYPIR parameters ----
LORA_MODULES_LIST=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
IFS=','
LORA_MODULES="${LORA_MODULES_LIST[*]}"
unset IFS
LORA_RANK=256
MODEL_T=200
COEFF_T=200

# ---- Processing parameters ----
PATCH_SIZE=512
STRIDE=256
TILE_SIZE=512
TILE_STRIDE=256
SCALE_BY="factor"
CAPTIONER="empty"
SEED=231
DEVICE="cuda"

# ---- Helper functions ----
select_rm_model() {
    case "$1" in
        bid) echo "$RM_BID_MODEL" ;;
        bfr) echo "$RM_BFR_MODEL" ;;
        bsr) echo "$RM_BSR_MODEL" ;;
        *)   echo "$RM_BID_MODEL" ;;
    esac
}

get_task_upscales() {
    case "$TASK" in
        bsr)
            RM_UPSCALE=${RM_UPSCALE_OVERRIDE:-4.0}
            HYPIR_UPSCALE=${HYPIR_UPSCALE_OVERRIDE:-1.0}
            ;;
        *)
            RM_UPSCALE=${RM_UPSCALE_OVERRIDE:-1.0}
            HYPIR_UPSCALE=${HYPIR_UPSCALE_OVERRIDE:-4.0}
            ;;
    esac
}

# ---- Main ----
get_task_upscales

echo "==========================================="
echo "Alignment Ablation Test"
echo "Mode: $MODE | Task: $TASK"
echo "RM upscale: $RM_UPSCALE | HYPIR upscale: $HYPIR_UPSCALE"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "==========================================="

mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    baseline)
        echo "Running baseline (HYPIR only)..."
        python test_alignment.py \
            --mode baseline \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR/baseline_$TASK/" \
            --base_model_path "$BASE_MODEL_PATH" \
            --hypir_weight_path "$HYPIR_WEIGHT_PATH" \
            --rm_task "$TASK" \
            --rm_upscale "$RM_UPSCALE" \
            --hypir_upscale "$HYPIR_UPSCALE" \
            --lora_modules "$LORA_MODULES" \
            --lora_rank "$LORA_RANK" \
            --model_t "$MODEL_T" \
            --coeff_t "$COEFF_T" \
            --patch_size "$PATCH_SIZE" \
            --stride "$STRIDE" \
            --scale_by "$SCALE_BY" \
            --captioner "$CAPTIONER" \
            --seed "$SEED" \
            --device "$DEVICE"
        ;;
    rm)
        RM_MODEL=$(select_rm_model "$TASK")
        echo "Running RM + HYPIR (no alignment)..."
        echo "RM model: $RM_MODEL"
        python test_alignment.py \
            --mode rm \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR/rm_$TASK/" \
            --base_model_path "$BASE_MODEL_PATH" \
            --hypir_weight_path "$HYPIR_WEIGHT_PATH" \
            --rm_weight_path "$RM_MODEL" \
            --rm_task "$TASK" \
            --rm_upscale "$RM_UPSCALE" \
            --hypir_upscale "$HYPIR_UPSCALE" \
            --lora_modules "$LORA_MODULES" \
            --lora_rank "$LORA_RANK" \
            --model_t "$MODEL_T" \
            --coeff_t "$COEFF_T" \
            --tile_size "$TILE_SIZE" \
            --tile_stride "$TILE_STRIDE" \
            --patch_size "$PATCH_SIZE" \
            --stride "$STRIDE" \
            --scale_by "$SCALE_BY" \
            --captioner "$CAPTIONER" \
            --save_intermediate \
            --seed "$SEED" \
            --device "$DEVICE"
        ;;
    full)
        RM_MODEL=$(select_rm_model "$TASK")
        echo "Running FULL (RM + Alignment + HYPIR)..."
        echo "RM model: $RM_MODEL"
        echo "Alignment weights: $ALIGNMENT_WEIGHT_PATH"
        python test_alignment.py \
            --mode full \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR/full_$TASK/" \
            --base_model_path "$BASE_MODEL_PATH" \
            --hypir_weight_path "$HYPIR_WEIGHT_PATH" \
            --alignment_weight_path "$ALIGNMENT_WEIGHT_PATH" \
            --rm_weight_path "$RM_MODEL" \
            --rm_task "$TASK" \
            --rm_upscale "$RM_UPSCALE" \
            --hypir_upscale "$HYPIR_UPSCALE" \
            --lora_modules "$LORA_MODULES" \
            --lora_rank "$LORA_RANK" \
            --model_t "$MODEL_T" \
            --coeff_t "$COEFF_T" \
            --tile_size "$TILE_SIZE" \
            --tile_stride "$TILE_STRIDE" \
            --patch_size "$PATCH_SIZE" \
            --stride "$STRIDE" \
            --scale_by "$SCALE_BY" \
            --captioner "$CAPTIONER" \
            --save_intermediate \
            --seed "$SEED" \
            --device "$DEVICE"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Usage: $0 [mode] [task] [input_dir] [output_dir] [rm_upscale] [hypir_upscale]"
        echo "  mode: baseline, rm, full"
        echo "  task: bid, bfr, bsr"
        exit 1
        ;;
esac

echo "==========================================="
echo "Test completed. Results: $OUTPUT_DIR"
echo "==========================================="
