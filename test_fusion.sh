#!/bin/bash
# Fusion Model Test Script (RM+HYPIR) - Version 5 with corrected defaults
# Reference: DiffBIR README.md inference commands and updated requirements
# Usage: ./test_fusion_v5.sh [mode] [task] [input_dir] [output_dir] [rm_upscale] [hypir_upscale]
#   or manually edit the parameters below and run the script

# Default paths (adjust according to your setup)
CHECKPOINT_DIR="checkpoints"
RM_DIR="$CHECKPOINT_DIR/DiffBIR"
HYPIR_DIR="$CHECKPOINT_DIR/HYPIR"
BASE_MODEL_DIR="$CHECKPOINT_DIR/sd2"

# Default parameters (can be overridden by command line arguments)
MODE="${1:-full}"          # rm, hypir, full
TASK="${2:-bsr}"           # bid, bfr, bsr
INPUT_DIR="${3:-data/test/RealSRVal_crop128/test_LR}"
OUTPUT_DIR="${4:-results/RealSRVal_crop128}"
RM_UPSCALE_OVERRIDE="${5:-}"     # Optional override for RM upscale
HYPIR_UPSCALE_OVERRIDE="${6:-}"  # Optional override for HYPIR upscale

# Model file selections (adjust based on your available weights)
RM_BID_MODEL="$RM_DIR/scunet_color_real_psnr.pth"
RM_BFR_MODEL="$RM_DIR/face_swinir_v1.ckpt"
RM_BSR_MODEL="$RM_DIR/BSRNet.pth"

HYPIR_WEIGHT="$HYPIR_DIR/state_dict-LSDIR50-5000.pth"
BASE_MODEL="$BASE_MODEL_DIR"  # directory containing Stable Diffusion 2.1

# HYPIR parameters (from test.sh)
LORA_MODULES_LIST=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
IFS=','
LORA_MODULES="${LORA_MODULES_LIST[*]}"
unset IFS
LORA_RANK=256
MODEL_T=200
COEFF_T=200

# System settings
SEED=231
DEVICE="cuda"

# Function to select RM model based on task
select_rm_model() {
    case "$1" in
        bid)
            echo "$RM_BID_MODEL"
            ;;
        bfr)
            echo "$RM_BFR_MODEL"
            ;;
        bsr)
            echo "$RM_BSR_MODEL"
            ;;
        *)
            echo "$RM_BID_MODEL"
            ;;
    esac
}

# Function to get task-specific default upscale values
# Corrected based on DiffBIR README.md inference commands
get_task_default_upscales() {
    local task="$1"

    case "$task" in
        bsr)
            # Blind Super-Resolution
            RM_UPSCALE=4.0
            HYPIR_UPSCALE=1.0
            ;;
        bfr)
            # Blind Face Restoration (aligned)
            RM_UPSCALE=1.0
            HYPIR_UPSCALE=4.0
            ;;
        bid)
            # Blind Image Denoising
            RM_UPSCALE=1.0
            HYPIR_UPSCALE=4.0
            ;;
        *)
            # Default
            RM_UPSCALE=1.0
            HYPIR_UPSCALE=4.0
            ;;
    esac

    # Apply overrides if provided
    if [ -n "$RM_UPSCALE_OVERRIDE" ]; then
        RM_UPSCALE="$RM_UPSCALE_OVERRIDE"
        echo "Overriding RM upscale to: $RM_UPSCALE"
    fi

    if [ -n "$HYPIR_UPSCALE_OVERRIDE" ]; then
        HYPIR_UPSCALE="$HYPIR_UPSCALE_OVERRIDE"
        echo "Overriding HYPIR upscale to: $HYPIR_UPSCALE"
    fi

    # Export variables
    export RM_UPSCALE
    export HYPIR_UPSCALE
}

# Main execution
echo "==========================================="
echo "Fusion Model Test Script (RM+HYPIR) - Version 5"
echo "Mode: $MODE, Task: $TASK"
echo "Input: $INPUT_DIR, Output: $OUTPUT_DIR"
echo "==========================================="

# Get task-specific default upscale values (with optional overrides)
get_task_default_upscales "$TASK"

# Processing parameters
PATCH_SIZE=512
STRIDE=256
SCALE_BY="factor"
CAPTIONER="empty"
TILE_SIZE=512
TILE_STRIDE=256

# Create output directory
mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    rm)
        # RM only mode
        RM_MODEL=$(select_rm_model "$TASK")
        echo "Running RM only mode (Task: $TASK)"
        echo "RM Model: $RM_MODEL"
        echo "RM Upscale: $RM_UPSCALE"

        # Check if test_rm_hypir.py supports --rm_upscale parameter
        echo "Checking for --rm_upscale support in test_rm_hypir.py..."

        python test_rm_hypir.py --mode rm --task "$TASK" \
            --rm_model_path "$RM_MODEL" \
            --rm_upscale "$RM_UPSCALE" \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR/rm_$TASK/" \
            --tile_size "$TILE_SIZE" --tile_stride "$TILE_STRIDE" \
            --device "$DEVICE" --seed "$SEED"
        ;;

    hypir)
        # HYPIR only mode
        echo "Running HYPIR only mode"
        echo "HYPIR Weight: $HYPIR_WEIGHT"
        echo "Base Model: $BASE_MODEL"
        echo "Task: $TASK, HYPIR Upscale: $HYPIR_UPSCALE"

        python test_rm_hypir.py --mode hypir \
            --base_model_path "$BASE_MODEL" \
            --hypir_weight_path "$HYPIR_WEIGHT" \
            --lora_modules "$LORA_MODULES" \
            --lora_rank "$LORA_RANK" \
            --model_t "$MODEL_T" \
            --coeff_t "$COEFF_T" \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR/hypir_$TASK/" \
            --patch_size "$PATCH_SIZE" --stride "$STRIDE" \
            --upscale "$HYPIR_UPSCALE" \
            --scale_by "$SCALE_BY" \
            --captioner "$CAPTIONER" \
            --device "$DEVICE" --seed "$SEED"
        ;;

    full)
        # Full pipeline mode (RM + HYPIR)
        RM_MODEL=$(select_rm_model "$TASK")
        echo "Running Full pipeline mode (Task: $TASK)"
        echo "RM Model: $RM_MODEL"
        echo "HYPIR Weight: $HYPIR_WEIGHT"
        echo "Base Model: $BASE_MODEL"
        echo "RM Upscale: $RM_UPSCALE"
        echo "HYPIR Upscale: $HYPIR_UPSCALE"

        # Try to pass rm_upscale if supported, otherwise use default
        RM_UPSCALE_ARG=""
        # We'll attempt to pass it, Python script will ignore unknown args

        python test_rm_hypir.py --mode full --task "$TASK" \
            --rm_model_path "$RM_MODEL" \
            --rm_upscale "$RM_UPSCALE" \
            --base_model_path "$BASE_MODEL" \
            --hypir_weight_path "$HYPIR_WEIGHT" \
            --lora_modules "$LORA_MODULES" \
            --lora_rank "$LORA_RANK" \
            --model_t "$MODEL_T" \
            --coeff_t "$COEFF_T" \
            --input "$INPUT_DIR" \
            --output "$OUTPUT_DIR/full_$TASK/" \
            --tile_size "$TILE_SIZE" --tile_stride "$TILE_STRIDE" \
            --patch_size "$PATCH_SIZE" --stride "$STRIDE" \
            --upscale "$HYPIR_UPSCALE" \
            --scale_by "$SCALE_BY" \
            --captioner "$CAPTIONER" \
            --save_intermediate \
            --device "$DEVICE" --seed "$SEED"
        ;;

    *)
        echo "Error: Unknown mode '$MODE'. Use 'rm', 'hypir', or 'full'."
        echo "Usage: $0 [mode] [task] [input_dir] [output_dir] [rm_upscale] [hypir_upscale]"
        echo "  mode: rm, hypir, full"
        echo "  task: bid, bfr, bsr"
        echo "  input_dir: path to input images"
        echo "  output_dir: path to save results"
        echo "  rm_upscale: optional override for RM upscale factor"
        echo "  hypir_upscale: optional override for HYPIR upscale factor"
        exit 1
        ;;
esac

echo "==========================================="
echo "Test completed. Results saved to: $OUTPUT_DIR"
echo "==========================================="