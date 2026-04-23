#!/bin/bash
# ============================================================
# Alignment Module Ablation Test Script
# ============================================================
# HYPIR is always the baseline, RM and alignment are ablated.
#
# Modes:
#   baseline  : HYPIR only (no RM, no alignment)   - control
#   rm        : RM + HYPIR (no alignment)           - RM contribution
#   align     : HYPIR + alignment (no RM)            - alignment contribution
#   full      : RM + alignment + HYPIR               - complete pipeline
#
# Usage:
#   bash test_alignment.sh                   # run all modes
#   bash test_alignment.sh baseline          # run single mode
#   bash test_alignment.sh full <input_dir>  # run with custom input
# ============================================================

set -e

# ---- Configuration ----
BASE_MODEL_PATH="checkpoints/sd2"
HYPIR_WEIGHT_PATH="checkpoints/HYPIR/HYPIR_sd2.pth"
ALIGNMENT_WEIGHT_PATH="output_alignment/checkpoint-5000/state_dict.pth"
RM_WEIGHT_PATH="checkpoints/DiffBIR/scunet_color_real_psnr.pth"
RM_TASK="bid"          # bid | bfr | bsr
UPSCALE=4
PATCH_SIZE=512
STRIDE=256
SEED=231
DEVICE="cuda"
CAPTIONER="empty"

INPUT_DIR="${2:-examples/lq}"
OUTPUT_BASE="results/ablation_$(date +%Y%m%d_%H%M%S)"

LORA_MODULES_LIST=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
IFS=','
LORA_MODULES="${LORA_MODULES_LIST[*]}"
unset IFS

# ---- Helper ----
run_test() {
    local mode=$1
    local output_dir="${OUTPUT_BASE}/${mode}"

    echo ""
    echo "============================================================"
    echo " Running mode: ${mode}"
    echo " Output: ${output_dir}"
    echo "============================================================"

    python test_alignment.py \
        --mode "${mode}" \
        --input "${INPUT_DIR}" \
        --output "${output_dir}" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --hypir_weight_path "${HYPIR_WEIGHT_PATH}" \
        --alignment_weight_path "${ALIGNMENT_WEIGHT_PATH}" \
        --rm_weight_path "${RM_WEIGHT_PATH}" \
        --rm_task "${RM_TASK}" \
        --lora_modules "${LORA_MODULES}" \
        --lora_rank 256 \
        --model_t 200 \
        --coeff_t 200 \
        --upscale "${UPSCALE}" \
        --patch_size "${PATCH_SIZE}" \
        --stride "${STRIDE}" \
        --scale_by factor \
        --captioner "${CAPTIONER}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --save_intermediate
}

# ---- Run ----
MODE="${1:-all}"

case "${MODE}" in
    all)
        echo "Running full ablation study..."
        echo "  Input: ${INPUT_DIR}"
        echo "  Output base: ${OUTPUT_BASE}"
        run_test baseline
        run_test rm
        run_test align
        run_test full
        ;;
    baseline|rm|align|full)
        run_test "${MODE}"
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: bash test_alignment.sh [all|baseline|rm|align|full] [input_dir]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " Ablation test completed"
echo " Results: ${OUTPUT_BASE}/"
echo ""
echo " Modes:"
echo "   baseline/  - HYPIR only (control)"
echo "   rm/        - RM + HYPIR"
echo "   align/     - HYPIR + alignment"
echo "   full/      - RM + alignment + HYPIR"
echo ""
echo " To evaluate metrics:"
echo "   bash scripts/calRes.sh <hq_dir> ${OUTPUT_BASE}/<mode>/result"
echo "============================================================"
