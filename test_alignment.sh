# --- Default paths ---
CHECKPOINT_DIR="checkpoints"
RM_DIR="$CHECKPOINT_DIR/DiffBIR"
HYPIR_DIR="$CHECKPOINT_DIR/HYPIR"
BASE_MODEL_DIR="$CHECKPOINT_DIR/sd2"

# --- Default parameters (overridable via CLI) ---
MODE="${1:-full}"
TASK="${2:-bid}"
INPUT_DIR="${3:-data/test/DIV2K_V2_val/lq}"
OUTPUT_DIR="${4:-results/DIV2K_V2_val/BID}"

# --- Model files ---
RM_BID_MODEL="$RM_DIR/scunet_color_real_psnr.pth"
RM_BFR_MODEL="$RM_DIR/face_swinir_v1.ckpt"
RM_BSR_MODEL="$RM_DIR/BSRNet.pth"

HYPIR_WEIGHT="$HYPIR_DIR/HYPIR_sd2.pth"
ALIGNMENT_WEIGHT="$HYPIR_DIR/state_dict-LSDIR50-ALIGNMENT3000-LORA5000.pth"
BASE_MODEL="$BASE_MODEL_DIR"

# --- HYPIR parameters ---
LORA_MODULES="to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj"
LORA_RANK=256
MODEL_T=200
COEFF_T=200

# --- System ---
SEED=231
DEVICE="cuda"
PATCH_SIZE=512
STRIDE=256
SCALE_BY="factor"
CAPTIONER="empty"
TILE_SIZE=512
TILE_STRIDE=256

# --- Select RM model by task ---
select_rm_model() {
    case "$1" in
        bid) echo "$RM_BID_MODEL" ;;
        bfr) echo "$RM_BFR_MODEL" ;;
        bsr) echo "$RM_BSR_MODEL" ;;
        *)   echo "$RM_BID_MODEL" ;;
    esac
}

# --- Auto upscale per task ---
get_upscales() {
    case "$1" in
        bsr) RM_UP=4.0; HYPIR_UP=1.0 ;;
        bfr) RM_UP=1.0; HYPIR_UP=4.0 ;;
        bid) RM_UP=1.0; HYPIR_UP=4.0 ;;
        *)   RM_UP=1.0; HYPIR_UP=1.0 ;;
    esac
}
get_upscales "$TASK"

echo "==========================================="
echo "Alignment Pipeline Test"
echo "Mode: $MODE | Task: $TASK"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "==========================================="

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS="--input $INPUT_DIR --output $OUTPUT_DIR \
    --base_model_path $BASE_MODEL \
    --lora_modules $LORA_MODULES --lora_rank $LORA_RANK \
    --model_t $MODEL_T --coeff_t $COEFF_T \
    --patch_size $PATCH_SIZE --stride $STRIDE \
    --scale_by $SCALE_BY --captioner $CAPTIONER \
    --tile_size $TILE_SIZE --tile_stride $TILE_STRIDE \
    --device $DEVICE --seed $SEED"

case "$MODE" in
    hypir)
        echo ">> HYPIR baseline"
        python test_alignment_pipeline.py --mode hypir \
            $COMMON_ARGS \
            --hypir_weight_path "$HYPIR_WEIGHT" \
            --upscale "$HYPIR_UP"
        ;;

    rm_hypir)
        RM_MODEL=$(select_rm_model "$TASK")
        echo ">> RM + HYPIR (no alignment)"
        python test_alignment_pipeline.py --mode rm_hypir --task "$TASK" \
            $COMMON_ARGS \
            --rm_model_path "$RM_MODEL" \
            --hypir_weight_path "$HYPIR_WEIGHT" \
            --rm_upscale "$RM_UP" --upscale "$HYPIR_UP" \
            --save_intermediate
        ;;

    alignment)
        echo ">> Alignment-enhanced HYPIR (no RM)"
        python test_alignment_pipeline.py --mode alignment --task "$TASK" \
            $COMMON_ARGS \
            --alignment_weight_path "$ALIGNMENT_WEIGHT" \
            --upscale "$HYPIR_UP"
        ;;

    full)
        RM_MODEL=$(select_rm_model "$TASK")
        echo ">> Full Pipeline: RM + Alignment + HYPIR"
        python test_alignment_pipeline.py --mode full --task "$TASK" \
            $COMMON_ARGS \
            --rm_model_path "$RM_MODEL" \
            --alignment_weight_path "$ALIGNMENT_WEIGHT" \
            --rm_upscale "$RM_UP" --upscale "$HYPIR_UP" \
            --save_intermediate
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [hypir|rm_hypir|alignment|full] [bid|bfr|bsr] [input] [output]"
        exit 1
        ;;
esac

echo "==========================================="
echo "Done. Results in $OUTPUT_DIR"
echo "==========================================="
