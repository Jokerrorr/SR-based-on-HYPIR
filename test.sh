LORA_MODULES_LIST=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
IFS=','
LORA_MODULES="${LORA_MODULES_LIST[*]}"
unset IFS

python test.py \
--base_model_type sd2 \
--base_model_path checkpoints/sd2 \
--model_t 200 \
--coeff_t 200 \
--lora_rank 256 \
--lora_modules $LORA_MODULES \
--weight_path checkpoints/HYPIR/HYPIR_sd2.pth \
--patch_size 512 \
--stride 256 \
--lq_dir  examples/lq \
--scale_by factor \
--upscale 4 \
--captioner empty \
--output_dir results/examples \
--seed 231 \
--device cuda