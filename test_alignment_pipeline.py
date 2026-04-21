#!/usr/bin/env python3
"""
Alignment Pipeline Test Script (RM + Alignment + HYPIR)

Tests the full alignment-enhanced pipeline:
  LQ → RM(DiffBIR) → VAE Encoder(HYPIR) → Alignment(FaithDiff) → UNet(HYPIR) → VAE Decoder(HYPIR) → HQ

Supports four testing modes:
  1. hypir:      Test original HYPIR model (no RM, no alignment)
  2. rm_hypir:   Test RM + HYPIR two-stage pipeline (no alignment)
  3. alignment:  Test alignment-enhanced HYPIR (no RM, direct LQ input)
  4. full:       Test full RM + Alignment + HYPIR pipeline

Usage examples:
  # Mode 1: HYPIR only (baseline)
  python test_alignment_pipeline.py --mode hypir \
      --base_model_path checkpoints/sd2 \
      --hypir_weight_path checkpoints/HYPIR/HYPIR_sd2.pth \
      --input examples/lq/ --output results/hypir/

  # Mode 2: RM + HYPIR (no alignment)
  python test_alignment_pipeline.py --mode rm_hypir --task bsr \
      --rm_model_path checkpoints/DiffBIR/BSRNet.pth \
      --base_model_path checkpoints/sd2 \
      --hypir_weight_path checkpoints/HYPIR/HYPIR_sd2.pth \
      --input data/test/RealSRVal_crop128/test_LR/ --output results/rm_hypir/

  # Mode 3: Alignment-enhanced HYPIR (no RM)
  python test_alignment_pipeline.py --mode alignment \
      --base_model_path checkpoints/sd2 \
      --alignment_weight_path checkpoints/HYPIR/state_dict-LSDIR50-ALIGNMENT3000-LORA5000.pth \
      --input examples/lq/ --output results/alignment/

  # Mode 4: Full pipeline (RM + Alignment + HYPIR)
  python test_alignment_pipeline.py --mode full --task bsr \
      --rm_model_path checkpoints/DiffBIR/BSRNet.pth \
      --base_model_path checkpoints/sd2 \
      --alignment_weight_path checkpoints/HYPIR/state_dict-LSDIR50-ALIGNMENT3000-LORA5000.pth \
      --input data/test/RealSRVal_crop128/test_LR/ --output results/full_alignment/ \
      --save_intermediate
"""

import argparse
import os
import sys
from pathlib import Path
from time import time

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

# --- Module imports with graceful fallbacks ---

try:
    from HYPIR.rm.restoration_module import RestorationModule
    RM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RM module not available: {e}")
    RM_AVAILABLE = False

try:
    from HYPIR.enhancer.sd2 import SD2Enhancer
    from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner
    HYPIR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HYPIR enhancer not available: {e}")
    HYPIR_AVAILABLE = False

try:
    from HYPIR.enhancer.sd2_alignment import SD2AlignmentEnhancer
    ALIGNMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Alignment enhancer not available: {e}")
    ALIGNMENT_AVAILABLE = False


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Alignment Pipeline (RM + Alignment + HYPIR)"
    )

    # --- Mode ---
    parser.add_argument("--mode", type=str, required=True,
                        choices=["hypir", "rm_hypir", "alignment", "full"],
                        help="Test mode: "
                             "'hypir' (HYPIR baseline), "
                             "'rm_hypir' (RM+HYPIR, no alignment), "
                             "'alignment' (Alignment+HYPIR, no RM), "
                             "'full' (RM+Alignment+HYPIR)")

    # --- RM arguments ---
    parser.add_argument("--task", type=str, default="bid",
                        choices=["bid", "bfr", "bsr"],
                        help="RM task type")
    parser.add_argument("--rm_model_path", type=str,
                        help="Path to RM weight file")
    parser.add_argument("--rm_config_path", type=str,
                        help="Optional RM config file (YAML)")
    parser.add_argument("--rm_upscale", type=float, default=None,
                        help="RM upscale factor (auto-set per task if omitted)")

    # --- HYPIR base model arguments ---
    parser.add_argument("--base_model_path", type=str,
                        default="checkpoints/sd2",
                        help="Path to base SD model directory")
    parser.add_argument("--hypir_weight_path", type=str,
                        help="Path to HYPIR LoRA weight file (no alignment)")
    parser.add_argument("--alignment_weight_path", type=str,
                        help="Path to alignment-enhanced weight file (LoRA + alignment_handler)")
    parser.add_argument("--lora_modules", type=str,
                        default="to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
                        help="Comma-separated LoRA module names")
    parser.add_argument("--lora_rank", type=int, default=256,
                        help="LoRA rank")
    parser.add_argument("--model_t", type=int, default=200,
                        help="Model input timestep")
    parser.add_argument("--coeff_t", type=int, default=200,
                        help="Timestep for conversion coefficients")

    # --- Input / Output ---
    parser.add_argument("--input", type=str, required=True,
                        help="Input image directory or single image file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save RM output as intermediate result")

    # --- Processing parameters ---
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for RM tiled inference")
    parser.add_argument("--tile_stride", type=int, default=256,
                        help="Tile stride for RM tiled inference")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Patch size for HYPIR processing")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for HYPIR processing")
    parser.add_argument("--upscale", type=float, default=None,
                        help="HYPIR upscale factor (auto-set per task if omitted)")
    parser.add_argument("--scale_by", type=str, default="factor",
                        choices=["factor", "longest_side"],
                        help="Scaling method")
    parser.add_argument("--target_longest_side", type=int, default=None,
                        help="Target longest side for 'longest_side' scaling")
    parser.add_argument("--ensure_min_size", action="store_true",
                        help="Resize RM output to meet HYPIR minimum size (512x512)")

    # --- Prompt ---
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for HYPIR. If empty, captioner is used.")
    parser.add_argument("--txt_dir", type=str, default=None,
                        help="Directory with text prompts matching image names")
    parser.add_argument("--captioner", type=str, default="empty",
                        choices=["empty", "fixed"],
                        help="Captioner type")
    parser.add_argument("--fixed_caption", type=str, default=None,
                        help="Fixed caption if captioner='fixed'")

    # --- System ---
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=231,
                        help="Random seed")

    return parser.parse_args()


# ============================================================================
# Utility helpers
# ============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(input_path: str) -> list:
    """Find all image files in directory or return single file."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    images = []
    for root, _, files in os.walk(p):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / f)
    images.sort()
    return images


def get_default_upscales(task: str):
    """Return (rm_upscale, hypir_upscale) defaults for a given task."""
    defaults = {
        "bsr": (4.0, 1.0),   # BSR: RM does 4x upsample, HYPIR 1x
        "bfr": (1.0, 1.0),   # BFR: both 1x (face-aligned)
        "bid": (1.0, 1.0),   # BID: both 1x (denoising only)
    }
    return defaults.get(task, (1.0, 1.0))


def resolve_rel_path(img_path: Path, input_dir: str) -> Path:
    """Compute relative path from input directory for output layout."""
    src = Path(input_dir)
    if src.is_file():
        return Path(img_path.name)
    try:
        return img_path.relative_to(src)
    except ValueError:
        return Path(img_path.name)


def get_prompt(img_path: Path, args, captioner, to_tensor) -> str:
    """Resolve prompt for an image from txt_dir, --prompt flag, or captioner."""
    if args.txt_dir is not None:
        rel = resolve_rel_path(img_path, args.input)
        prompt_path = Path(args.txt_dir) / rel.with_suffix(".txt")
        if prompt_path.exists():
            with open(prompt_path, "r") as fp:
                return fp.read().strip()
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    if args.prompt:
        return args.prompt
    img = Image.open(img_path).convert("RGB")
    return captioner(img)


def select_rm_model(task: str, rm_model_path: str = None) -> str:
    """Select default RM model path for a task if not explicitly given."""
    if rm_model_path:
        return rm_model_path
    defaults = {
        "bid": "checkpoints/DiffBIR/scunet_color_real_psnr.pth",
        "bfr": "checkpoints/DiffBIR/face_swinir_v1.ckpt",
        "bsr": "checkpoints/DiffBIR/BSRNet.pth",
    }
    return defaults[task]


# ============================================================================
# Mode 1: HYPIR only (baseline)
# ============================================================================

def test_hypir_only(args, images, output_dir):
    if not HYPIR_AVAILABLE:
        raise ImportError("HYPIR modules not available.")

    weight_path = args.hypir_weight_path or args.alignment_weight_path
    if not weight_path:
        raise ValueError("--hypir_weight_path or --alignment_weight_path required for hypir mode.")

    print("=" * 60)
    print("Mode: HYPIR Only (baseline)")
    print(f"Base model: {args.base_model_path}")
    print(f"Weight:     {weight_path}")
    print(f"Images:     {len(images)}")
    print("=" * 60)

    model = SD2Enhancer(
        base_model_path=args.base_model_path,
        weight_path=weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    print("Loading models...")
    t0 = time()
    model.init_models()
    print(f"Models loaded in {time() - t0:.2f}s")

    if args.captioner == "empty":
        captioner = EmptyCaptioner(args.device)
    else:
        if not args.fixed_caption:
            raise ValueError("--fixed_caption required when --captioner=fixed")
        captioner = FixedCaptioner(args.device, args.fixed_caption)

    to_tensor = transforms.ToTensor()
    result_dir = output_dir / "result"
    prompt_dir = output_dir / "prompt"

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        rel = resolve_rel_path(img_path, args.input)

        img = Image.open(img_path).convert("RGB")
        lq_tensor = to_tensor(img).unsqueeze(0).to(args.device)
        prompt = get_prompt(img_path, args, captioner, to_tensor)

        prompt_path = prompt_dir / rel.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_path, "w") as fp:
            fp.write(prompt)

        t0 = time()
        with torch.no_grad():
            result = model.enhance(
                lq=lq_tensor,
                prompt=prompt,
                scale_by=args.scale_by,
                upscale=args.upscale,
                target_longest_side=args.target_longest_side,
                patch_size=args.patch_size,
                stride=args.stride,
                return_type="pil",
            )[0]
        dt = time() - t0

        out_path = result_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        print(f"  Saved: {out_path}  ({dt:.2f}s, {img.size} -> {result.size})")

    print(f"\nDone. Results in {result_dir}")


# ============================================================================
# Mode 2: RM + HYPIR (no alignment)
# ============================================================================

def test_rm_hypir(args, images, output_dir):
    if not RM_AVAILABLE or not HYPIR_AVAILABLE:
        raise ImportError("Both RM and HYPIR modules required.")

    rm_up = args.rm_upscale or get_default_upscales(args.task)[0]
    hypir_up = args.upscale or get_default_upscales(args.task)[1]
    rm_path = select_rm_model(args.task, args.rm_model_path)
    hypir_path = args.hypir_weight_path or args.alignment_weight_path
    if not hypir_path:
        raise ValueError("--hypir_weight_path or --alignment_weight_path required.")

    print("=" * 60)
    print("Mode: RM + HYPIR (no alignment)")
    print(f"Task:       {args.task}")
    print(f"RM model:   {rm_path}")
    print(f"RM upscale: {rm_up}")
    print(f"HYPIR wt:   {hypir_path}")
    print(f"HYPIR up:   {hypir_up}")
    print(f"Images:     {len(images)}")
    print("=" * 60)

    # Load RM
    print("Loading RM...")
    rm_model = RestorationModule(task=args.task, device=args.device)
    rm_model.load(weight_path=rm_path, config_path=args.rm_config_path)

    # Load HYPIR
    print("Loading HYPIR...")
    hypir_model = SD2Enhancer(
        base_model_path=args.base_model_path,
        weight_path=hypir_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    hypir_model.init_models()

    if args.captioner == "empty":
        captioner = EmptyCaptioner(args.device)
    else:
        if not args.fixed_caption:
            raise ValueError("--fixed_caption required when --captioner=fixed")
        captioner = FixedCaptioner(args.device, args.fixed_caption)

    result_dir = output_dir / "result"
    intermediate_dir = output_dir / "intermediate"

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        rel = resolve_rel_path(img_path, args.input)

        img = Image.open(img_path).convert("RGB")

        # Stage 1: RM
        t0 = time()
        rm_output = rm_model.process_image(
            image=img,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            ensure_min_size=args.ensure_min_size,
            upscale=rm_up,
        )
        rm_time = time() - t0
        print(f"  RM: {img.size} -> {rm_output.shape[2:]} ({rm_time:.2f}s)")

        if args.save_intermediate:
            int_path = intermediate_dir / rel.with_suffix(".png")
            int_path.parent.mkdir(parents=True, exist_ok=True)
            rm_model.save_output(rm_output, int_path)

        # Stage 2: HYPIR
        prompt = get_prompt(img_path, args, captioner, None)
        rm_tensor = rm_output.to(args.device)
        t1 = time()
        with torch.no_grad():
            result = hypir_model.enhance(
                lq=rm_tensor,
                prompt=prompt,
                scale_by=args.scale_by,
                upscale=hypir_up,
                target_longest_side=args.target_longest_side,
                patch_size=args.patch_size,
                stride=args.stride,
                return_type="pil",
            )[0]
        hypir_time = time() - t1
        print(f"  HYPIR: {rm_output.shape[2:]} -> {result.size} ({hypir_time:.2f}s)")

        out_path = result_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        print(f"  Saved: {out_path}  (total {rm_time + hypir_time:.2f}s)")

    print(f"\nDone. Results in {result_dir}")


# ============================================================================
# Mode 3: Alignment-enhanced HYPIR (no RM)
# ============================================================================

def test_alignment(args, images, output_dir):
    if not ALIGNMENT_AVAILABLE:
        raise ImportError("Alignment enhancer not available.")

    hypir_up = args.upscale or 1.0
    weight_path = args.alignment_weight_path
    if not weight_path:
        raise ValueError("--alignment_weight_path required for alignment mode.")

    print("=" * 60)
    print("Mode: Alignment-enhanced HYPIR (no RM)")
    print(f"Base model:      {args.base_model_path}")
    print(f"Alignment wt:    {weight_path}")
    print(f"HYPIR upscale:   {hypir_up}")
    print(f"Images:          {len(images)}")
    print("=" * 60)

    model = SD2AlignmentEnhancer(
        base_model_path=args.base_model_path,
        weight_path=weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    print("Loading models...")
    t0 = time()
    model.init_models()
    print(f"Models loaded in {time() - t0:.2f}s")

    if args.captioner == "empty":
        captioner = EmptyCaptioner(args.device)
    else:
        if not args.fixed_caption:
            raise ValueError("--fixed_caption required when --captioner=fixed")
        captioner = FixedCaptioner(args.device, args.fixed_caption)

    to_tensor = transforms.ToTensor()
    result_dir = output_dir / "result"

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        rel = resolve_rel_path(img_path, args.input)

        img = Image.open(img_path).convert("RGB")
        lq_tensor = to_tensor(img).unsqueeze(0).to(args.device)
        prompt = get_prompt(img_path, args, captioner, to_tensor)

        # For alignment mode (no RM), use LQ itself as the alignment reference
        t0 = time()
        with torch.no_grad():
            model.set_rm_output(lq_tensor)

            result = model.enhance(
                lq=lq_tensor,
                prompt=prompt,
                scale_by=args.scale_by,
                upscale=hypir_up,
                target_longest_side=args.target_longest_side,
                patch_size=args.patch_size,
                stride=args.stride,
                return_type="pil",
            )[0]
        dt = time() - t0

        out_path = result_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        print(f"  Saved: {out_path}  ({dt:.2f}s, {img.size} -> {result.size})")

    print(f"\nDone. Results in {result_dir}")


# ============================================================================
# Mode 4: Full pipeline (RM + Alignment + HYPIR)
# ============================================================================

def test_full(args, images, output_dir):
    if not RM_AVAILABLE or not ALIGNMENT_AVAILABLE:
        raise ImportError("Both RM and Alignment modules required.")

    rm_up = args.rm_upscale or get_default_upscales(args.task)[0]
    hypir_up = args.upscale or get_default_upscales(args.task)[1]
    rm_path = select_rm_model(args.task, args.rm_model_path)
    weight_path = args.alignment_weight_path
    if not weight_path:
        raise ValueError("--alignment_weight_path required for full mode.")

    print("=" * 60)
    print("Mode: Full Pipeline (RM + Alignment + HYPIR)")
    print(f"Task:            {args.task}")
    print(f"RM model:        {rm_path}")
    print(f"RM upscale:      {rm_up}")
    print(f"Alignment wt:    {weight_path}")
    print(f"HYPIR upscale:   {hypir_up}")
    print(f"Images:          {len(images)}")
    print("=" * 60)

    # Load RM
    print("Loading RM...")
    rm_model = RestorationModule(task=args.task, device=args.device)
    rm_model.load(weight_path=rm_path, config_path=args.rm_config_path)

    # Load alignment-enhanced HYPIR
    print("Loading Alignment-enhanced HYPIR...")
    alignment_model = SD2AlignmentEnhancer(
        base_model_path=args.base_model_path,
        weight_path=weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    alignment_model.init_models()
    print("All models loaded.")

    if args.captioner == "empty":
        captioner = EmptyCaptioner(args.device)
    else:
        if not args.fixed_caption:
            raise ValueError("--fixed_caption required when --captioner=fixed")
        captioner = FixedCaptioner(args.device, args.fixed_caption)

    result_dir = output_dir / "result"
    intermediate_dir = output_dir / "intermediate"

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        rel = resolve_rel_path(img_path, args.input)

        img = Image.open(img_path).convert("RGB")

        # --- Stage 1: RM degradation removal ---
        t_rm_start = time()
        rm_output = rm_model.process_image(
            image=img,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            ensure_min_size=args.ensure_min_size,
            upscale=rm_up,
        )
        t_rm = time() - t_rm_start
        print(f"  [RM]  {img.size} -> {rm_output.shape[2:]} ({t_rm:.2f}s)")

        if args.save_intermediate:
            int_path = intermediate_dir / rel.with_suffix(".png")
            int_path.parent.mkdir(parents=True, exist_ok=True)
            rm_model.save_output(rm_output, int_path)
            print(f"  Intermediate saved: {int_path}")

        # --- Stage 2: Alignment-enhanced HYPIR ---
        prompt = get_prompt(img_path, args, captioner, None)
        rm_tensor = rm_output.to(args.device)

        t_hypir_start = time()
        with torch.no_grad():
            # Pass RM pixel output — enhance() will VAE-encode it at the correct scale
            alignment_model.set_rm_output(rm_tensor)

            result = alignment_model.enhance(
                lq=rm_tensor,
                prompt=prompt,
                scale_by=args.scale_by,
                upscale=hypir_up,
                target_longest_side=args.target_longest_side,
                patch_size=args.patch_size,
                stride=args.stride,
                return_type="pil",
            )[0]
        t_hypir = time() - t_hypir_start
        print(f"  [Alignment+HYPIR] {rm_output.shape[2:]} -> {result.size} ({t_hypir:.2f}s)")

        out_path = result_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        print(f"  Saved: {out_path}  (total {t_rm + t_hypir:.2f}s)")

    print(f"\nDone. Results in {result_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- Validate mode requirements ---
    if args.mode in ("rm_hypir", "full"):
        if not RM_AVAILABLE:
            raise ImportError("RM module required for this mode.")
        if not args.rm_model_path and not os.path.exists(select_rm_model(args.task)):
            raise ValueError(f"--rm_model_path required (default for {args.task} not found)")

    if args.mode == "hypir":
        if not HYPIR_AVAILABLE:
            raise ImportError("HYPIR module required for this mode.")
        if not args.hypir_weight_path and not args.alignment_weight_path:
            raise ValueError("--hypir_weight_path or --alignment_weight_path required")

    if args.mode in ("alignment", "full"):
        if not ALIGNMENT_AVAILABLE:
            raise ImportError("Alignment module required for this mode.")
        if not args.alignment_weight_path:
            raise ValueError("--alignment_weight_path required for this mode")

    # --- Find images ---
    images = find_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        sys.exit(1)
    print(f"Found {len(images)} image(s)")

    # --- Auto-set upscale factors if not specified ---
    if args.upscale is None:
        args.upscale = get_default_upscales(args.task)[1]
    if args.rm_upscale is None:
        args.rm_upscale = get_default_upscales(args.task)[0]

    # --- Create output directory ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run selected mode ---
    try:
        mode_funcs = {
            "hypir": test_hypir_only,
            "rm_hypir": test_rm_hypir,
            "alignment": test_alignment,
            "full": test_full,
        }
        mode_funcs[args.mode](args, images, output_dir)

        print("\n" + "=" * 60)
        print(f"Test completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
