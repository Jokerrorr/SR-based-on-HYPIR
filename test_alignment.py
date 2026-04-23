#!/usr/bin/env python3
"""
Alignment Module Test Script for RM+HYPIR Pipeline

Ablation study modes (HYPIR is always the baseline):
1. baseline: HYPIR only (no RM, no alignment) - control group
2. rm: RM + HYPIR (no alignment)
3. align: HYPIR + alignment (no RM)
4. full: RM + alignment + HYPIR (complete pipeline)

Usage examples:
  # Single mode test
  python test_alignment.py --mode baseline --input examples/lq/ --output results/baseline/
  python test_alignment.py --mode full --input examples/lq/ --output results/full/

  # Run ablation study (all modes)
  bash test_alignment.sh
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

# Import modules
try:
    from HYPIR.rm.restoration_module import RestorationModule
    RM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RM modules not available: {e}")
    RM_AVAILABLE = False

try:
    from HYPIR.enhancer.sd2 import SD2Enhancer
    HYPIR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HYPIR modules not available: {e}")
    HYPIR_AVAILABLE = False

try:
    from HYPIR.enhancer.sd2_alignment import SD2AlignmentEnhancer
    ALIGNMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Alignment modules not available: {e}")
    ALIGNMENT_AVAILABLE = False

try:
    from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner
    CAPTIONER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Captioner modules not available: {e}")
    CAPTIONER_AVAILABLE = False


DEFAULT_LORA_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0",
    "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
    "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test Alignment Module - Ablation Study")

    # Mode selection: HYPIR baseline + RM/alignment ablation
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "rm", "align", "full"],
                        help="Test mode: baseline(HYPIR only), rm(RM+HYPIR), align(HYPIR+align), full(RM+align+HYPIR)")

    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input image directory or single image file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate results (RM output)")

    # Model paths
    parser.add_argument("--base_model_path", type=str, default="checkpoints/sd2",
                        help="Path to base Stable Diffusion model")
    parser.add_argument("--hypir_weight_path", type=str,
                        default="checkpoints/HYPIR/HYPIR_sd2.pth",
                        help="Path to HYPIR LoRA weight file")
    parser.add_argument("--alignment_weight_path", type=str,
                        default="output_alignment/checkpoint-5000/state_dict.pth",
                        help="Path to alignment weight file (Stage 2 output)")
    parser.add_argument("--rm_weight_path", type=str,
                        default="checkpoints/DiffBIR/scunet_color_real_psnr.pth",
                        help="Path to RM weight file")

    # RM settings
    parser.add_argument("--rm_task", type=str, default="bid",
                        choices=["bid", "bfr", "bsr"],
                        help="RM task type")

    # HYPIR settings
    parser.add_argument("--lora_modules", type=str, default=",".join(DEFAULT_LORA_MODULES),
                        help="Comma-separated LoRA module names")
    parser.add_argument("--lora_rank", type=int, default=256,
                        help="LoRA rank")
    parser.add_argument("--model_t", type=int, default=200,
                        help="Model input timestep")
    parser.add_argument("--coeff_t", type=int, default=200,
                        help="Timestep for conversion coefficients")

    # Processing parameters
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for RM tiled inference")
    parser.add_argument("--tile_stride", type=int, default=256,
                        help="Tile stride for RM tiled inference")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Patch size for HYPIR processing")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for HYPIR processing")
    parser.add_argument("--upscale", type=int, default=4,
                        help="Upscale factor")
    parser.add_argument("--scale_by", type=str, default="factor",
                        choices=["factor", "longest_side"],
                        help="Scaling method")
    parser.add_argument("--target_longest_side", type=int, default=None,
                        help="Target longest side for scaling")

    # Prompt settings
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt (empty for captioner)")
    parser.add_argument("--captioner", type=str, default="empty",
                        choices=["empty", "fixed"],
                        help="Captioner type")
    parser.add_argument("--fixed_caption", type=str, default=None,
                        help="Fixed caption if captioner is 'fixed'")
    parser.add_argument("--txt_dir", type=str, default=None,
                        help="Directory containing text prompts")

    # System settings
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--seed", type=int, default=231,
                        help="Random seed")

    return parser.parse_args()


def find_images(input_path: str) -> list:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                images.append(Path(root) / file)
    images.sort()
    return images


def get_captioner(args):
    if not CAPTIONER_AVAILABLE:
        return None
    if args.captioner == "empty":
        return EmptyCaptioner(args.device)
    elif args.captioner == "fixed":
        if not args.fixed_caption:
            raise ValueError("--fixed_caption required when --captioner=fixed")
        return FixedCaptioner(args.device, args.fixed_caption)
    raise ValueError(f"Unknown captioner: {args.captioner}")


def get_prompt(args, img_path, captioner, img):
    if args.txt_dir is not None:
        txt_dir = Path(args.txt_dir)
        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        prompt_path = txt_dir / rel_path.with_suffix(".txt")
        if prompt_path.exists():
            with open(prompt_path, "r") as fp:
                return fp.read().strip()
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    elif args.prompt:
        return args.prompt
    return captioner(img) if captioner else ""


def test_baseline(args, images, output_dir):
    """
    Mode: baseline - HYPIR only (no RM, no alignment)
    Control group for ablation study.
    """
    if not HYPIR_AVAILABLE:
        raise ImportError("HYPIR modules not available")

    print("=" * 60)
    print("Mode: BASELINE (HYPIR only, no RM, no alignment)")
    print(f"Base model: {args.base_model_path}")
    print(f"HYPIR weights: {args.hypir_weight_path}")
    print(f"Input: {len(images)} images")
    print("=" * 60)

    print("Loading HYPIR model...")
    model = SD2Enhancer(
        base_model_path=args.base_model_path,
        weight_path=args.hypir_weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    model.init_models()

    captioner = get_captioner(args)
    to_tensor = transforms.ToTensor()

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        img = Image.open(img_path).convert("RGB")
        lq_tensor = to_tensor(img).unsqueeze(0).to(args.device)
        prompt = get_prompt(args, img_path, captioner, img)

        print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")

        start_time = time()
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
        proc_time = time() - start_time

        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        output_path = output_dir / rel_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        print(f"  Saved: {output_path}")
        print(f"  Time: {proc_time:.2f}s | {img.size} -> {result.size}")


def test_rm(args, images, output_dir):
    """
    Mode: rm - RM + HYPIR (no alignment)
    Test RM contribution on top of HYPIR baseline.
    """
    if not RM_AVAILABLE or not HYPIR_AVAILABLE:
        raise ImportError("RM and HYPIR modules required")

    print("=" * 60)
    print("Mode: RM + HYPIR (no alignment)")
    print(f"RM task: {args.rm_task}")
    print(f"RM weights: {args.rm_weight_path}")
    print(f"HYPIR weights: {args.hypir_weight_path}")
    print(f"Input: {len(images)} images")
    print("=" * 60)

    print("Loading RM model...")
    rm_model = RestorationModule(task=args.rm_task, device=args.device)
    rm_model.load(weight_path=args.rm_weight_path)

    print("Loading HYPIR model...")
    hypir_model = SD2Enhancer(
        base_model_path=args.base_model_path,
        weight_path=args.hypir_weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    hypir_model.init_models()

    captioner = get_captioner(args)

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        img = Image.open(img_path).convert("RGB")

        # Stage 1: RM
        start_time = time()
        rm_output = rm_model.process_image(
            image=img,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            ensure_min_size=True,
        )
        rm_time = time() - start_time

        if args.save_intermediate:
            rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
            intermediate_path = output_dir / "intermediate" / rel_path.with_suffix(".png")
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)
            rm_model.save_output(rm_output, intermediate_path)
            print(f"  RM intermediate: {intermediate_path}")

        # Stage 2: HYPIR
        prompt = get_prompt(args, img_path, captioner, img)
        print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")

        start_time = time()
        with torch.no_grad():
            result = hypir_model.enhance(
                lq=rm_output,
                prompt=prompt,
                scale_by=args.scale_by,
                upscale=args.upscale,
                target_longest_side=args.target_longest_side,
                patch_size=args.patch_size,
                stride=args.stride,
                return_type="pil",
            )[0]
        hypir_time = time() - start_time

        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        output_path = output_dir / rel_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        print(f"  Saved: {output_path}")
        print(f"  RM: {rm_time:.2f}s | HYPIR: {hypir_time:.2f}s | Total: {rm_time + hypir_time:.2f}s")
        print(f"  {img.size} -> RM: {rm_output.shape[2:]} -> {result.size}")


def test_align(args, images, output_dir):
    """
    Mode: align - HYPIR + alignment (no RM)
    Test alignment contribution on top of HYPIR baseline.
    """
    if not ALIGNMENT_AVAILABLE:
        raise ImportError("Alignment modules not available")

    print("=" * 60)
    print("Mode: HYPIR + ALIGNMENT (no RM)")
    print(f"Base model: {args.base_model_path}")
    print(f"Alignment weights: {args.alignment_weight_path}")
    print(f"Input: {len(images)} images")
    print("=" * 60)

    print("Loading Alignment model...")
    model = SD2AlignmentEnhancer(
        base_model_path=args.base_model_path,
        weight_path=args.alignment_weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    model.init_models()

    captioner = get_captioner(args)
    to_tensor = transforms.ToTensor()

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        img = Image.open(img_path).convert("RGB")
        lq_tensor = to_tensor(img).unsqueeze(0).to(args.device)
        prompt = get_prompt(args, img_path, captioner, img)

        print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")

        start_time = time()
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
        proc_time = time() - start_time

        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        output_path = output_dir / rel_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        print(f"  Saved: {output_path}")
        print(f"  Time: {proc_time:.2f}s | {img.size} -> {result.size}")


def test_full(args, images, output_dir):
    """
    Mode: full - RM + alignment + HYPIR (complete pipeline)
    Full two-stage pipeline with alignment.
    """
    if not RM_AVAILABLE or not ALIGNMENT_AVAILABLE:
        raise ImportError("RM and Alignment modules required")

    print("=" * 60)
    print("Mode: FULL (RM + Alignment + HYPIR)")
    print(f"RM task: {args.rm_task}")
    print(f"RM weights: {args.rm_weight_path}")
    print(f"Alignment weights: {args.alignment_weight_path}")
    print(f"Input: {len(images)} images")
    print("=" * 60)

    print("Loading RM model...")
    rm_model = RestorationModule(task=args.rm_task, device=args.device)
    rm_model.load(weight_path=args.rm_weight_path)

    print("Loading Alignment model...")
    align_model = SD2AlignmentEnhancer(
        base_model_path=args.base_model_path,
        weight_path=args.alignment_weight_path,
        lora_modules=args.lora_modules.split(","),
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t,
        device=args.device,
    )
    align_model.init_models()

    captioner = get_captioner(args)

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        img = Image.open(img_path).convert("RGB")

        # Stage 1: RM
        start_time = time()
        rm_output = rm_model.process_image(
            image=img,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            ensure_min_size=True,
        )
        rm_time = time() - start_time

        if args.save_intermediate:
            rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
            intermediate_path = output_dir / "intermediate" / rel_path.with_suffix(".png")
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)
            rm_model.save_output(rm_output, intermediate_path)
            print(f"  RM intermediate: {intermediate_path}")

        # Stage 2: Alignment + HYPIR
        prompt = get_prompt(args, img_path, captioner, img)
        print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")

        align_model.set_rm_output(rm_output)

        start_time = time()
        with torch.no_grad():
            result = align_model.enhance(
                lq=rm_output,
                prompt=prompt,
                scale_by=args.scale_by,
                upscale=args.upscale,
                target_longest_side=args.target_longest_side,
                patch_size=args.patch_size,
                stride=args.stride,
                return_type="pil",
            )[0]
        align_time = time() - start_time

        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        output_path = output_dir / rel_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        print(f"  Saved: {output_path}")
        print(f"  RM: {rm_time:.2f}s | Align+HYPIR: {align_time:.2f}s | Total: {rm_time + align_time:.2f}s")
        print(f"  {img.size} -> RM: {rm_output.shape[2:]} -> {result.size}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    images = find_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(images)} image(s)")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Base model: {args.base_model_path}\n")
        f.write(f"HYPIR weights: {args.hypir_weight_path}\n")
        f.write(f"Alignment weights: {args.alignment_weight_path}\n")
        f.write(f"RM task: {args.rm_task}\n")
        f.write(f"RM weights: {args.rm_weight_path}\n")
        f.write(f"Upscale: {args.upscale}\n")
        f.write(f"Seed: {args.seed}\n")

    mode_handlers = {
        "baseline": test_baseline,
        "rm": test_rm,
        "align": test_align,
        "full": test_full,
    }

    try:
        handler = mode_handlers[args.mode]
        handler(args, images, output_dir)

        print("\n" + "=" * 60)
        print(f"Test completed: {args.mode}")
        print(f"Results: {output_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
