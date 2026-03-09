#!/usr/bin/env python3
"""
RM+HYPIR Two-Stage Model Test Script

Supports three testing modes:
1. RM only: Test restoration module alone
2. HYPIR only: Test HYPIR model alone
3. Full: Test two-stage pipeline (RM + HYPIR)

Usage examples:
  # Mode 1: Test RM only (BID task)
  python test_rm_hypir.py --mode rm --task bid --rm_model_path weights/scunet.pth \
         --input examples/lq/ --output results/rm/

  # Mode 2: Test HYPIR only
  python test_rm_hypir.py --mode hypir --base_model_path stabilityai/stable-diffusion-2-1-base \
         --hypir_weight_path weights/HYPIR_sd2.pth --input examples/lq/ --output results/hypir/

  # Mode 3: Test full pipeline (BFR task)
  python test_rm_hypir.py --mode full --task bfr --rm_model_path weights/swinir_face.pth \
         --base_model_path stabilityai/stable-diffusion-2-1-base --hypir_weight_path weights/HYPIR_sd2.pth \
         --input examples/lq/ --output results/full/
"""

import argparse
import os
import sys
from pathlib import Path
from time import time
import torch
from PIL import Image
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from HYPIR.rm.restoration_module import RestorationModule
    from HYPIR.pipeline.rm_hypir_pipeline import RMHYPIRPipeline
    RM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RM modules not available: {e}")
    RM_AVAILABLE = False

try:
    from HYPIR.enhancer.sd2 import SD2Enhancer
    from HYPIR.utils.captioner import EmptyCaptioner, FixedCaptioner
    HYPIR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HYPIR modules not available: {e}")
    HYPIR_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test RM+HYPIR two-stage model")

    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                        choices=["rm", "hypir", "full"],
                        help="Test mode: 'rm' (RM only), 'hypir' (HYPIR only), 'full' (both)")

    # RM-specific arguments
    parser.add_argument("--task", type=str, default="bid",
                        choices=["bid", "bfr", "bsr"],
                        help="RM task type: 'bid' (denoising), 'bfr' (face), 'bsr' (super-res)")
    parser.add_argument("--rm_model_path", type=str,
                        help="Path to RM weight file (.pth, .ckpt, .pt)")
    parser.add_argument("--rm_config_path", type=str,
                        help="Optional path to RM config file (YAML)")
    parser.add_argument("--rm_upscale", type=float, default=1.0,
                        help="Upscale factor for RM stage. Note: RM models may have built-in upsampling.")

    # HYPIR-specific arguments
    parser.add_argument("--base_model_path", type=str,
                        help="Path to base Stable Diffusion model (directory or HF repo)")
    parser.add_argument("--hypir_weight_path", type=str,
                        help="Path to HYPIR LoRA weight file")
    parser.add_argument("--lora_modules", type=str, default="to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj",
                        help="Comma-separated LoRA module names")
    parser.add_argument("--lora_rank", type=int, default=256,
                        help="LoRA rank")
    parser.add_argument("--model_t", type=int, default=200,
                        help="Model input timestep")
    parser.add_argument("--coeff_t", type=int, default=200,
                        help="Timestep for conversion coefficients")
    parser.add_argument("--scale_by", type=str, default="factor", choices=["factor", "longest_side"],
                        help="Method to scale the input images. 'factor' scales by a fixed factor, 'longest_side' scales by the longest side (to a fixed size).")
    parser.add_argument("--target_longest_side", type=int, default=None,
                        help="Target longest side for scaling if 'scale_by' is set to 'longest_side'.")
    parser.add_argument("--txt_dir", type=str, default=None,
                        help="Directory containing text prompts for images. If provided, prompts will be read from text files matching image names.")

    # Common arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input image directory or single image file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate results (RM output in full mode)")

    # Processing parameters
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for RM tiled inference")
    parser.add_argument("--tile_stride", type=int, default=256,
                        help="Tile stride for RM tiled inference")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Patch size for HYPIR processing")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for HYPIR processing")
    parser.add_argument("--upscale", type=float, default=1.0,
                        help="Upscale factor for HYPIR")
    parser.add_argument("--ensure_min_size", action="store_true",
                        help="Resize RM output to meet minimum size (512x512) for HYPIR")

    # Prompt/caption settings
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional text prompt for HYPIR enhancement. If empty, captioner will be used.")
    parser.add_argument("--captioner", type=str, default="empty",
                        choices=["empty", "fixed"],
                        help="Captioner type for HYPIR (default: empty)")
    parser.add_argument("--fixed_caption", type=str,
                        help="Fixed caption if captioner is 'fixed'")

    # System settings
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=231,
                        help="Random seed")

    return parser.parse_args()


def find_images(input_path: str) -> list:
    """Find all image files in directory or return single image."""
    input_path = Path(input_path)

    if input_path.is_file():
        # Single image file
        return [input_path]

    # Directory - find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = []

    for root, dirs, files in os.walk(input_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                images.append(Path(root) / file)

    images.sort()
    return images


def test_rm_only(args, images, output_dir):
    """Test RM model only."""
    if not RM_AVAILABLE:
        raise ImportError("RM modules not available. Check installation.")

    print("=" * 60)
    print("Testing RM Model Only")
    print(f"Task: {args.task}")
    print(f"Model: {args.rm_model_path}")
    print(f"RM Upscale: {args.rm_upscale}")
    print(f"Input: {len(images)} images")
    print("=" * 60)

    # Initialize RM model
    print("Loading RM model...")
    rm_model = RestorationModule(task=args.task, device=args.device)
    rm_model.load(
        weight_path=args.rm_model_path,
        config_path=args.rm_config_path
    )

    # Process images
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Process with RM
        start_time = time()
        output_tensor = rm_model.process_image(
            image=img,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            ensure_min_size=args.ensure_min_size,
            upscale=args.rm_upscale
        )
        proc_time = time() - start_time

        # Save output
        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        output_path = output_dir / rel_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rm_model.save_output(output_tensor, output_path)

        print(f"  Saved to: {output_path}")
        print(f"  Processing time: {proc_time:.2f}s")
        print(f"  Input size: {img.size}, Output size: {output_tensor.shape[2:]}")


def test_hypir_only(args, images, output_dir):
    """Test HYPIR model only."""
    if not HYPIR_AVAILABLE:
        raise ImportError("HYPIR modules not available. Check installation.")

    print("=" * 60)
    print("Testing HYPIR Model Only")
    print(f"Base model: {args.base_model_path}")
    print(f"LoRA weights: {args.hypir_weight_path}")
    print(f"Input: {len(images)} images")
    print("=" * 60)

    # Initialize captioner
    if args.captioner == "empty":
        captioner = EmptyCaptioner(args.device)
    elif args.captioner == "fixed":
        if not args.fixed_caption:
            raise ValueError("--fixed_caption required when --captioner=fixed")
        captioner = FixedCaptioner(args.device, args.fixed_caption)
    else:
        raise ValueError(f"Unknown captioner: {args.captioner}")

    # Initialize HYPIR model
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

    # Process images
    to_tensor = transforms.ToTensor()

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        # Load and prepare image
        img = Image.open(img_path).convert("RGB")
        lq_tensor = to_tensor(img).unsqueeze(0).to(args.device)

        # Get prompt
        if args.txt_dir is not None:
            # Read prompt from text file
            txt_dir = Path(args.txt_dir)
            rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
            prompt_path = txt_dir / rel_path.with_suffix(".txt")
            if prompt_path.exists():
                with open(prompt_path, "r") as fp:
                    prompt = fp.read().strip()
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        elif args.prompt:
            prompt = args.prompt
        else:
            prompt = captioner(img)

        print(f"  Prompt: {prompt}")

        # Process with HYPIR
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

        # Save output
        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        output_path = output_dir / rel_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result.save(output_path)

        print(f"  Saved to: {output_path}")
        print(f"  Processing time: {proc_time:.2f}s")
        print(f"  Input size: {img.size}, Output size: {result.size}")


def test_full_pipeline(args, images, output_dir):
    """Test full RM+HYPIR pipeline."""
    if not RM_AVAILABLE or not HYPIR_AVAILABLE:
        raise ImportError("Both RM and HYPIR modules required for full pipeline.")

    print("=" * 60)
    print("Testing Full RM+HYPIR Pipeline")
    print(f"RM task: {args.task}")
    print(f"RM model: {args.rm_model_path}")
    print(f"RM Upscale: {args.rm_upscale}")
    print(f"HYPIR base: {args.base_model_path}")
    print(f"HYPIR Upscale: {args.upscale}")
    print(f"Input: {len(images)} images")
    print("=" * 60)

    # Initialize pipeline
    print("Loading pipeline...")
    pipeline = RMHYPIRPipeline(device=args.device)

    # Load RM model
    pipeline.load_rm(
        task=args.task,
        weight_path=args.rm_model_path,
        config_path=args.rm_config_path
    )

    # Load HYPIR model
    pipeline.load_hypir(
        base_model_path=args.base_model_path,
        weight_path=args.hypir_weight_path,
        lora_modules=args.lora_modules,
        lora_rank=args.lora_rank,
        model_t=args.model_t,
        coeff_t=args.coeff_t
    )

    # Initialize captioner
    if args.captioner == "empty":
        captioner = EmptyCaptioner(args.device)
    elif args.captioner == "fixed":
        if not args.fixed_caption:
            raise ValueError("--fixed_caption required when --captioner=fixed")
        captioner = FixedCaptioner(args.device, args.fixed_caption)
    else:
        raise ValueError(f"Unknown captioner: {args.captioner}")

    # Process images
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Set up intermediate save path if requested
        intermediate_path = None
        if args.save_intermediate:
            rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
            intermediate_path = output_dir / "intermediate" / rel_path.with_suffix(".png")
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)

        # Get prompt (handle txt_dir, prompt, or captioner)
        if args.txt_dir is not None:
            # Read prompt from text file
            txt_dir = Path(args.txt_dir)
            rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
            prompt_path = txt_dir / rel_path.with_suffix(".txt")
            if prompt_path.exists():
                with open(prompt_path, "r") as fp:
                    prompt = fp.read().strip()
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        elif args.prompt:
            prompt = args.prompt
        else:
            # Use captioner to generate prompt
            prompt = captioner(img)

        # Run full pipeline
        start_time = time()
        output_tensor = pipeline.run(
            lq_image=img,
            prompt=prompt,
            rm_tile_size=args.tile_size,
            rm_tile_stride=args.tile_stride,
            rm_upscale=args.rm_upscale,
            hypir_patch_size=args.patch_size,
            hypir_stride=args.stride,
            hypir_upscale=args.upscale,
            save_intermediate=str(intermediate_path) if intermediate_path else None,
            captioner=args.captioner,
            fixed_caption=args.fixed_caption,
            scale_by=args.scale_by,
            target_longest_side=args.target_longest_side,
            ensure_min_size=args.ensure_min_size
        )
        proc_time = time() - start_time

        # Save final output
        rel_path = img_path.relative_to(args.input) if Path(args.input).is_dir() else Path(img_path.name)
        output_path = output_dir / "final" / rel_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline.save_output(output_tensor, output_path)

        print(f"  Saved to: {output_path}")
        if intermediate_path:
            print(f"  Intermediate saved to: {intermediate_path}")
        print(f"  Total processing time: {proc_time:.2f}s")
        print(f"  Input size: {img.size}, Output size: {output_tensor.shape[2:]}")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Check mode requirements
    if args.mode in ["rm", "full"] and not args.rm_model_path:
        raise ValueError("--rm_model_path required for RM and full modes")

    if args.mode in ["hypir", "full"] and not args.base_model_path:
        raise ValueError("--base_model_path required for HYPIR and full modes")

    if args.mode in ["hypir", "full"] and not args.hypir_weight_path:
        raise ValueError("--hypir_weight_path required for HYPIR and full modes")

    # Find input images
    images = find_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(images)} image(s)")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run appropriate test mode
    try:
        if args.mode == "rm":
            test_rm_only(args, images, output_dir)
        elif args.mode == "hypir":
            test_hypir_only(args, images, output_dir)
        elif args.mode == "full":
            test_full_pipeline(args, images, output_dir)

        print("\n" + "=" * 60)
        print(f"Test completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()