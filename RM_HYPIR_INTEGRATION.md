# RM+HYPIR Two-Stage Model Integration

This document describes the integration of DiffBIR Restoration Modules (RM) with HYPIR for two-stage image restoration.

## Overview

The integration combines:
1. **Restoration Module (RM)**: Removes noise, artifacts, and degradations
2. **HYPIR**: Generates realistic details and textures

### Supported RM Tasks
- **BID** (Blind Image Denoising): SCUNet model
- **BFR** (Blind Face Restoration): SwinIR model
- **BSR** (Blind Super-Resolution): BSRNet (RRDBNet) model

## Directory Structure

```
HYPIR/
├── DiffBIR/                    # Local copy of DiffBIR models
│   ├── models/                # Model definitions
│   │   ├── swinir.py
│   │   ├── scunet.py
│   │   └── bsrnet.py
│   ├── inference/             # Inference utilities
│   ├── configs/               # Model configurations
│   └── utils/                 # Utility functions
├── HYPIR/
│   ├── rm/                    # RM integration module
│   │   ├── model_loader.py    # RM model loader
│   │   ├── restoration_module.py  # Unified RM interface
│   │   └── tiled_inference.py # Tiled inference support
│   └── pipeline/              # Pipeline classes
│       └── rm_hypir_pipeline.py  # Two-stage pipeline
├── configs/
│   └── rm_examples/           # Example configurations
├── scripts/                   # Utility scripts
├── test_rm_hypir.py           # Main test script
└── test_import.py             # Import test script
```

## Installation

### Prerequisites
1. HYPIR base installation (see HYPIR/README.md)
2. PyYAML for configuration loading

```bash
pip install PyYAML
```

### Copy DiffBIR Models
The necessary DiffBIR model files are already copied to `HYPIR/DiffBIR/`.

## Usage

### 1. RM-Only Testing

Test the Restoration Module alone:

```bash
# BID task (Blind Image Denoising)
python test_rm_hypir.py --mode rm --task bid \
  --rm_model_path /path/to/scunet.pth \
  --input examples/lq/ \
  --output results/rm_bid/

# BFR task (Blind Face Restoration)
python test_rm_hypir.py --mode rm --task bfr \
  --rm_model_path /path/to/swinir_face.pth \
  --input examples/faces/ \
  --output results/rm_bfr/

# Using convenience script
python scripts/test_rm_only.py --task bid \
  --rm_model_path /path/to/scunet.pth \
  --input examples/lq/ --output results/rm/
```

### 2. HYPIR-Only Testing

Test HYPIR alone (baseline):

```bash
python test_rm_hypir.py --mode hypir \
  --base_model_path stabilityai/stable-diffusion-2-1-base \
  --hypir_weight_path /path/to/HYPIR_sd2.pth \
  --input examples/lq/ \
  --output results/hypir/

# Using convenience script
python scripts/test_hypir_only.py \
  --base_model_path stabilityai/stable-diffusion-2-1-base \
  --hypir_weight_path /path/to/HYPIR_sd2.pth \
  --input examples/lq/ --output results/hypir/
```

### 3. Full Two-Stage Testing

Test the complete RM+HYPIR pipeline:

```bash
# BID + HYPIR pipeline
python test_rm_hypir.py --mode full --task bid \
  --rm_model_path /path/to/scunet.pth \
  --base_model_path stabilityai/stable-diffusion-2-1-base \
  --hypir_weight_path /path/to/HYPIR_sd2.pth \
  --input examples/lq/ \
  --output results/full_bid/ \
  --save_intermediate

# BFR + HYPIR pipeline
python test_rm_hypir.py --mode full --task bfr \
  --rm_model_path /path/to/swinir_face.pth \
  --base_model_path stabilityai/stable-diffusion-2-1-base \
  --hypir_weight_path /path/to/HYPIR_sd2.pth \
  --input examples/faces/ \
  --output results/full_bfr/ \
  --prompt "a high quality face photo"
```

## Key Parameters

### RM Parameters
- `--task`: Task type (`bid`, `bfr`, `bsr`)
- `--rm_model_path`: Path to RM weight file (.pth, .ckpt, .pt)
- `--rm_config_path`: Optional RM config file (YAML)
- `--tile_size`: Tile size for tiled inference (default: 512)
- `--tile_stride`: Tile stride for tiled inference (default: 256)

### HYPIR Parameters
- `--base_model_path`: Base Stable Diffusion model path
- `--hypir_weight_path`: HYPIR LoRA weight file
- `--lora_modules`: LoRA module names (default: "attn1,attn2,ff")
- `--lora_rank`: LoRA rank (default: 256)
- `--patch_size`: HYPIR patch size (default: 512)
- `--stride`: HYPIR stride (default: 256)
- `--upscale`: Upscale factor (default: 1.0)

### Prompt/Caption Settings
- `--prompt`: Text prompt for HYPIR enhancement
- `--captioner`: Captioner type (`empty` or `fixed`)
- `--fixed_caption`: Fixed caption when using `fixed` captioner

## API Usage

### Programmatic Usage

```python
from HYPIR.pipeline.rm_hypir_pipeline import RMHYPIRPipeline

# Initialize pipeline
pipeline = RMHYPIRPipeline(device="cuda")

# Load models
pipeline.load_rm(
    task="bid",
    weight_path="/path/to/scunet.pth"
)

pipeline.load_hypir(
    base_model_path="stabilityai/stable-diffusion-2-1-base",
    weight_path="/path/to/HYPIR_sd2.pth"
)

# Process image
from PIL import Image
image = Image.open("input.jpg")

output = pipeline.run(
    lq_image=image,
    prompt="a high quality photo",
    save_intermediate="intermediate.jpg"
)

# Save result
pipeline.save_output(output, "result.jpg")
```

### Using RestorationModule Directly

```python
from HYPIR.rm.restoration_module import RestorationModule

# Initialize RM
rm = RestorationModule(task="bid", device="cuda")
rm.load(weight_path="/path/to/scunet.pth")

# Process image
from PIL import Image
image = Image.open("noisy.jpg")
clean = rm.process_image(image, tile_size=512, tile_stride=256)

# Save result
rm.save_output(clean, "clean.jpg")
```

## Configuration Files

Model configurations are in `HYPIR/DiffBIR/configs/inference/`:

- `swinir.yaml`: SwinIR configuration (BFR task)
- `scunet.yaml`: SCUNet configuration (BID task)
- `bsrnet.yaml`: BSRNet configuration (BSR task)

These files have been modified to use the local `DiffBIR.models` module path.

## Model Weights

You need to obtain the following pre-trained weights:

### RM Weights
- **SCUNet** (BID): `scunet_color_real_psnr.pth`
- **SwinIR** (BFR): `face_swinir_v1.ckpt`
- **BSRNet** (BSR): `BSRNet.pth`

### HYPIR Weights
- **HYPIR LoRA**: `HYPIR_sd2.pth`
- **Base SD2.1**: `stabilityai/stable-diffusion-2-1-base` (from HuggingFace)

## Troubleshooting

### Import Errors
- **"No module named 'yaml'**": Install PyYAML: `pip install PyYAML`
- **"No module named 'torch'**": Install PyTorch: `pip install torch torchvision`
- **DiffBIR import errors**: Check that model files are in `HYPIR/DiffBIR/models/`

### Model Loading Errors
- Check weight file paths are correct
- Ensure config files point to correct model paths
- Verify model compatibility with task type

### Performance Issues
- Adjust `--tile_size` and `--tile_stride` for memory constraints
- Use `--device cpu` if GPU memory is insufficient
- Reduce `--patch_size` for HYPIR if needed

## Development

### Adding New RM Models
1. Add model definition to `HYPIR/DiffBIR/models/`
2. Create config file in `HYPIR/DiffBIR/configs/inference/`
3. Update `RMModelLoader.SUPPORTED_TASKS` in `model_loader.py`
4. Add model loading logic in `RMModelLoader.create_model()`

### Extending Pipeline
- Modify `RMHYPIRPipeline` class in `rm_hypir_pipeline.py`
- Add new processing stages or options
- Update test scripts to support new features

## References

- **DiffBIR**: https://github.com/XPixelGroup/DiffBIR
- **HYPIR**: https://github.com/XPixelGroup/HYPIR
- **SwinIR**: https://arxiv.org/abs/2108.10257
- **SCUNet**: https://arxiv.org/abs/2201.07351
- **BSRNet**: https://arxiv.org/abs/2103.14006