# RM Configuration Examples

This directory contains example configurations for different Restoration Module tasks.

## Available Configurations

### Minimal Configuration
```yaml
# minimal.yaml
task: bid
rm_model_path: /path/to/scunet.pth
base_model_path: stabilityai/stable-diffusion-2-1-base
hypir_weight_path: /path/to/HYPIR_sd2.pth
```

### BID (Blind Image Denoising) Task
```yaml
# bid_config.yaml
task: bid
rm_model_path: /path/to/scunet_color_real_psnr.pth
rm_config_path: DiffBIR/configs/inference/scunet.yaml
tile_size: 512
tile_stride: 256
```

### BFR (Blind Face Restoration) Task
```yaml
# bfr_config.yaml
task: bfr
rm_model_path: /path/to/face_swinir_v1.ckpt
rm_config_path: DiffBIR/configs/inference/swinir.yaml
tile_size: 512
tile_stride: 256
prompt: "a high quality face photo"
```

### BSR (Blind Super-Resolution) Task
```yaml
# bsr_config.yaml
task: bsr
rm_model_path: /path/to/BSRNet.pth
rm_config_path: DiffBIR/configs/inference/bsrnet.yaml
tile_size: 512
tile_stride: 256
hypir_upscale: 1.0  # RM already does 4x upscaling
```

## Usage

Load configuration in Python:

```python
import yaml

with open("configs/rm_examples/bfr_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Use config values
task = config["task"]
rm_model_path = config["rm_model_path"]
# ...
```

Or pass config file to test script:

```bash
python test_rm_hypir.py --mode full \
  --task $(yq eval '.task' configs/rm_examples/bfr_config.yaml) \
  --rm_model_path $(yq eval '.rm_model_path' configs/rm_examples/bfr_config.yaml) \
  --input examples/ --output results/
```

## Configuration Parameters

### Required Parameters
- `task`: RM task type (`bid`, `bfr`, `bsr`)
- `rm_model_path`: Path to RM weight file
- `base_model_path`: Path to base Stable Diffusion model
- `hypir_weight_path`: Path to HYPIR LoRA weights

### Optional Parameters
- `rm_config_path`: Custom RM config file
- `tile_size`: Tile size for RM (default: 512)
- `tile_stride`: Tile stride for RM (default: 256)
- `patch_size`: Patch size for HYPIR (default: 512)
- `stride`: Stride for HYPIR (default: 256)
- `upscale`: Upscale factor (default: 1.0)
- `prompt`: Text prompt for HYPIR
- `device`: Device to use (`cuda` or `cpu`)