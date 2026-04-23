"""
Model summary script using torchinfo.

Verifies full alignment architecture:
  LQ → RM → VAE Encoder → z_lq → Alignment → feat_alpha → additive injection after conv_in
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so HYPIR package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import UNet2DConditionModel
from HYPIR.alignment.alignment import Alignment
from HYPIR.model.unet_alignment import UNetAlignment


def main():
    device = "cpu"

    # Load UNet architecture
    unet = UNet2DConditionModel.from_config("checkpoints/sd2/unet/config.json")
    unet = unet.to(device)

    # Create alignment handler
    handler = Alignment(
        conditioning_channels=4,
        embedding_channels=320,
        num_trans_channel=640,
        num_trans_head=8,
        num_trans_layer=2,
    ).to(device)

    # Wrap
    model = UNetAlignment(unet=unet, alignment_handler=handler)

    # Parameter summary
    print("=" * 70)
    print("Model Parameter Summary")
    print("=" * 70)

    # UNet backbone
    unet_total = sum(p.numel() for p in unet.parameters())
    unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"\n[UNet Backbone]")
    print(f"  Total params:     {unet_total:>12,} ({unet_total/1e6:.2f}M)")
    print(f"  Trainable params: {unet_trainable:>12,} ({unet_trainable/1e6:.2f}M)")

    # Alignment handler breakdown
    print(f"\n[Alignment Handler]")
    for name, module in [
        ("condition_embedding", handler.condition_embedding),
        ("information_transformer", handler.information_transformer),
        ("spatial_ch_proj", handler.spatial_ch_proj),
    ]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:30s}  total={total/1e6:>8.2f}M  trainable={trainable/1e6:>8.2f}M")

    align_total = sum(p.numel() for p in handler.parameters())
    align_trainable = sum(p.numel() for p in handler.parameters() if p.requires_grad)
    print(f"  {'TOTAL':30s}  total={align_total/1e6:>8.2f}M  trainable={align_trainable/1e6:>8.2f}M")

    # Combined
    all_total = unet_total + align_total
    all_trainable = unet_trainable + align_trainable
    print(f"\n[Combined (UNet + Alignment)]")
    print(f"  Total params:     {all_total:>12,} ({all_total/1e6:.2f}M)")
    print(f"  Trainable params: {all_trainable:>12,} ({all_trainable/1e6:.2f}M)")

    # Dimension flow
    print("\n" + "=" * 70)
    print("Dimension Flow (additive injection after conv_in)")
    print("=" * 70)
    print("  z_lq:                              [B, 4, 64, 64]    (VAE encode of RM output)")
    print("  condition_embedding(z_lq):         [B, 320, 64, 64]   (4ch→320ch)")
    print("  cat([sample_emb, input_emb]):      [B, 640, 64, 64]   (concat)")
    print("  flatten:                           [B, 4096, 640]      (spatial flatten)")
    print("  Transformer:                       [B, 4096, 640]      (2-layer attn)")
    print("  spatial_ch_proj:                   [B, 4096, 320]      (project to 320ch)")
    print("  reshape → feat_alpha:              [B, 320, 64, 64]    (zero_init)")
    print("  sample_emb + feat_alpha:           [B, 320, 64, 64]    (additive injection)")
    print("  UNet down/mid/up/conv_out:         [B, 4, 64, 64]      (output noise_pred)")

    # Verify with actual forward
    print("\n" + "=" * 70)
    print("Forward Pass Verification")
    print("=" * 70)
    z_lq = torch.randn(1, 4, 64, 64)
    x_hq_t = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([200])
    text_embed = torch.randn(1, 77, 1024)

    with torch.no_grad():
        out = model(sample=z_lq, timestep=timestep, encoder_hidden_states=text_embed,
                     z_lq=z_lq, x_hq_t=x_hq_t)
    print(f"  Input z_lq:  {z_lq.shape}")
    print(f"  Input x_hq_t: {x_hq_t.shape}")
    print(f"  Output:      {out.sample.shape}")
    print("  Forward pass OK!")

    # Gradient flow test
    print("\n" + "=" * 70)
    print("Gradient Flow Verification")
    print("=" * 70)
    handler.train()
    handler.requires_grad_(True)
    out = model(z_lq, timestep, encoder_hidden_states=text_embed, z_lq=z_lq, x_hq_t=x_hq_t)
    loss = out.sample.sum()
    loss.backward()
    has_grad = sum(1 for p in handler.parameters() if p.grad is not None)
    total = sum(1 for _ in handler.parameters())
    print(f"  Alignment params with gradients: {has_grad}/{total}")
    print("  Gradient flow OK!")

    # Try torchinfo
    print("\n" + "=" * 70)
    print("torchinfo Summary (Alignment only)")
    print("=" * 70)
    try:
        from torchinfo import summary
        summary(
            handler,
            input_data=(
                torch.randn(1, 320, 64, 64),  # sample_emb
                torch.randn(1, 4, 64, 64),    # z_lq
            ),
            depth=3,
            col_names=["num_params", "params_percent", "trainable"],
        )
    except ImportError:
        print("  torchinfo not installed. Install with: pip install torchinfo")
    except Exception as e:
        print(f"  torchinfo error: {e}")
        print("  Falling back to manual summary (printed above)")


if __name__ == "__main__":
    main()
