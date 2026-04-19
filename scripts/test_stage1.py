"""
Smoke test for Stage 1 alignment pretraining trainer.

Verifies:
1. Model initialization (frozen UNet, trainable alignment only)
2. Parameter counts match expectations
3. Forward pass produces correct output shape
4. Only alignment params have gradients
"""

import sys
import os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

# Minimal config for testing (no real data/models needed for structure check)
cfg = OmegaConf.create({
    "base_model_path": "checkpoints/sd2",
    "gradient_checkpointing": False,
    "alignment": {
        "enabled": True,
        "latent_channels": 4,
        "encoder_block_out_channels": [128, 256, 512, 512],
        "transformer_layers": 2,
        "transformer_dim": 640,
        "transformer_heads": 8,
        "use_condition_embedding": True,
        "add_sample": True,
    },
})


def test_alignment_handler_only():
    """Test alignment handler independently."""
    from HYPIR.alignment.alignment_handler import AlignmentHandler

    handler = AlignmentHandler(
        unet_conv_channels=320,
        latent_channels=4,
        encoder_block_out_channels=(128, 256, 512, 512),
        transformer_layers=2,
        transformer_dim=640,
        transformer_heads=8,
    )

    total_params = sum(p.numel() for p in handler.parameters())
    trainable_params = sum(p.numel() for p in handler.parameters() if p.requires_grad)
    print(f"AlignmentHandler: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")

    # Forward pass
    x_en = torch.randn(1, 4, 64, 64)
    unet_sample = torch.randn(1, 320, 64, 64)
    out = handler(x_en, unet_sample)
    assert out.shape == (1, 320, 64, 64), f"Expected (1,320,64,64), got {out.shape}"
    print(f"AlignmentHandler forward: {out.shape} OK")


def test_unet_alignment_no_lora():
    """Test UNetAlignment without LoRA (Stage 1 scenario)."""
    from diffusers import UNet2DConditionModel
    from HYPIR.alignment.alignment_handler import AlignmentHandler
    from HYPIR.model.unet_alignment import UNetAlignment

    unet = UNet2DConditionModel.from_pretrained(
        "checkpoints/sd2", subfolder="unet",
        torch_dtype=torch.float32,
    )
    unet.eval().requires_grad_(False)

    # Verify UNet is frozen
    unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    assert unet_trainable == 0, f"UNet should be frozen but has {unet_trainable} trainable params"
    print(f"UNet frozen: trainable={unet_trainable} OK")

    handler = AlignmentHandler(
        unet_conv_channels=320,
        latent_channels=4,
        encoder_block_out_channels=(128, 256, 512, 512),
        transformer_layers=2,
        transformer_dim=640,
        transformer_heads=8,
    )

    G = UNetAlignment(unet=unet, alignment_handler=handler)

    # Count trainable params
    G_trainable = sum(p.numel() for p in G.parameters() if p.requires_grad)
    handler_params = sum(p.numel() for p in handler.parameters())
    print(f"UNetAlignment trainable: {G_trainable/1e6:.2f}M (should equal handler: {handler_params/1e6:.2f}M)")
    assert G_trainable == handler_params, "Trainable params should only be alignment handler"

    # Forward pass
    z_in = torch.randn(1, 4, 64, 64)
    timesteps = torch.tensor([200])
    text_embed = torch.randn(1, 77, 1024)
    x_en = torch.randn(1, 4, 64, 64)

    with torch.no_grad():
        eps = G(z_in, timesteps, encoder_hidden_states=text_embed, x_en=x_en).sample
    assert eps.shape == (1, 4, 64, 64), f"Expected (1,4,64,64), got {eps.shape}"
    print(f"UNetAlignment forward (no LoRA): {eps.shape} OK")

    # Gradient check — only alignment should have gradients
    G.zero_grad()
    z_in = torch.randn(1, 4, 64, 64)
    eps = G(z_in, timesteps, encoder_hidden_states=text_embed, x_en=x_en).sample
    loss = eps.mean()
    loss.backward()

    align_grads = sum(1 for p in handler.parameters() if p.grad is not None)
    unet_grads = sum(1 for p in unet.parameters() if p.grad is not None)
    total_handler = sum(1 for _ in handler.parameters())
    print(f"Gradient flow: alignment={align_grads}/{total_handler}, UNet={unet_grads}")
    assert align_grads == total_handler, f"All alignment params should have gradients"
    assert unet_grads == 0, f"UNet should have zero gradients"
    print("Gradient flow: OK")


def test_stage1_save_load():
    """Test that Stage1 save/load only handles alignment weights."""
    from diffusers import UNet2DConditionModel
    from HYPIR.alignment.alignment_handler import AlignmentHandler
    from HYPIR.model.unet_alignment import UNetAlignment
    import tempfile, os

    unet = UNet2DConditionModel.from_pretrained(
        "checkpoints/sd2", subfolder="unet",
        torch_dtype=torch.float32,
    )
    unet.eval().requires_grad_(False)

    handler = AlignmentHandler(
        unet_conv_channels=320,
        latent_channels=4,
    )
    G = UNetAlignment(unet=unet, alignment_handler=handler)

    # Simulate Stage1 save
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dict = {}
        for name, param in handler.named_parameters():
            state_dict[f"alignment_handler.{name}"] = param.detach().clone()
        save_path = os.path.join(tmpdir, "state_dict.pth")
        torch.save(state_dict, save_path)
        print(f"Saved {len(state_dict)} alignment params")

        # Create new model and load
        handler2 = AlignmentHandler(unet_conv_channels=320, latent_channels=4)
        G2 = UNetAlignment(unet=unet, alignment_handler=handler2)

        loaded_sd = torch.load(save_path, map_location="cpu")
        align_sd = {k: v for k, v in loaded_sd.items() if k.startswith("alignment_handler.")}
        m, u = G2.load_state_dict(align_sd, strict=False)
        print(f"Loaded {len(align_sd)} params, missing: {len(m)}, unexpected: {len(u)}")

        # Verify weights match
        for name in handler.named_parameters():
            pass  # just iterate
        for (n1, p1), (n2, p2) in zip(handler.named_parameters(), handler2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch at {n1}"
        print("Stage1 save/load: weights match OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: AlignmentHandler standalone")
    print("=" * 60)
    test_alignment_handler_only()

    print()
    print("=" * 60)
    print("Test 2: UNetAlignment without LoRA (Stage 1)")
    print("=" * 60)
    test_unet_alignment_no_lora()

    print()
    print("=" * 60)
    print("Test 3: Stage1 save/load")
    print("=" * 60)
    test_stage1_save_load()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
