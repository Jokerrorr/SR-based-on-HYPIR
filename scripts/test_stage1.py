"""
Smoke test for Stage 1 alignment pretraining trainer.

Verifies:
1. Alignment initialization and forward pass
2. UNetAlignment with frozen UNet (Stage 1 scenario)
3. Only alignment params have gradients
4. Stage 1 save/load roundtrip
"""

import sys
import os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from HYPIR.alignment.alignment import Alignment
from HYPIR.model.unet_alignment import UNetAlignment


def test_alignment_standalone():
    """Test Alignment independently."""
    handler = Alignment(
        conditioning_channels=4,
        embedding_channels=320,
        num_trans_channel=640,
        num_trans_head=8,
        num_trans_layer=2,
    )

    total_params = sum(p.numel() for p in handler.parameters())
    print(f"Alignment: total={total_params/1e6:.2f}M")

    # Forward pass
    sample_emb = torch.randn(1, 320, 64, 64)
    z_lq = torch.randn(1, 4, 64, 64)
    out = handler(sample_emb, z_lq)
    assert out.shape == (1, 320, 64, 64), f"Expected (1,320,64,64), got {out.shape}"
    print(f"Alignment forward: {out.shape} OK")

    # Zero-init check
    max_val = out.abs().max().item()
    assert max_val < 1e-6, f"Zero-init failed: max abs = {max_val}"
    print(f"Zero-init: max abs = {max_val:.2e} OK")


def test_unet_alignment_stage1():
    """Test UNetAlignment with frozen UNet (Stage 1 scenario)."""
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        "checkpoints/sd2", subfolder="unet",
        torch_dtype=torch.float32,
    )
    unet.eval().requires_grad_(False)

    unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    assert unet_trainable == 0, f"UNet should be frozen but has {unet_trainable} trainable params"
    print(f"UNet frozen: trainable={unet_trainable} OK")

    handler = Alignment(
        conditioning_channels=4,
        embedding_channels=320,
        num_trans_channel=640,
        num_trans_head=8,
        num_trans_layer=2,
    )

    G = UNetAlignment(unet=unet, alignment_handler=handler)

    # Count trainable params
    G_trainable = sum(p.numel() for p in G.parameters() if p.requires_grad)
    handler_params = sum(p.numel() for p in handler.parameters())
    print(f"UNetAlignment trainable: {G_trainable/1e6:.2f}M (should equal handler: {handler_params/1e6:.2f}M)")
    assert G_trainable == handler_params, "Trainable params should only be alignment handler"

    # Forward pass (additive injection after conv_in)
    z_lq = torch.randn(1, 4, 64, 64)
    x_hq_t = torch.randn(1, 4, 64, 64)
    timesteps = torch.tensor([200])
    text_embed = torch.randn(1, 77, 1024)

    with torch.no_grad():
        eps = G(z_lq, timesteps, encoder_hidden_states=text_embed,
                z_lq=z_lq, x_hq_t=x_hq_t).sample
    assert eps.shape == (1, 4, 64, 64), f"Expected (1,4,64,64), got {eps.shape}"
    print(f"UNetAlignment forward: {eps.shape} OK")

    # Gradient check — only alignment should have gradients
    G.zero_grad()
    eps = G(z_lq, timesteps, encoder_hidden_states=text_embed,
            z_lq=z_lq, x_hq_t=x_hq_t).sample
    loss = eps.mean()
    loss.backward()

    align_grads = sum(1 for p in handler.parameters() if p.grad is not None)
    unet_grads = sum(1 for p in unet.parameters() if p.grad is not None)
    total_handler = sum(1 for _ in handler.parameters())
    print(f"Gradient flow: alignment={align_grads}/{total_handler}, UNet={unet_grads}")
    assert align_grads == total_handler, "All alignment params should have gradients"
    assert unet_grads == 0, "UNet should have zero gradients"
    print("Gradient flow: OK")


def test_stage1_save_load():
    """Test that Stage1 save/load only handles alignment weights."""
    from diffusers import UNet2DConditionModel
    import tempfile

    unet = UNet2DConditionModel.from_pretrained(
        "checkpoints/sd2", subfolder="unet",
        torch_dtype=torch.float32,
    )
    unet.eval().requires_grad_(False)

    handler = Alignment(
        conditioning_channels=4,
        embedding_channels=320,
        num_trans_channel=640,
        num_trans_head=8,
        num_trans_layer=2,
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
        handler2 = FaithDiffAlignment(
            conditioning_channels=4,
            embedding_channels=320,
            num_trans_channel=640,
            num_trans_head=8,
            num_trans_layer=2,
        )
        G2 = UNetAlignment(unet=unet, alignment_handler=handler2)

        loaded_sd = torch.load(save_path, map_location="cpu")
        align_sd = {k: v for k, v in loaded_sd.items() if k.startswith("alignment_handler.")}
        m, u = G2.load_state_dict(align_sd, strict=False)
        print(f"Loaded {len(align_sd)} params, missing: {len(m)}, unexpected: {len(u)}")

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(handler.named_parameters(), handler2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch at {n1}"
        print("Stage1 save/load: weights match OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Alignment standalone")
    print("=" * 60)
    test_alignment_standalone()

    print()
    print("=" * 60)
    print("Test 2: UNetAlignment with frozen UNet (Stage 1)")
    print("=" * 60)
    test_unet_alignment_stage1()

    print()
    print("=" * 60)
    print("Test 3: Stage1 save/load")
    print("=" * 60)
    test_stage1_save_load()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
