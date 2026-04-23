"""
Smoke test for Stage 2: verify FaithDiffAlignment pretrained loading works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from HYPIR.alignment.faithdiff_alignment import FaithDiffAlignment
from HYPIR.model.unet_alignment import UNetAlignment


def test_stage2_pretrained_loading():
    """Simulate Stage1 → Stage2 weight transfer with FaithDiffAlignment."""
    # Step 1: Create Stage1 model and save alignment weights
    unet = UNet2DConditionModel.from_pretrained(
        "checkpoints/sd2", subfolder="unet",
        torch_dtype=torch.float32,
    )
    unet.eval().requires_grad_(False)

    handler1 = FaithDiffAlignment(
        conditioning_channels=4,
        embedding_channels=320,
        num_trans_channel=640,
        num_trans_head=8,
        num_trans_layer=2,
    )
    G1 = UNetAlignment(unet=unet, alignment_handler=handler1)

    # Run a forward pass to make weights non-trivial
    z_in = torch.randn(1, 4, 64, 64)
    x_hq_t = torch.randn(1, 4, 64, 64)
    t = torch.tensor([200])
    text = torch.randn(1, 77, 1024)
    with torch.no_grad():
        _ = G1(z_in, t, encoder_hidden_states=text, z_lq=z_in, x_hq_t=x_hq_t)

    # Simulate Stage1 training (modify alignment weights slightly)
    with torch.no_grad():
        for p in handler1.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    # Save Stage1 checkpoint (alignment only)
    with tempfile.TemporaryDirectory() as tmpdir:
        stage1_path = os.path.join(tmpdir, "state_dict.pth")
        stage1_sd = {}
        for name, param in handler1.named_parameters():
            stage1_sd[f"alignment_handler.{name}"] = param.detach().clone()
        torch.save(stage1_sd, stage1_path)
        print(f"Stage1 saved: {len(stage1_sd)} alignment params")

        # Step 2: Create Stage2 model (UNet + LoRA + FaithDiffAlignment)
        unet2 = UNet2DConditionModel.from_pretrained(
            "checkpoints/sd2", subfolder="unet",
            torch_dtype=torch.float32,
        )
        unet2.eval().requires_grad_(False)

        # Add LoRA
        lora_cfg = LoraConfig(
            r=256, lora_alpha=256,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet2.add_adapter(lora_cfg)

        handler2 = FaithDiffAlignment(
            conditioning_channels=4,
            embedding_channels=320,
            num_trans_channel=640,
            num_trans_head=8,
            num_trans_layer=2,
        )
        G2 = UNetAlignment(unet=unet2, alignment_handler=handler2)

        # Count trainable before loading
        lora_params = sum(1 for p in unet2.parameters() if p.requires_grad)
        align_params = sum(1 for _ in handler2.parameters())
        print(f"Stage2 before loading: LoRA trainable={lora_params}, Alignment={align_params}")

        # Load Stage1 alignment weights (simulating alignment_pretrained_path)
        pretrained_sd = torch.load(stage1_path, map_location="cpu")
        align_sd = {k: v for k, v in pretrained_sd.items() if k.startswith("alignment_handler.")}
        m, u = G2.load_state_dict(align_sd, strict=False)
        print(f"Stage2 loaded: {len(align_sd)} alignment params, missing: {len(m)}, unexpected: {len(u)}")

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(handler1.named_parameters(), handler2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch at {n1}"
        print("Stage1 → Stage2 weight transfer: weights match OK")

        # Verify forward pass works
        with torch.no_grad():
            eps = G2(z_in, t, encoder_hidden_states=text,
                     z_lq=z_in, x_hq_t=x_hq_t).sample
        assert eps.shape == (1, 4, 64, 64), f"Expected (1,4,64,64), got {eps.shape}"
        print(f"Stage2 forward pass: {eps.shape} OK")

        # Verify gradient flow to both LoRA and alignment
        G2.zero_grad()
        eps = G2(z_in, t, encoder_hidden_states=text,
                 z_lq=z_in, x_hq_t=x_hq_t).sample
        loss = eps.mean()
        loss.backward()

        lora_grads = sum(1 for p in unet2.parameters() if p.requires_grad and p.grad is not None)
        align_grads = sum(1 for p in handler2.parameters() if p.grad is not None)
        print(f"Stage2 gradient flow: LoRA={lora_grads}, Alignment={align_grads}/{align_params}")
        assert align_grads == align_params, "All alignment params should have gradients"
        print("Stage2 gradient flow: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 2 Pretrained Loading Test")
    print("=" * 60)
    test_stage2_pretrained_loading()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
