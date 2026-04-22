"""
Unit tests for v4 FaithDiff dual-side alignment module.

Tests:
1. AlignmentModule dimension flow
2. AlignmentHandler interface
3. UNetAlignment injection (before conv_in)
4. Training forward pass (noise prediction)
5. Inference forward pass (single-step)
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HYPIR.alignment.alignment_module import AlignmentModule, zero_module
from HYPIR.alignment.alignment_handler import AlignmentHandler
from HYPIR.model.unet_alignment import UNetAlignment


class TestAlignmentModule:
    """Test AlignmentModule dimension flow."""

    def test_output_shape(self):
        """Input [B,4,H,W] + [B,4,H,W] → output [B,4,H,W]."""
        module = AlignmentModule(latent_channels=4, hidden_channels=32,
                                 encoder_channels=(32, 64, 128), num_layers=1, num_heads=4)
        z_lq = torch.randn(2, 4, 32, 32)
        x_hq_t = torch.randn(2, 4, 32, 32)
        out = module(z_lq, x_hq_t)
        assert out.shape == (2, 4, 32, 32), f"Expected (2,4,32,32), got {out.shape}"

    def test_residual_initial_zero(self):
        """With zero_init proj_out, output ≈ x_hq_t at initialization."""
        module = AlignmentModule(latent_channels=4, hidden_channels=32,
                                 encoder_channels=(32, 64, 128), num_layers=1, num_heads=4)
        z_lq = torch.randn(2, 4, 16, 16)
        x_hq_t = torch.randn(2, 4, 16, 16)
        out = module(z_lq, x_hq_t)
        # proj_out is zero-initialized, so out should equal x_hq_t
        assert torch.allclose(out, x_hq_t, atol=1e-6), \
            f"Expected output ≈ x_hq_t with zero_init, max diff: {(out - x_hq_t).abs().max()}"

    def test_various_spatial_sizes(self):
        """Test with different spatial sizes (must be divisible by 1, no constraint)."""
        module = AlignmentModule(latent_channels=4, hidden_channels=32,
                                 encoder_channels=(32, 64, 128), num_layers=1, num_heads=4)
        for h, w in [(16, 16), (32, 48), (64, 64), (8, 8)]:
            z_lq = torch.randn(1, 4, h, w)
            x_hq_t = torch.randn(1, 4, h, w)
            out = module(z_lq, x_hq_t)
            assert out.shape == (1, 4, h, w), f"Failed for size ({h},{w})"

    def test_gradient_flow(self):
        """Gradients should flow through all parameters."""
        module = AlignmentModule(latent_channels=4, hidden_channels=32,
                                 encoder_channels=(32, 64, 128), num_layers=1, num_heads=4)
        z_lq = torch.randn(1, 4, 16, 16, requires_grad=True)
        x_hq_t = torch.randn(1, 4, 16, 16, requires_grad=True)
        out = module(z_lq, x_hq_t)
        loss = out.sum()
        loss.backward()
        assert z_lq.grad is not None, "No gradient for z_lq"
        assert x_hq_t.grad is not None, "No gradient for x_hq_t"
        for name, p in module.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestAlignmentHandler:
    """Test AlignmentHandler wrapper."""

    def test_forward_interface(self):
        """Handler forward(z_lq, x_hq_t) → aligned."""
        handler = AlignmentHandler(latent_channels=4, hidden_channels=32,
                                   encoder_channels=(32, 64, 128), num_layers=1, num_heads=4)
        z_lq = torch.randn(2, 4, 16, 16)
        x_hq_t = torch.randn(2, 4, 16, 16)
        out = handler(z_lq, x_hq_t)
        assert out.shape == (2, 4, 16, 16)

    def test_legacy_config_params(self):
        """Handler should accept legacy config parameters."""
        handler = AlignmentHandler(
            latent_channels=4,
            encoder_block_out_channels=[32, 64, 128],
            transformer_layers=1,
            transformer_heads=4,
            # Legacy params (should be ignored)
            unet_conv_channels=320,
            transformer_dim=640,
            use_condition_embedding=True,
            add_sample=True,
        )
        z_lq = torch.randn(1, 4, 16, 16)
        x_hq_t = torch.randn(1, 4, 16, 16)
        out = handler(z_lq, x_hq_t)
        assert out.shape == (1, 4, 16, 16)


class TestUNetAlignment:
    """Test UNetAlignment with pre-conv_in injection."""

    @pytest.fixture
    def unet_alignment(self):
        """Create a minimal UNetAlignment for testing."""
        from diffusers import UNet2DConditionModel
        # Use tiny UNet for testing speed
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        handler = AlignmentHandler(latent_channels=4, hidden_channels=16,
                                   encoder_channels=(16, 32), num_layers=1, num_heads=2)
        return UNetAlignment(unet=unet, alignment_handler=handler)

    def test_forward_without_alignment(self, unet_alignment):
        """Forward without z_lq/x_hq_t should use standard UNet path."""
        sample = torch.randn(1, 4, 32, 32)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32)
        out = unet_alignment(sample, timestep, encoder_hidden_states=text_embed)
        assert out.sample.shape == (1, 4, 32, 32)

    def test_forward_with_alignment(self, unet_alignment):
        """Forward with z_lq and x_hq_t should use alignment path."""
        sample = torch.randn(1, 4, 32, 32)
        z_lq = torch.randn(1, 4, 32, 32)
        x_hq_t = torch.randn(1, 4, 32, 32)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32)
        out = unet_alignment(sample, timestep,
                             encoder_hidden_states=text_embed,
                             z_lq=z_lq, x_hq_t=x_hq_t)
        assert out.sample.shape == (1, 4, 32, 32)

    def test_alignment_changes_output(self, unet_alignment):
        """Output should differ when alignment is applied vs not."""
        sample = torch.randn(1, 4, 32, 32)
        z_lq = torch.randn(1, 4, 32, 32)
        x_hq_t = torch.randn(1, 4, 32, 32)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32)

        out_no_align = unet_alignment(sample, timestep,
                                       encoder_hidden_states=text_embed)
        out_with_align = unet_alignment(sample, timestep,
                                         encoder_hidden_states=text_embed,
                                         z_lq=z_lq, x_hq_t=x_hq_t)
        # Outputs should be different (alignment modifies the input)
        assert not torch.allclose(out_no_align.sample, out_with_align.sample, atol=1e-5)

    def test_conv_in_restored_after_forward(self, unet_alignment):
        """Original conv_in should be restored after forward pass."""
        original_conv_in = unet_alignment.unet.conv_in
        sample = torch.randn(1, 4, 32, 32)
        z_lq = torch.randn(1, 4, 32, 32)
        x_hq_t = torch.randn(1, 4, 32, 32)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32)

        unet_alignment(sample, timestep,
                       encoder_hidden_states=text_embed,
                       z_lq=z_lq, x_hq_t=x_hq_t)

        assert unet_alignment.unet.conv_in is original_conv_in, \
            "conv_in was not restored after forward pass"

    def test_gradient_flow_with_mixed_dtype(self, unet_alignment):
        """
        Test gradient flow with mixed precision (UNet bf16, handler fp32).

        Due to zero_init on proj_out, only proj_out params receive gradients
        at initialization. After one optimizer step, all params receive gradients.
        This is expected behavior for residual learning.
        """
        # Move UNet to bf16, keep handler fp32 (matching training)
        unet_alignment.unet = unet_alignment.unet.to(dtype=torch.bfloat16)

        sample = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        z_lq = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        x_hq_t = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32, dtype=torch.bfloat16)

        # Forward
        out = unet_alignment(sample, timestep,
                             encoder_hidden_states=text_embed,
                             z_lq=z_lq, x_hq_t=x_hq_t)

        # Backward
        loss = out.sample.sum()
        loss.backward()

        # At initialization, only proj_out gets gradients (zero_init behavior)
        # This is expected - proj_out is the "bottleneck" that must learn first
        handler = unet_alignment.alignment_handler
        grad_params = [name for name, p in handler.named_parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0]

        # proj_out should always have gradients
        assert any('proj_out' in name for name in grad_params), \
            f"proj_out should have gradients, got: {grad_params}"

        # After one simulated optimizer step, all params should get gradients
        # (verified in separate test below)


class TestParameterCount:
    """Test parameter counts are reasonable."""

    def test_alignment_module_params(self):
        """AlignmentModule should have ~20M params with default config."""
        module = AlignmentModule(latent_channels=4, hidden_channels=128,
                                 encoder_channels=(128, 256, 512), num_layers=2, num_heads=8)
        total = sum(p.numel() for p in module.parameters())
        print(f"AlignmentModule params: {total/1e6:.2f}M")
        assert 5e6 < total < 50e6, f"Unexpected param count: {total/1e6:.2f}M"

    def test_gradient_propagation_after_training_step(self):
        """
        Verify that after one optimizer step, all alignment params receive gradients.

        This tests the zero_init behavior: at initialization, only proj_out gets gradients.
        After proj_out learns (non-zero weights), gradients propagate to all layers.
        """
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel(
            sample_size=32, in_channels=4, out_channels=4,
            layers_per_block=1, block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        handler = AlignmentHandler(latent_channels=4, hidden_channels=16,
                                   encoder_channels=(16, 32), num_layers=1, num_heads=2)
        model = UNetAlignment(unet=unet, alignment_handler=handler)

        # Move UNet to bf16, keep handler fp32
        model.unet = model.unet.to(dtype=torch.bfloat16)

        optimizer = torch.optim.Adam(handler.parameters(), lr=1e-3)

        # First step: only proj_out gets gradients
        optimizer.zero_grad()
        z_lq = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        x_hq_t = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        sample = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32, dtype=torch.bfloat16)

        out = model(sample, timestep, encoder_hidden_states=text_embed,
                    z_lq=z_lq, x_hq_t=x_hq_t)
        loss = out.sample.sum()
        loss.backward()
        optimizer.step()

        # Second step: all params should get gradients now
        optimizer.zero_grad()
        z_lq = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        x_hq_t = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        sample = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)

        out = model(sample, timestep, encoder_hidden_states=text_embed,
                    z_lq=z_lq, x_hq_t=x_hq_t)
        loss = out.sample.sum()
        loss.backward()

        # Count params with gradients
        grad_count = sum(1 for p in handler.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
        total_params = sum(1 for _ in handler.parameters())

        assert grad_count == total_params, \
            f"After training step, expected {total_params} params with grad, got {grad_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
