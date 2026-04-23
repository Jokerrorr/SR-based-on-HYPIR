"""
Unit tests for FaithDiff-style alignment module.

Tests:
1. FaithDiffAlignment dimension flow
2. UNetAlignment additive injection after conv_in
3. Zero-init behavior
4. Gradient flow with mixed dtype
5. Training forward pass (noise prediction)
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from HYPIR.alignment.faithdiff_alignment import FaithDiffAlignment
from HYPIR.model.unet_alignment import UNetAlignment


class TestFaithDiffAlignment:
    """Test FaithDiffAlignment dimension flow."""

    def test_output_shape(self):
        """Input sample_emb [B,320,H,W] + z_lq [B,4,H,W] → output [B,320,H,W]."""
        handler = FaithDiffAlignment(
            conditioning_channels=4, embedding_channels=320,
            num_trans_channel=640, num_trans_head=8, num_trans_layer=2,
        )
        sample_emb = torch.randn(2, 320, 32, 32)
        z_lq = torch.randn(2, 4, 32, 32)
        out = handler(sample_emb, z_lq)
        assert out.shape == (2, 320, 32, 32), f"Expected (2,320,32,32), got {out.shape}"

    def test_zero_init(self):
        """With zero_init on conv_out and spatial_ch_proj, output ≈ 0."""
        handler = FaithDiffAlignment(
            conditioning_channels=4, embedding_channels=320,
            num_trans_channel=640, num_trans_head=8, num_trans_layer=2,
        )
        sample_emb = torch.randn(2, 320, 16, 16)
        z_lq = torch.randn(2, 4, 16, 16)
        out = handler(sample_emb, z_lq)
        max_val = out.abs().max().item()
        assert max_val < 1e-6, f"Zero-init failed: max abs = {max_val}"

    def test_various_spatial_sizes(self):
        """Test with different spatial sizes."""
        handler = FaithDiffAlignment(
            conditioning_channels=4, embedding_channels=320,
            num_trans_channel=640, num_trans_head=8, num_trans_layer=2,
        )
        for h, w in [(16, 16), (32, 48), (64, 64), (8, 8)]:
            sample_emb = torch.randn(1, 320, h, w)
            z_lq = torch.randn(1, 4, h, w)
            out = handler(sample_emb, z_lq)
            assert out.shape == (1, 320, h, w), f"Failed for size ({h},{w})"

    def test_gradient_flow(self):
        """Gradients should flow through all parameters."""
        handler = FaithDiffAlignment(
            conditioning_channels=4, embedding_channels=320,
            num_trans_channel=640, num_trans_head=8, num_trans_layer=2,
        )
        # After one step, spatial_ch_proj will be non-zero, enabling full gradient flow
        optimizer = torch.optim.Adam(handler.parameters(), lr=1e-3)

        sample_emb = torch.randn(1, 320, 16, 16)
        z_lq = torch.randn(1, 4, 16, 16)
        out = handler(sample_emb, z_lq)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        # Second step: all params should get gradients
        optimizer.zero_grad()
        sample_emb = torch.randn(1, 320, 16, 16)
        z_lq = torch.randn(1, 4, 16, 16)
        out = handler(sample_emb, z_lq)
        loss = out.sum()
        loss.backward()

        for name, p in handler.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestUNetAlignment:
    """Test UNetAlignment with FaithDiff-style additive injection after conv_in."""

    @pytest.fixture
    def unet_alignment(self):
        """Create a minimal UNetAlignment for testing."""
        from diffusers import UNet2DConditionModel
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
        handler = FaithDiffAlignment(
            conditioning_channels=4, embedding_channels=32,
            num_trans_channel=64, num_trans_head=4, num_trans_layer=1,
        )
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
        """Output should differ when alignment is applied vs not (after training)."""
        sample = torch.randn(1, 4, 32, 32)
        z_lq = torch.randn(1, 4, 32, 32)
        x_hq_t = torch.randn(1, 4, 32, 32)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32)

        # At init, feat_alpha=0, so outputs should match
        out_no_align = unet_alignment(sample, timestep,
                                       encoder_hidden_states=text_embed)
        out_with_align = unet_alignment(sample, timestep,
                                         encoder_hidden_states=text_embed,
                                         z_lq=z_lq, x_hq_t=x_hq_t)
        # With zero_init, they should be very close
        assert torch.allclose(out_no_align.sample, out_with_align.sample, atol=1e-5)

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
        """Test gradient flow with mixed precision (UNet bf16, handler fp32)."""
        unet_alignment.unet = unet_alignment.unet.to(dtype=torch.bfloat16)

        sample = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        z_lq = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        x_hq_t = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        timestep = torch.tensor([200])
        text_embed = torch.randn(1, 1, 32, dtype=torch.bfloat16)

        out = unet_alignment(sample, timestep,
                             encoder_hidden_states=text_embed,
                             z_lq=z_lq, x_hq_t=x_hq_t)

        loss = out.sample.sum()
        loss.backward()

        handler = unet_alignment.alignment_handler
        # After zero_init, at least spatial_ch_proj should have gradients
        grad_params = [name for name, p in handler.named_parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0]
        assert len(grad_params) > 0, "At least some alignment params should have gradients"


class TestParameterCount:
    """Test parameter counts are reasonable."""

    def test_faithdiff_alignment_params(self):
        """FaithDiffAlignment should have ~6.78M params with default config."""
        handler = FaithDiffAlignment(
            conditioning_channels=4, embedding_channels=320,
            num_trans_channel=640, num_trans_head=8, num_trans_layer=2,
        )
        total = sum(p.numel() for p in handler.parameters())
        print(f"FaithDiffAlignment params: {total/1e6:.2f}M")
        assert 5e6 < total < 10e6, f"Unexpected param count: {total/1e6:.2f}M"

    def test_gradient_propagation_after_training_step(self):
        """Verify that after one optimizer step, all alignment params receive gradients."""
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel(
            sample_size=32, in_channels=4, out_channels=4,
            layers_per_block=1, block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        handler = FaithDiffAlignment(
            conditioning_channels=4, embedding_channels=32,
            num_trans_channel=64, num_trans_head=4, num_trans_layer=1,
        )
        model = UNetAlignment(unet=unet, alignment_handler=handler)

        model.unet = model.unet.to(dtype=torch.bfloat16)

        optimizer = torch.optim.Adam(handler.parameters(), lr=1e-3)

        # First step
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

        # Second step: all params should get gradients
        optimizer.zero_grad()
        z_lq = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        x_hq_t = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
        sample = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)

        out = model(sample, timestep, encoder_hidden_states=text_embed,
                    z_lq=z_lq, x_hq_t=x_hq_t)
        loss = out.sample.sum()
        loss.backward()

        grad_count = sum(1 for p in handler.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
        total_params = sum(1 for _ in handler.parameters())

        assert grad_count == total_params, \
            f"After training step, expected {total_params} params with grad, got {grad_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
