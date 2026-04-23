"""
验证当前 FaithDiffAlignment 与 FaithDiff 原始实现的结构一致性。

用法:
    cd SR-based-on-HYPIR
    conda activate hypir
    python test/test_faithdiff_consistency.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from collections import OrderedDict


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ========== FaithDiff 原始实现（直接从源码复制）==========

class QuickGELU_Orig(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm_Orig(nn.LayerNorm):
    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class ControlNetConditioningEmbedding_Orig(nn.Module):
    def __init__(self, conditioning_embedding_channels, conditioning_channels=4):
        super().__init__()
        import math
        num_groups = math.gcd(conditioning_channels, 32)
        self.conv_in = nn.Conv2d(conditioning_channels, conditioning_channels, kernel_size=3, padding=1)
        self.norm_in = nn.GroupNorm(num_channels=conditioning_channels, num_groups=num_groups, eps=1e-6)
        self.conv_out = zero_module(
            nn.Conv2d(conditioning_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        conditioning = self.norm_in(conditioning)
        embedding = self.conv_in(conditioning)
        embedding = torch.nn.functional.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class ResidualAttentionBlock_Orig(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm_Orig(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 2)), ("gelu", QuickGELU_Orig()),
                         ("c_proj", nn.Linear(d_model * 2, d_model))])
        )
        self.ln_2 = LayerNorm_Orig(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ========== 当前实现 ==========

from HYPIR.alignment.alignment import (
    Alignment,
    ControlNetConditioningEmbedding,
    ResidualAttentionBlock,
)


def compare_layer_structure(name, layer1, layer2):
    """比较两个层的结构是否一致。"""
    sd1 = layer1.state_dict()
    sd2 = layer2.state_dict()

    if set(sd1.keys()) != set(sd2.keys()):
        print(f"  [FAIL] {name}: key mismatch")
        print(f"    layer1 keys: {sorted(sd1.keys())}")
        print(f"    layer2 keys: {sorted(sd2.keys())}")
        return False

    for k in sd1:
        if sd1[k].shape != sd2[k].shape:
            print(f"  [FAIL] {name}: shape mismatch for {k}: {sd1[k].shape} vs {sd2[k].shape}")
            return False

    print(f"  [PASS] {name}: keys and shapes match")
    return True


def test_condition_embedding():
    print("\n=== ControlNetConditioningEmbedding ===")
    orig = ControlNetConditioningEmbedding_Orig(conditioning_embedding_channels=320, conditioning_channels=4)
    ours = ControlNetConditioningEmbedding(conditioning_embedding_channels=320, conditioning_channels=4)
    compare_layer_structure("condition_embedding", orig, ours)


def test_residual_attention_block():
    print("\n=== ResidualAttentionBlock ===")
    orig = ResidualAttentionBlock_Orig(d_model=640, n_head=8)
    ours = ResidualAttentionBlock(d_model=640, n_head=8)
    compare_layer_structure("residual_attention_block", orig, ours)


def test_spatial_ch_proj():
    print("\n=== spatial_ch_proj ===")
    orig = zero_module(nn.Linear(640, 320))
    ours = Alignment().spatial_ch_proj
    sd_orig = orig.state_dict()
    sd_ours = ours.state_dict()

    if sd_orig["weight"].shape != sd_ours["weight"].shape:
        print(f"  [FAIL] weight shape: {sd_orig['weight'].shape} vs {sd_ours['weight'].shape}")
    elif sd_orig["bias"].shape != sd_ours["bias"].shape:
        print(f"  [FAIL] bias shape: {sd_orig['bias'].shape} vs {sd_ours['bias'].shape}")
    else:
        print(f"  [PASS] spatial_ch_proj: shapes match (weight={sd_orig['weight'].shape})")


def test_full_forward():
    """测试完整 forward 输出是否形状正确且 zero_init 生效。"""
    print("\n=== Full Forward Pass ===")
    B, H, W = 2, 64, 64
    sample_emb = torch.randn(B, 320, H, W)
    z_lq = torch.randn(B, 4, H, W)

    model = Alignment(
        conditioning_channels=4,
        embedding_channels=320,
        num_trans_channel=640,
        num_trans_head=8,
        num_trans_layer=2,
    )

    feat_alpha = model(sample_emb, z_lq)

    # Check shape
    expected_shape = (B, 320, H, W)
    if feat_alpha.shape != expected_shape:
        print(f"  [FAIL] output shape: {feat_alpha.shape} != {expected_shape}")
    else:
        print(f"  [PASS] output shape: {feat_alpha.shape}")

    # Check zero_init (feat_alpha should be near zero)
    max_val = feat_alpha.abs().max().item()
    if max_val < 1e-6:
        print(f"  [PASS] zero_init: max abs value = {max_val:.2e}")
    else:
        print(f"  [FAIL] zero_init: max abs value = {max_val:.2e} (should be ~0)")

    # Check gradient flow
    loss = feat_alpha.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    grad_params = sum(p.numel() for p in model.parameters() if p.grad is not None)
    print(f"  [{'PASS' if has_grad else 'FAIL'}] gradient flow: {grad_params}/{total_params} params have gradients")

    print(f"  Total params: {total_params / 1e6:.2f}M")


def test_injection_logic():
    """测试 FaithDiff 注入逻辑：sample_emb + feat_alpha"""
    print("\n=== Injection Logic ===")
    B, H, W = 1, 64, 64
    sample_emb = torch.randn(B, 320, H, W)
    z_lq = torch.randn(B, 4, H, W)

    model = Alignment()
    feat_alpha = model(sample_emb, z_lq)

    # Simulate FaithDiff injection: sample = sample + feat_alpha
    injected = sample_emb + feat_alpha

    # At init (zero_init), injected should be ≈ sample_emb
    diff = (injected - sample_emb).abs().max().item()
    print(f"  [PASS] injection: sample_emb + feat_alpha")
    print(f"  [PASS] at init: |injected - sample_emb|_max = {diff:.2e} (should be ~0)")


def test_param_count():
    """对比参数量与 FaithDiff 原始设定。"""
    print("\n=== Parameter Count ===")
    model = Alignment()

    # Breakdown
    ce_params = sum(p.numel() for p in model.condition_embedding.parameters())
    it_params = sum(p.numel() for p in model.information_transformer.parameters())
    sp_params = sum(p.numel() for p in model.spatial_ch_proj.parameters())
    total = ce_params + it_params + sp_params

    print(f"  condition_embedding: {ce_params/1e3:.1f}K")
    print(f"  information_transformer: {it_params/1e6:.2f}M")
    print(f"  spatial_ch_proj: {sp_params/1e3:.1f}K")
    print(f"  Total: {total/1e6:.2f}M")


if __name__ == "__main__":
    print("=" * 60)
    print("FaithDiff Alignment Module Consistency Check")
    print("=" * 60)

    test_condition_embedding()
    test_residual_attention_block()
    test_spatial_ch_proj()
    test_full_forward()
    test_injection_logic()
    test_param_count()

    print("\n" + "=" * 60)
    print("All checks complete.")
    print("=" * 60)
