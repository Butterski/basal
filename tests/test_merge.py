"""Merged q/k/v and gate/up projections: the same layer outputs as the separate linears (bf16 / fp32 weights
concatenated); an FP8 checkpoint's merged weight re-quantised per tensor stays within FP8 rounding."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import FP8Linear, merge_projections, merged_linear


def tiny(bias=True):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, attention_bias=bias, mlp_bias=bias, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


def test_merged_model_gives_the_same_logits():
    for bias in (True, False):
        m = tiny(bias)
        ids = torch.randint(0, 64, (2, 12))
        with torch.no_grad():
            ref = m(input_ids=ids).logits
            assert merge_projections(m) == 2
            got = m(input_ids=ids).logits
        a = m.model.layers[0].self_attn
        assert not hasattr(a, "q_proj") and a.qkv_proj.weight.shape == (32 + 2 * 16, 32)
        assert m.model.layers[0].mlp.gate_up_proj.weight.shape == (96, 32)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def fp8(lin):
    w = lin.weight.detach().float()
    s = w.abs().amax() / 448.0
    return FP8Linear((w / s).to(torch.float8_e4m3fn), s, torch.tensor(0.01), lin.bias.detach())


def test_fp8_merge_requantises_per_tensor():
    torch.manual_seed(1)
    lins = [torch.nn.Linear(32, n) for n in (32, 16, 16)]
    lins[1].weight.data *= 3  # a larger amax: the merged scale is the largest
    f = [fp8(x) for x in lins]
    m = merged_linear(f)
    assert isinstance(m, FP8Linear) and m.weight.shape == (64, 32) and m.weight.dtype == torch.float8_e4m3fn
    assert torch.isclose(m.weight_scale, max(x.weight_scale for x in f))
    x = torch.randn(5, 32)
    ref = torch.cat([y(x) for y in f], -1)
    got = m(x)
    assert (got - ref).abs().max() < 0.05 * ref.abs().max()      # FP8 rounding of the re-quantised weights


def test_merge_argument():
    from basal.server import merge_arg, parser
    assert [merge_arg(x) for x in ("auto", "on", "off")] == ["auto", True, False]
    assert parser().parse_args([]).merge == "auto"
