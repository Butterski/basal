"""MLX backend: equals the PyTorch readout (tiny random Llama with the basal layout: attention and MLP biases, untied
head, grouped KV heads, transformers-5 `rope_parameters`), with a shared prefix (SOAM group), for unrelated prompts and
with 8-bit weights made at load time (mlx-q8)."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import MLXBackend, mlx_config_overrides


def tiny_llama(path):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=256, attention_bias=True,
                      mlp_bias=True, tie_word_embeddings=False, rope_parameters={"rope_type": "default",
                                                                                   "rope_theta": 1000000.0})
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():  # random biases (initialised to zero), so that a dropped bias would be noticed
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.1)
            elif n.endswith(("q_proj.weight", "k_proj.weight")):
                p.mul_(8.0)  # peaky attention, so that positions (rotary settings) matter
    m.save_pretrained(path)
    return m


def reference(m, toks, lids):
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1).tolist())
    return out


class _Tok:
    pad_token_id = 0


def mlx_stub(path, quant=None):
    from concurrent.futures import ThreadPoolExecutor
    be = MLXBackend.__new__(MLXBackend)  # no tokenizer files in the tiny model directory
    be.pool, be.tok, be.max_rows = ThreadPoolExecutor(max_workers=1), _Tok(), 3
    be.pool.submit(be._load, path, quant).result()
    return be


def test_rope_parameters_override():
    assert mlx_config_overrides({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}) == {"rope_theta": 1e6}
    assert mlx_config_overrides({"rope_theta": 5e5}) == {}
    o = mlx_config_overrides({"rope_parameters": {"rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0}})
    assert o["rope_theta"] == 5e5 and o["rope_scaling"]["rope_type"] == "llama3"


STATE = list(range(21, 61)) * 3  # 120 shared tokens
# SOAM group: state | question 1 (two orders) | question 2 (two orders) | question 3, suffixes of unequal length
TOKS = [STATE + [5, 6, 10, 11], STATE + [5, 6, 11, 10], STATE + [7, 8, 9, 10, 11, 12], STATE + [7, 8, 9, 12, 11, 10],
        STATE + [3]]
LIDS = [[10, 11], [10, 11], [10, 11, 12], [10, 11, 12], [10, 11, 12, 13]]


def test_mlx_equals_torch(tmp_path):
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = mlx_stub(tmp_path)
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    # unrelated prompts of different lengths (right-padded batches, no shared prefix), and a mix of both kinds
    singles = [list(range(1, 40)), list(range(5, 64)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]
    sl = [[10, 11], [12, 13, 14], [10, 15]]
    ref_s = reference(m, singles, sl)
    got_s = be.run(singles, sl)
    for r, g in zip(ref_s, got_s):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    mixed = be.run_shared([([singles[0]], [sl[0]]), (TOKS[:2], LIDS[:2])])
    assert max(abs(x - y) for x, y in zip(mixed[0][0], ref_s[0])) < 2e-4
    assert max(abs(x - y) for x, y in zip(mixed[1][1], ref[1])) < 2e-4


def test_mlx_without_override_differs(tmp_path):
    """Guard for the override: with mlx-lm's default rope_theta (10000) the readout changes."""
    pytest.importorskip("mlx.core")
    from mlx_lm.utils import load_model
    m = tiny_llama(tmp_path)
    toks, lids = [list(range(21, 61)) * 3 + [5]], [[10, 11, 12, 13]]
    be = mlx_stub(tmp_path)
    be.model, _ = load_model(tmp_path)  # no override
    assert max(abs(x - y) for x, y in zip(reference(m, toks, lids)[0], be.run(toks, lids)[0])) > 1e-3


def test_mlx_q8_quantises_decoder_blocks_only(tmp_path):
    pytest.importorskip("mlx.core")
    import mlx.nn as nn
    m = tiny_llama(tmp_path)
    be = mlx_stub(tmp_path, "q8")
    assert be.quantization == {"group_size": 64, "bits": 8}
    assert isinstance(be.model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(be.model.model.layers[1].mlp.down_proj, nn.QuantizedLinear)
    assert not isinstance(be.model.lm_head, nn.QuantizedLinear)
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):  # 8-bit weights: close to bf16 / fp32, not equal
        assert max(abs(x - y) for x, y in zip(r, g)) < 0.05
    with pytest.raises(ValueError, match="unknown MLX quantisation"):
        mlx_stub(tmp_path, "q4")
