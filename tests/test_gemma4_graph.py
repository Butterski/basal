"""The Gemma 4 forward of the CUDA graph engine (basal/gemma4_graph.py) equals transformers' Gemma 4 (tiny random
models, CPU, float64 / float32): plain rows, the readout rows only, and SOAM's two passes through
GraphBackend.run_shared, with sliding-window and global layers, KV-shared layers and per-layer inputs (E2B / E4B) or
one projection for keys and values (12B / 31B)."""
import pytest
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from basal.engine import GraphBackend
from basal.gemma4_graph import Gemma4Graph, supports

S_, G_ = "sliding_attention", "full_attention"


def tiny(kind, dtype=torch.float64):
    """kind "e4b": per-layer inputs, the last 2 of 6 layers reuse keys / values; "31b": keys = values projection in
    the global layers (one KV head there). Window 5, so the states below are longer than it; random norms and layer
    scalars, so every weight matters."""
    torch.manual_seed(0)
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
              num_key_value_heads=2, head_dim=8, global_head_dim=16, sliding_window=5,
              layer_types=[S_, S_, G_, S_, S_, G_], final_logit_softcapping=3.0, max_position_embeddings=256)
    if kind == "e4b":
        kw.update(hidden_size_per_layer_input=8, vocab_size_per_layer_input=64, num_kv_shared_layers=2)
    else:
        kw.update(hidden_size_per_layer_input=0, attention_k_eq_v=True, num_global_key_value_heads=1)
    m = Gemma4ForCausalLM(Gemma4TextConfig(**kw)).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
        for layer in m.model.layers:
            layer.layer_scalar.fill_(0.9)
    m.config._attn_implementation = "sdpa"
    return m.to(dtype)


class Stub(GraphBackend):
    """GraphBackend without loading or CUDA graphs: every unit is longer than the largest graph (PLAIN_LENS / LENS
    = [4]), so it runs the uncompiled forwards (long single prompts, long roots, pass 2 at the readout rows)."""

    def __init__(self, model):
        self.model, self.dev, self._wrows, self.graphs = model, torch.device("cpu"), {}, {}
        self.gemma = Gemma4Graph(model)
        self.softcap = self.gemma.softcap
        self.tok = type("Tok", (), {"pad_token_id": 0})()
        self.shared, self.compact, self.prefix, self.state_cache = True, False, False, None
        self.PLAIN_LENS = self.LENS = [4]
        self.TOKEN_BUDGET = 64
        self._alloc_kv()

    def encode(self, prompts):
        return prompts


def separate(m, toks, lids):
    """Reference: one transformers forward per prompt, the full log-softmax of its (softcapped) logits, softmax over
    the codes."""
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1))
    return out


@pytest.mark.parametrize("kind", ["e4b", "31b"])
@pytest.mark.parametrize("merge", [False, True])
def test_plain_rows_equal_transformers(kind, merge):
    m = tiny(kind)
    ids = torch.randint(0, 64, (2, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = m.model(input_ids=ids).last_hidden_state
    g = Gemma4Graph(m, merge=merge)
    assert supports(m) is None and g.n_kv == (4 if kind == "e4b" else 6)
    assert g.merged == (6 if merge else 0) and len(g.qkv) == (g.n_kv if merge else 0)
    if merge:  # the modules' weights are views of the merged ones: transformers' forward is unchanged
        with torch.no_grad():
            assert torch.equal(m.model(input_ids=ids).last_hidden_state, ref)
    assert torch.allclose(g.forward(ids), ref, atol=1e-10)
    ro = torch.tensor([[12, 3], [7, 7]])  # the readout rows only (the KV-shared layers, the last own-key layer)
    assert torch.allclose(g.forward(ids, ro=ro), ref[torch.arange(2)[:, None], ro], atol=1e-10)


@pytest.mark.parametrize("kind", ["e4b", "31b"])
def test_two_passes_equal_separate(kind):
    """Every question x option order of a request over one state (SOAM), a duplicate, and a prompt that is the shared
    prefix itself (its readout in pass 1): pass 1 over the root into per-layer key / value buffers (two head sizes),
    pass 2 over the branches with the trie mask and the window on positions, read at the readout rows only."""
    m = tiny(kind, torch.float32)  # the engine's additive masks are float32 at most
    be = Stub(m)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]  # longer than the window: sliding layers drop its start
    toks = [state + [20 + q, 21, 30 + o, 31 - o, 40 + q, 41] + [50] * q for q in range(3) for o in range(2)]
    toks += [list(toks[0]), list(state)]
    lids = [[7, 8, 9] if q % 2 else [7, 8] for q in range(3) for o in range(2)] + [[7, 8], [7, 9]]
    ref = separate(m, toks, lids)
    res = be.run_shared([(toks, lids), ([toks[3]], [lids[3]])])
    for r, e in zip(res[0] + res[1], ref + [ref[3]]):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)


def test_moe_not_supported():
    cfg = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=16, enable_moe_block=True,
                           num_experts=2, top_k_experts=1, moe_intermediate_size=16, max_position_embeddings=64)
    assert "MoE" in supports(Gemma4ForCausalLM(cfg))


def test_readout_slots():
    """Pass-2 readouts (2, row, column) -> readout positions per row (padded with the row's first) and each readout's
    slot; a duplicate shares its column's slot, pass-1 readouts have none."""
    sel = [(2, 0, 5), (1, 1, 0), (2, 1, 3), (2, 0, 9), (2, 0, 5), (2, 1, 7)]
    ro, slot = GraphBackend._readout_slots(sel, 3)
    assert ro.tolist() == [[5, 9], [3, 7], [0, 0]] and slot == [0, None, 0, 1, 0, 1]


@pytest.mark.parametrize("L", [1, 129, 300, 1000])
@pytest.mark.parametrize("W", [None, 5, 128, 200, 512])
def test_plain_block_mask_is_the_causal_window(L, W):
    """The block mask of plain rows (built from block arithmetic): full blocks + the mask inside partial blocks give
    exactly the causal (sliding-window) mask."""
    g = Gemma4Graph(tiny("e4b"))
    g.window = W or 1
    bm = g._plain_block(L, "sliding_attention" if W else "full_attention", "cpu")
    q, k = torch.arange(L)[:, None], torch.arange(L)[None]
    true = (k <= q) & ((q - k < W) if W else True)
    got = torch.zeros(L, L, dtype=torch.bool)
    B = g.BLOCK
    for i in range(-(-L // B)):
        rows = slice(i * B, (i + 1) * B)
        for c in range(int(bm.full_kv_num_blocks[0, 0, i])):
            j = int(bm.full_kv_indices[0, 0, i, c])
            got[rows, j * B: (j + 1) * B] = True
        for c in range(int(bm.kv_num_blocks[0, 0, i])):
            j = int(bm.kv_indices[0, 0, i, c])
            got[rows, j * B: (j + 1) * B] = true[rows, j * B: (j + 1) * B]
    assert torch.equal(got, true)


def test_state_cache_second_round_equals_separate():
    """A request cut at its state (--state-cache-gb): the first call computes the state and stores its keys and values
    (own-key layers only, two head sizes), a second request over the same state runs pass 2 only over the cached
    ones; both equal separate prompts."""
    from basal.engine import StateCache
    m = tiny("e4b", torch.float32)
    be = Stub(m)
    be.state_cache = StateCache(1 << 30)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    first = [state + [20, 21, 30 + o, 31 - o] for o in range(2)]
    second = [state + [22, 23, 24, 30 + o] for o in range(2)]
    lids = [[7, 8], [7, 8]]
    for toks in (first, second):
        res = be.run_shared([(toks, lids, len(state))])[0]
        for r, e in zip(res, separate(m, toks, lids)):
            assert torch.allclose(torch.tensor(r), e, atol=1e-6)
    s = be.state_cache.stats()
    assert s["entries"] == 1 and s["hits"] == 1 and s["misses"] == 1


def test_long_lane_equals_separate():
    """The long lane (run_long): every unit alone, one decoder layer at a time (between(i) before layer i)."""
    m = tiny("31b", torch.float32)
    be = Stub(m)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14, 15, 16]
    toks = [state + [20, 21, 30 + o, 31 - o] for o in range(2)] + [state + [40, 41]]
    lids = [[7, 8], [7, 8], [7, 9]]
    calls = []
    res = be.run_long([(toks[:2], lids[:2]), (toks[2:], lids[2:])], between=calls.append)
    for r, e in zip(res[0] + res[1], separate(m, toks, lids)):
        assert torch.allclose(torch.tensor(r), e, atol=1e-6)
    assert calls == list(range(6)) * 2  # the root and the single prompt layer by layer; pass 2 runs whole


def test_modelopt_fp8_gemma4_loads_tied_and_runs(tmp_path):
    """A ModelOpt FP8 Gemma 4 checkpoint (the -FP8 repositories): the decoder linears become FP8Linear, the output head
    is tied to the embedding (not in the checkpoint), and the graph engine's forward equals transformers' forward of
    the loaded model (the FP8 product emulated on CPU)."""
    import json

    from safetensors.torch import save_file

    from basal.engine import FP8Linear, load_modelopt_fp8
    m = tiny("e4b", torch.float32)
    m.save_pretrained(tmp_path)
    sd = {}
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name and "per_layer" not in name:
            w = mod.weight.detach().float()
            ws = w.abs().max() / 448.0
            sd[f"{name}.weight"] = (w / ws).to(torch.float8_e4m3fn)
            sd[f"{name}.weight_scale"], sd[f"{name}.input_scale"] = ws.reshape(()), torch.tensor(0.05)
    for k, v in m.state_dict().items():
        if k != "lm_head.weight" and k.rsplit(".", 1)[0] + ".weight_scale" not in sd:
            sd.setdefault(k, v.detach().clone())
    for f in tmp_path.glob("*.safetensors"):
        f.unlink()
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "model.safetensors"))
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "FP8"}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    q = load_modelopt_fp8(tmp_path, "cpu", torch.float32)
    assert q.lm_head.weight.data_ptr() == q.model.embed_tokens.weight.data_ptr()
    assert sum(isinstance(x, FP8Linear) for x in q.modules()) > 0
    ids = torch.randint(0, 64, (1, 13), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = q.model(input_ids=ids).last_hidden_state[0, -1]
    assert torch.allclose(Gemma4Graph(q).forward(ids, ro=torch.tensor([[12]]))[0, 0], ref, atol=1e-5)
