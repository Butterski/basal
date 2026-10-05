"""Shared-prefix packing is exactly equivalent to separate forward passes (tiny random model, CPU)."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import GraphBackend


class Stub(GraphBackend):
    def __init__(self, model):  # no loading, no CUDA graphs
        self.model, self.dev, self._wrows = model, torch.device("cpu"), {}


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return m


def separate(m, toks, lids):
    """Reference: one forward per prompt, full log-softmax over the vocabulary, softmax over the letters."""
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1))
    return out


def test_pack():
    ids, pos, seg, last, par = GraphBackend._pack([[1, 2, 3, 4, 5], [1, 2, 3, 7, 8, 9]])
    assert ids == [1, 2, 3, 4, 5, 7, 8, 9] and pos == [0, 1, 2, 3, 4, 3, 4, 5]
    assert seg == [0, 0, 0, 1, 1, 2, 2, 2] and last == [4, 7] and par == [-1, 0, 0]


def test_pack_trie_shares_nested_prefixes():
    """state | question 1 (two orders) | question 2 (two orders): the question text is shared by its orders."""
    s, q1, q2 = [1, 2, 3], [10, 11], [20, 21]
    toks = [s + q1 + [4, 5], s + q1 + [5, 4], s + q2 + [4, 5], s + q2 + [5, 4]]
    ids, pos, seg, last, par = GraphBackend._pack(toks)
    assert len(ids) == 3 + 2 * (2 + 2 * 2) and par == [-1, 0, 1, 1, 0, 4, 4]
    for m, tk in enumerate(toks):  # each readout's visible tokens (ancestors + own block) are exactly its prompt
        chain, b = [], seg[last[m]]
        while b >= 0:
            chain.append(b); b = par[b]
        seen = [ids[i] for i in range(last[m] + 1) if seg[i] in chain]
        assert seen == tk and pos[last[m]] == len(tk) - 1


def test_shared_equals_separate():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    be = Stub(m)
    a, b = [1, 5, 9, 11, 3, 4, 6], [1, 5, 9, 11, 3, 8, 2, 7]
    ids, pos, seg, last, par = GraphBackend._pack([a, b])
    h = be._forward_masked(torch.tensor([ids + [0] * 3]), be._mask_from_seg(torch.tensor([seg + [-1] * 3]), [par]),
                           torch.tensor([pos + [0] * 3]))
    with torch.no_grad():
        ha = m.model(input_ids=torch.tensor([a])).last_hidden_state[0, -1]
        hb = m.model(input_ids=torch.tensor([b])).last_hidden_state[0, -1]
    assert torch.allclose(h[0, last[0]], ha, atol=1e-4) and torch.allclose(h[0, last[1]], hb, atol=1e-4)


def test_soam_many_branches_equal_separate():
    """Every question x option order of a request packed into one row over the state (SOAM) equals separate prompts,
    with the row-only readout, including a row bisected because it exceeds the largest length."""
    m = tiny()
    be = Stub(m)
    state = [1, 5, 9, 11, 3, 4, 6, 12, 13, 14]
    toks = [state + [20 + q, 21, 30 + o, 31 - o, 40 + q, 41] for q in range(5) for o in range(2)]
    toks.append(list(toks[0]))  # a duplicate prompt reads out at the same position
    lids = [[7, 8, 9] if q % 2 else [7, 8] for q in range(5) for o in range(2)] + [[7, 8]]
    ref = separate(m, toks, lids)
    for lens in ([64], [24]):  # 24 < packed length: the group is bisected into smaller rows
        be.LENS = lens
        units = be._units([(toks, lids)])
        assert (len(units) > 1) == (lens == [24])
        probs = []
        for pk, ul, k, a in units:
            probs.append(be._eager_shared(pk, ul))
        res = GraphBackend._assemble([(toks, lids)], units, probs)[0]
        for r, e in zip(res, ref):
            assert torch.allclose(torch.tensor(r), e, atol=1e-5)


def test_row_only_readout_equals_full_softmax():
    m = tiny()
    be = Stub(m)
    h = torch.randn(3, 32)
    lids = [[3, 4], [4, 5, 6], [10, 3]]
    got = be._readout(h, lids)
    lp = torch.log_softmax(m.lm_head(h).float(), -1)
    for g, ids, row in zip(got, lids, lp):
        assert len(g) == len(ids) and torch.allclose(torch.tensor(g), torch.softmax(row[ids], -1), atol=1e-6)


def test_default_device_env_override(monkeypatch):
    from basal.engine import default_device
    monkeypatch.setenv("BASAL_DEVICE", "cpu")
    assert default_device() == "cpu"
    from basal.server import parser
    assert parser().parse_args(["--mode", "eager", "--device", "mps"]).device == "mps"


def test_sglang_letter_readout_parsing_and_equivalence():
    """1.5: the SGLang backend reads the option letters' log-probabilities; their softmax equals the letter readout."""
    import torch
    from basal.engine import SGLangBackend
    logits = torch.tensor([2.0, -1.0, 0.5, 3.0, 0.0])
    lsm = torch.log_softmax(logits, -1)
    ids = [3, 0, 2]
    meta = {"output_token_ids_logprobs": [[(float(lsm[i]), i, None) for i in (0, 2, 3)]]}  # any order
    got = torch.softmax(torch.tensor(SGLangBackend._letters(meta, ids)), -1)
    ref = torch.softmax(logits[ids], -1)
    assert torch.allclose(got, ref, atol=1e-6)
    meta = {"output_token_ids_logprobs": [[(float(lsm[3]), 3, None)]]}  # a missing letter gets no mass
    assert SGLangBackend._letters(meta, [3, 1])[1] == -1e9
