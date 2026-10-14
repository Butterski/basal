"""Fast path routing on CPU (graphs replaced by eager replays of the same static buffers): single prompts in plain
graphs or a plain long forward, branched units in masked graphs or the two passes over the root's KV cache, all units
of all groups batched together and read out at once. Every readout equals a separate forward of its own prompt."""
import random

import torch

from basal.engine import GraphBackend, causal_sdpa
from tests.test_engine import separate, tiny


class Replay:
    """Stand-in for a captured CUDA graph: replay() recomputes the output from the static inputs, in place."""

    def __init__(self, fn, static):
        self.fn, self.static = fn, static
        self.out = fn(*static).clone()

    def replay(self):
        self.out.copy_(self.fn(*self.static))


class Stub(GraphBackend):
    LENS, PLAIN_LENS, BATCHES, TOKEN_BUDGET, SUFFIX = [16, 32], [16, 32, 48], [1, 2, 3, 4], 96, 4
    TREE_ROWS = [1, 2, 4]

    def __init__(self, model, shared=True):
        self.model, self.dev, self._wrows, self.graphs, self.shared = causal_sdpa(model), torch.device("cpu"), {}, {}, shared
        self.tok = type("T", (), {"pad_token_id": 0})()
        self.fwd, self.fwd_sfx = self._forward, self._forward_suffix
        self._alloc_kv()
        self.replays = []

    def _capture(self, key, fn, *static, warmup=3):
        r = Replay(fn, static)
        self.graphs[key] = (r, *static, r.out)
        return self.graphs[key]

    def _plain_hidden(self, packs, idx):
        self.replays.append(("plain", len(idx), max(len(packs[k][0]) for k in idx)))
        return super()._plain_hidden(packs, idx)

    def _tree_hidden(self, packs, idx, *cache):
        self.replays.append(("tree", len(idx), max(packs[k][2].index(1) for k in idx)))
        return super()._tree_hidden(packs, idx, *cache)


def prompt(rnd, state, n):
    return state + [rnd.randrange(3, 50) for _ in range(n)]


def groups(rnd):
    s_short, s_long = [1] + [rnd.randrange(3, 50) for _ in range(9)], [1] + [rnd.randrange(3, 50) for _ in range(40)]
    g = []
    g.append(([prompt(rnd, s_short, 3)], [[7, 8]]))                                    # plain, short
    g.append(([prompt(rnd, s_long, 2)], [[7, 8, 9]]))                                   # plain, 43 tokens (graph 48)
    g.append(([prompt(rnd, s_long + s_short, 4)], [[7, 8]]))                            # plain, longer than 48: eager
    q = prompt(rnd, s_short, 2)
    g.append(([q + [10, 11], q + [11, 10]], [[10, 11], [10, 11]]))                       # two orders, masked graph
    q1, q2 = prompt(rnd, s_long, 2), prompt(rnd, s_long, 3)
    g.append(([q1 + [10, 11], q1 + [11, 10], q2 + [12, 13, 14], q2 + [14, 13, 12]],      # SOAM row > 32: two passes
              [[10, 11], [10, 11], [12, 13, 14], [12, 13, 14]]))
    g.append(([prompt(rnd, s_short, 1)] * 2, [[7, 8], [7, 8]]))                          # duplicates: one block
    return g


def check(be, gs):
    m = be.model
    res = be.run_shared(gs)
    for (toks, lids, *_), got in zip(gs, res):
        for r, e in zip(got, separate(m, toks, lids)):
            assert torch.allclose(torch.tensor(r), e, atol=1e-5)
    return res


def test_routing_and_identity():
    rnd = random.Random(0)
    be = Stub(tiny())
    gs = groups(rnd)
    check(be, gs)
    kinds = [r[0] for r in be.replays]
    assert kinds.count("tree") >= 1 and "plain" in kinds
    assert any(r[0] == "plain" and r[2] > 48 for r in be.replays)  # the 55-token prompt: no graph
    assert all(key[2] <= 48 for key in be.graphs)
    assert any(r[0] == "tree" and r[2] > 16 for r in be.replays)    # the long SOAM root (41 tokens) in a graph


def test_long_root_beyond_the_graphs():
    """A branched unit whose root is longer than the largest graph: pass 1 without a graph, then pass 2."""
    rnd = random.Random(3)
    be = Stub(tiny())
    calls = []
    be.fwd = lambda *a: calls.append(a[0].shape) or be._forward(*a)  # the (compiled) forward of the graphs
    s = [1] + [rnd.randrange(3, 50) for _ in range(60)]
    toks = [s + [10, 11], s + [11, 10], s + [5, 6, 12, 13, 14]]
    check(be, [(toks, [[10, 11], [10, 11], [12, 13, 14]])])
    assert ("tree", 1, 61) in be.replays and all(key[2] <= 48 for key in be.graphs)
    assert all(shape[1] <= 48 for shape in calls)  # the long root runs uncompiled (no recompile on long requests)


def test_many_groups_batch_across_requests():
    rnd = random.Random(1)
    be = Stub(tiny())
    gs = [g for _ in range(5) for g in groups(rnd)]
    check(be, gs)
    plain_rows = [r[1] for r in be.replays if r[0] == "plain"]
    assert max(plain_rows) > 1  # single prompts of different groups share a forward


def test_without_sharing_every_prompt_is_plain():
    rnd = random.Random(2)
    be = Stub(tiny(), shared=False)
    check(be, groups(rnd))
    assert {r[0] for r in be.replays} == {"plain"}


def test_plain_forward_builds_no_mask(monkeypatch):
    """With the basal attention name transformers creates no mask for a plain forward (SDPA is_causal), and a 4D mask
    passes through unchanged."""
    import torch.nn.functional as F
    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, is_causal=False, **kw):
        seen.append((attn_mask is None, bool(is_causal)))
        return real(q, k, v, attn_mask=attn_mask, is_causal=is_causal, **kw)
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    be = Stub(tiny())
    be._forward(torch.tensor([[1, 2, 3, 4]]))
    assert seen and all(s == (True, True) for s in seen)
    seen.clear()
    be._eager_shared(GraphBackend._pack([[1, 2, 3], [1, 2, 4]]), [[5, 6], [5, 6]])
    assert seen and all(s == (False, False) for s in seen)


def test_prompt_that_ends_at_the_root():
    """One prompt is the shared prefix of the others: its readout comes from pass 1."""
    rnd = random.Random(4)
    be = Stub(tiny())
    s = [1] + [rnd.randrange(3, 50) for _ in range(12)]
    toks = [s, s + [10, 11], s + [11, 10]]
    check(be, [(toks, [[7, 8], [10, 11], [10, 11]])])
    assert [r[0] for r in be.replays] == ["tree"]


def test_pass2_graphs():
    """With the pass-2 graphs captured (warm), short branch rows replay a graph; longer ones run without one."""
    class Warm(Stub):
        SUFFIX_LENS = [8, 16]
    rnd = random.Random(5)
    be = Warm(tiny())
    be.warm()
    used = []
    for key, g in be.graphs.items():
        if key[0] == "x":
            g[0].replay = (lambda r, k: lambda: (used.append(k), Replay.replay(r)))(g[0], key)
    gs = [g for _ in range(3) for g in groups(rnd)]
    s = [1] + [rnd.randrange(3, 50) for _ in range(10)]
    gs.append(([s + list(range(3, 30)), s + list(range(30, 3, -1))], [[7, 8], [7, 8]]))  # 27-token branches: no graph
    check(be, gs)
    assert used and all(k[2] in (8, 16) for k in used)


def test_layered_forward_is_the_forward():
    """The long lane's forward, one decoder layer at a time, is the plain forward exactly (fp32): the same hidden
    states and the same keys / values written for pass 2."""
    be = Stub(tiny())
    ids = torch.tensor([[1] + [random.Random(5).randrange(3, 50) for _ in range(30)]])
    H, D = be.kv_shape
    kv1 = [(torch.zeros(1, H, 31, D), torch.zeros(1, H, 31, D)) for _ in be._kv]
    kv2 = [(torch.zeros(1, H, 31, D), torch.zeros(1, H, 31, D)) for _ in be._kv]
    seen = []
    assert torch.equal(be._forward_layered(ids, kv1, seen.append), be._forward(ids, kv2))
    assert seen == [0, 1] and all(torch.equal(a, b) for x, y in zip(kv1, kv2) for a, b in zip(x, y))


def test_long_lane_answers_as_the_main_lane():
    """run_long (every unit alone, layered, fresh buffers) gives the answers of run_shared and of separate forwards,
    for plain prompts, two orders and SOAM rows, with and without the state cache (a state stored by one lane is a
    hit for the other)."""
    from basal.engine import StateCache
    from tests.test_state_cache import requests
    rnd = random.Random(4)
    be = Stub(tiny())
    gs = groups(rnd)
    main, long = be.run_shared(gs), be.run_long(gs, between=lambda i: None)
    for (toks, lids, *_), a, b in zip(gs, main, long):
        for x, y, e in zip(a, b, separate(be.model, toks, lids)):
            assert torch.allclose(torch.tensor(y), e, atol=1e-5) and torch.allclose(torch.tensor(x), torch.tensor(y),
                                                                                  atol=1e-5)
    be.state_cache = StateCache(1e9)
    s = [1] + [rnd.randrange(3, 50) for _ in range(60)]                  # a root beyond the graphs
    cached = requests(rnd, s)
    first = be.run_long(cached)                                         # misses: the long lane stores the state
    assert be.state_cache.stats()["entries"] == 1
    for again in (be.run_long(cached), be.run_shared(cached)):          # hits in either lane
        for a, b in zip(first, again):
            assert all(torch.allclose(torch.tensor(x), torch.tensor(y), atol=1e-5) for x, y in zip(a, b))
    assert be.state_cache.stats()["hits"] >= 2


def test_code_logits_in_fp32():
    """The option-code logits are computed in fp32 from the (bf16) hidden state, not rounded to bf16 steps."""
    be = Stub(tiny())
    be.model = be.model.to(torch.bfloat16)
    h = (torch.randn(3, be.model.config.hidden_size) * 4).to(torch.bfloat16)
    lids = [[10, 11], [12, 13, 14], [10, 11]]
    W, _, _ = be._rows(lids)
    assert W.dtype == torch.float32
    got = be._readout(h, lids)
    full = h.float() @ be.model.lm_head.weight.float().T
    for row, ids in zip(range(3), lids):
        want = torch.softmax(full[row, ids], -1)
        assert torch.allclose(torch.tensor(got[row]), want, atol=1e-6)


def test_many_long_branches_are_split():
    """A request with many long questions over one state is split into packed rows of at most BRANCH_TOKENS tokens
    below the root (pass 2's dense masks grow with the square of that), and every readout still equals a separate
    forward of its own prompt."""
    rnd = random.Random(6)
    be = Stub(tiny())
    be.BRANCH_TOKENS = 40
    s = [1] + [rnd.randrange(3, 50) for _ in range(20)]
    toks = [s + [rnd.randrange(3, 50) for _ in range(15)] + [10, 11] for _ in range(6)]   # 6 x 17 tokens below the root
    units = be._units([(toks, [[10, 11]] * 6)])
    assert len(units) > 1
    assert all(len(pk[0]) - pk[2].index(1) <= 40 for pk, *_ in units if 1 in pk[2])
    check(be, [(toks, [[10, 11]] * 6)])
