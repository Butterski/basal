"""--prefix-cache on CPU: short requests are cut after their template prefix, the prefix comes from the state cache
after the first request, and everything below it runs as pass 2. The pass-2 forwards take more units than the
branched-unit graphs, and a cached root shared by several rows is copied once per layer. Every readout still equals a
separate forward of its own prompt (fp32). --batches fine adds plain-graph row counts. The server's cut rule:
- the template prefix for short requests;
- the state for long ones when states are cached;
- nothing otherwise."""
import random

import torch

from basal.engine import GraphBackend, StateCache
from tests.test_engine import tiny
from tests.test_engine16 import Replay, Stub, check
from tests.test_server import FakeServer


class Prefix(Stub):
    BATCHES, PREFIX_ROWS, PREFIX_SFX, PREFIX_ROOT = [1, 2, 3, 4, 6], [1, 2, 3, 4, 6], [8, 12, 16], 16

    def __init__(self, model):
        super().__init__(model)
        self.prefix, self.state_cache = True, StateCache(1e9)


def short_requests(rnd, template, n):
    """n single-prompt groups (one question, one order) after the same template, cut at the template."""
    return [([template + [rnd.randrange(3, 50) for _ in range(rnd.randrange(4, 12))] + [10, 11]], [[10, 11]],
             len(template)) for _ in range(n)]


def test_cached_template_prefix_gives_the_same_answers():
    rnd = random.Random(7)
    be = Prefix(tiny())
    template = [1] + [rnd.randrange(3, 50) for _ in range(7)]
    check(be, short_requests(rnd, template, 3))       # the first call computes the prefix (pass 1) and stores it
    assert be.state_cache.stats()["entries"] == 1
    seen = len(be.replays)
    check(be, short_requests(rnd, template, 11))      # then every request reads it from the cache
    st = be.state_cache.stats()
    assert st["entries"] == 1 and st["hits"] >= 11
    rows = [n for kind, n, _ in be.replays[seen:]]
    assert all(kind == "tree" for kind, _, _ in be.replays[seen:])
    assert max(rows) > max(be.TREE_ROWS)              # more units per pass-2 forward than the branched-unit graphs


def test_shared_root_copied_once_per_layer(monkeypatch):
    rnd = random.Random(8)
    be = Prefix(tiny())
    template = [1] + [rnd.randrange(3, 50) for _ in range(5)]
    check(be, short_requests(rnd, template, 2))
    copies = []
    orig = torch.Tensor.index_copy_
    monkeypatch.setattr(torch.Tensor, "index_copy_", lambda t, *a: (copies.append(1), orig(t, *a))[1])
    check(be, short_requests(rnd, template, 6))
    assert len(copies) == 2 * len(be._kv)             # keys and values of each layer, once for the six rows


def test_cached_roots_ignore_garbage_in_fresh_buffers(monkeypatch):
    """Every torch.empty returns NaN (what uninitialised memory can hold): the cached-root path still reads only
    memory it wrote or zeroed (before the shared buffers were zeroed, the readouts were NaN whenever the allocator
    handed back memory holding NaN)."""
    empty = torch.empty

    def nan_empty(*a, **k):
        t = empty(*a, **k)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    monkeypatch.setattr(torch, "empty", nan_empty)
    rnd = random.Random(7)
    be = Prefix(tiny())
    template = [1] + [rnd.randrange(3, 50) for _ in range(7)]
    check(be, short_requests(rnd, template, 3))
    check(be, short_requests(rnd, template, 11))
    assert be.state_cache.stats()["hits"] >= 11


def test_prefix_pass2_graphs():
    class Warm(Prefix):
        SUFFIX_LENS = [8, 16]
    rnd = random.Random(9)
    be = Warm(tiny())
    be.warm()
    assert ("x", 6, 12, 16) in be.graphs              # prefix rows / lengths beyond the branched-unit graphs
    used = []
    for key, g in be.graphs.items():
        if key[0] == "x":
            g[0].replay = (lambda r, k: lambda: (used.append(k), Replay.replay(r)))(g[0], key)
    template = [1] + [rnd.randrange(3, 50) for _ in range(6)]
    check(be, short_requests(rnd, template, 2))
    check(be, short_requests(rnd, template, 6))
    assert any(k[1] in (3, 6) or k[2] == 12 for k in used)


def test_fine_batches_add_plain_graphs():
    class Fine(Stub):
        FINE_BATCHES = [1, 2, 3, 4, 5]
    be = Fine(tiny())
    be.BATCHES = be.FINE_BATCHES
    assert any(b == 5 for b, _ in be.grid(be.PLAIN_LENS))
    rnd = random.Random(3)
    check(be, [([[1] + [rnd.randrange(3, 50) for _ in range(9)]], [[7, 8]]) for _ in range(5)])
    assert ("plain", 5, 10) in be.replays             # five prompts in one 5-row graph, not padded to more rows


def test_server_cut_rule():
    srv = FakeServer([0.5, 0.5])
    from basal.server import to_items

    class Be:
        PREFIX_SFX, prefix, state_cache = [0], True, StateCache(1e6)
    srv.backend = Be()
    q = to_items("Customer: the parcel arrived damaged.", {"q": {"type": "noul", "instructions": "Damaged?"}})[0]
    branches = [srv.branch(0, 0, q)]
    srv.encode(branches)
    t = srv.template_cut(q, branches[0][3][0])
    Be.PREFIX_SFX = [sum(len(p) - t for p in branches[0][3])]                 # this request is just short enough
    assert srv.cut_of(q, branches) == t
    assert "".join(map(chr, branches[0][3][0][:t])).endswith("State:\n")       # the template prefix
    srv.states_cached = True
    long = to_items("x" * 600, {"q": {"type": "noul", "instructions": "Damaged?"}})[0]
    lb = [srv.branch(0, 0, long)]
    srv.encode(lb)
    assert "".join(map(chr, lb[0][3][0][:srv.cut_of(long, lb)])).endswith("x" * 20)  # long: cut after the state
    srv.states_cached = False
    assert srv.cut_of(long, lb) is None                                            # long, no state cache: no cut
    srv.backend.prefix = False
    assert srv.cut_of(q, branches) is None


def test_options_refused_without_the_graph_engine(monkeypatch):
    import pytest
    import basal.server as server_module
    from basal.server import Server, parser
    monkeypatch.setattr(server_module, "deterministic_setup", lambda kind: None)
    for extra in (["--prefix-cache", "on", "--mode", "eager"], ["--prefix-cache", "on", "--mode", "fast", "--deterministic"],
                  ["--batches", "fine", "--mode", "eager"]):
        a = parser().parse_args(["--model", "x"] + extra)
        with pytest.raises(SystemExit):
            Server(a)
    assert GraphBackend.PREFIX_ROWS[-1] * GraphBackend.PREFIX_SFX[-1] <= GraphBackend.TOKEN_BUDGET
