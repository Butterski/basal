"""--deterministic: one queue entry per forward (no cross-request batching), refused for engines that batch on their
own, set before any model is loaded."""
import asyncio
import sys

import pytest

import basal.server as server_module
from basal.server import Server, parser


class Recorder:
    policies = {}

    def __init__(self):
        self.calls = []

    def run_shared(self, groups, policy=None):
        self.calls.append(len(groups))
        return [[[0.5] * len(ids) for ids in g[1]] for g in groups]


def drain(max_batch, n=5):
    srv = Server.__new__(Server)
    srv.backend, srv.max_batch, srv.wait = Recorder(), max_batch, 0.0

    async def go():
        srv.queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        futs = [loop.create_future() for _ in range(n)]
        for f in futs:  # requests already waiting when the worker looks
            await srv.queue.put(([[1, 2, 3]], [[7, 8]], f, None))
        w = loop.create_task(srv.worker())
        await asyncio.gather(*futs)
        w.cancel()
    asyncio.run(go())
    return srv.backend.calls


def test_deterministic_runs_one_request_per_forward():
    assert drain(64) == [5]          # default: everything waiting goes into one engine call
    assert drain(1) == [1] * 5       # --deterministic sets max_batch 1


@pytest.mark.parametrize("mode", ["vllm", "sglang", "ollama", "llamacpp"])
def test_deterministic_refused_for_engines_with_their_own_batching(monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", mode, "--deterministic"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--deterministic needs an engine"):
        server_module.main()


def test_deterministic_flag(monkeypatch):
    assert parser().parse_args(["--deterministic"]).deterministic and not parser().parse_args([]).deterministic
    seen = {}
    monkeypatch.setattr(server_module.torch, "use_deterministic_algorithms", lambda v: seen.update(det=v))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    import basal.engine as engine_module
    monkeypatch.setattr(engine_module, "FP8_FAST_ACCUM", [True])
    server_module.deterministic_setup("graph")
    import os
    assert seen["det"] is True and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert engine_module.FP8_FAST_ACCUM[0] is False  # FP8 GEMMs accumulate in full precision


def test_batch_window_waits_under_load_only():
    """After a forward with several requests the worker waits (up to the window) for as many again; after a single
    request it does not wait."""
    srv = Server.__new__(Server)
    srv.backend, srv.max_batch, srv.wait, srv.window = Recorder(), 64, 0.0, 0.05

    async def go():
        srv.queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        async def send(n, delay=0.0):
            await asyncio.sleep(delay)
            futs = [loop.create_future() for _ in range(n)]
            for f in futs:
                await srv.queue.put(([[1, 2]], [[7, 8]], f, None))
            return futs
        first = await send(4)
        w = loop.create_task(srv.worker())
        await asyncio.gather(*first)
        # the next four arrive one by one shortly after: with the window they still share one forward
        late = [await send(1, 0.002) for _ in range(4)]
        await asyncio.gather(*[f for fs in late for f in fs])
        w.cancel()
    asyncio.run(go())
    assert srv.backend.calls == [4, 4]


class Streaming(Recorder):
    """An engine that reports every group through done() before the call returns, and allows two calls in flight."""
    streams, inflight = True, 2

    def run_shared(self, groups, policy=None, done=None):
        out = super().run_shared(groups, policy)
        for k, r in enumerate(out):
            done(k, r)
        return out


def test_streaming_engine_answers_through_done():
    srv = Server.__new__(Server)
    srv.backend, srv.max_batch, srv.wait, srv.window = Streaming(), 64, 0.0, 0.0

    async def go():
        srv.queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        futs = [loop.create_future() for _ in range(3)]
        for f in futs:
            await srv.queue.put(([[1, 2]], [[7, 8]], f, None))
        w = loop.create_task(srv.worker())
        res = await asyncio.gather(*futs)
        w.cancel()
        return res
    assert asyncio.run(go()) == [[[0.5, 0.5]]] * 3 and srv.backend.calls == [3]
