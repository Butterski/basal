"""basal-loadtest --determinism: the same requests alone and under load -> identical answers, max |dp| and decision
flips (answers whose top option changed)."""
import argparse
import asyncio
import json

import httpx

from basal import loadtest


def test_flips_and_differences(monkeypatch, tmp_path):
    calls = {"n": 0}

    def handler(request):
        body = json.loads(request.content)
        calls["n"] += 1
        p = 0.6 if calls["n"] <= 2 else (0.4 if body["state"] == "a" else 0.6)  # request "a" flips under load
        return httpx.Response(200, json={"answers": {"q": {"probabilities": {"x": p, "y": 1 - p}}}})
    real = httpx.AsyncClient
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    bodies = [{"state": s, "questions": {}} for s in ("a", "b", "c", "d")]
    a = argparse.Namespace(determinism=2, url="http://test/v1/systemone", concurrency=2, out=str(tmp_path / "d.json"))
    asyncio.run(loadtest.determinism(a, bodies))
    d = json.loads((tmp_path / "d.json").read_text())["determinism"]
    assert d["requests"] == 2 and d["identical"] == 1 and d["decision_flips"] == 1
    assert abs(d["max_abs_diff"] - 0.2) < 1e-9
