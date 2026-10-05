"""Evidence output: span selection from start / end log-probabilities and the request flag in the server."""
import asyncio

import pytest
import torch

from basal.evidence import MAX_SPAN, best_spans
from tests.test_server import FakeServer

STATE = "Umowę zawarto 1 marca 2024 r. Okres wypowiedzenia wynosi trzy miesiące. Czynsz płatny do 10. dnia."


def char_rel(state):  # one "token" per character
    return [(i, i + 1) for i in range(len(state))]


def logp(n, peak):
    x = torch.full((n,), -20.0)
    x[peak] = 0.0
    return x.log_softmax(-1)


def test_best_span_is_verbatim_and_ordered():
    a = STATE.index("Okres")
    b = STATE.index("miesiące") + len("miesiące") - 1
    out = best_spans(logp(len(STATE), a), logp(len(STATE), b), char_rel(STATE), STATE)
    assert out[0]["text"] == "Okres wypowiedzenia wynosi trzy miesiące"
    assert STATE[out[0]["start"]:out[0]["end"]] == out[0]["text"]
    assert out[0]["probability"] > 0.99
    assert all(x["probability"] >= y["probability"] for x, y in zip(out, out[1:]))
    for x, y in [(x, y) for i, x in enumerate(out) for y in out[i + 1:]]:   # no overlaps
        assert x["end"] <= y["start"] or y["end"] <= x["start"]


def test_span_never_ends_before_it_starts_or_exceeds_max():
    n = 300
    ls, le = torch.randn(n).log_softmax(-1), torch.randn(n).log_softmax(-1)
    s = "x" * n
    for x in best_spans(ls, le, char_rel(s), s, top=5):
        assert 0 < x["end"] - x["start"] <= MAX_SPAN


class StubEvidence:
    available = True

    def __init__(self):
        self.calls = []

    def spans(self, state, question, options, lang, limit=None):
        self.calls.append((state, limit))
        i = state.index("trzy miesiące")
        return [dict(text="trzy miesiące", start=i, end=i + 13, probability=0.9)]


def test_request_flag_returns_spans_and_limits_to_the_original_state():
    srv = FakeServer([0.8, 0.2])
    srv.evidence = StubEvidence()
    body = {"state": STATE, "facts": "auto", "questions": {
        "notice": {"type": "noul", "instructions": "Czy okres wypowiedzenia jest dłuższy niż miesiąc?", "evidence": True},
        "other": {"type": "noul", "instructions": "Czy czynsz jest płatny z góry?"}}}
    out = asyncio.run(srv._run(body))
    ev = out["answers"]["notice"]["evidence"]
    assert ev[0]["text"] == "trzy miesiące" and STATE[ev[0]["start"]:ev[0]["end"]] == "trzy miesiące"
    assert "evidence" not in out["answers"]["other"]
    assert srv.evidence.calls[0][1] == len(STATE)          # facts block excluded from the spans


def test_evidence_refused_without_a_head():
    srv = FakeServer([0.8, 0.2])
    srv.evidence = None
    body = {"state": STATE, "questions": {"q": {"type": "noul", "instructions": "Czy?", "evidence": True}}}
    with pytest.raises(ValueError, match="evidence is not available"):
        asyncio.run(srv._run(body))
