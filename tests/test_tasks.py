"""Task types (basal/tasks.py) without a model: units, numbers, schema leaves, the four tasks through Server.decide with
a stub backend, and copy-only decoding with a character-level stand-in model."""
import asyncio
import re

import pytest
import torch

from basal import tasks
from basal.tasks import Copier, join_units, leaves, parse_number, units

from test_server import FakeServer


def by_text(pref):
    """Position probabilities from option texts: every option line "A. text" of the prompt gets pref(text) (default
    1), normalised over the shown options, so both option orders agree on the canonical probabilities."""
    def f(prompt, ids):
        s = "".join(map(chr, prompt))
        opts = re.findall(r"^[A-J]\. (.*)$", s, re.M)[-len(ids):]
        w = [pref(s, o) for o in opts]
        return [x / sum(w) for x in w]
    return f


def run(srv, body):
    return asyncio.run(srv._run(body))


def test_units_sentences_lines_paragraphs():
    t = "Faktura nr 12 z dnia 3.10.2026. Sprzedawca: ACME Sp. z o.o., ul. Długa 5. Termin płatności 14 dni!\n\nUwagi: brak."
    s = [t[a:b] for a, b in units(t)]
    assert s == ["Faktura nr 12 z dnia 3.10.2026.", "Sprzedawca: ACME Sp. z o.o., ul. Długa 5.",
                 "Termin płatności 14 dni!", "Uwagi: brak."]
    assert [t[a:b] for a, b in units(t, "paragraph")] == [t.split("\n\n")[0], "Uwagi: brak."]
    u = "a\n\n b \nc"
    assert [u[a:b] for a, b in units(u, "line")] == ["a", "b", "c"]
    sp = units(t)
    assert join_units(t, [(0, sp[0]), (1, sp[1]), (3, sp[3])]) == "Faktura nr 12 z dnia 3.10.2026. Sprzedawca: ACME Sp. " \
                                                                   "z o.o., ul. Długa 5.\nUwagi: brak."


@pytest.mark.parametrize("text,integer,value", [
    ("1 234,56 zł", False, 1234.56), ("1,234.56", False, 1234.56), ("12,5", False, 12.5), ("1.234.567", False, 1234567),
    ("PLN 99", False, 99), ("7 szt.", True, 7), ("2,5", True, None), ("brak", False, None), ("1 000,00", False, 1000)])
def test_parse_number(text, integer, value):
    assert parse_number(text, integer) == value


def test_schema_leaves():
    schema = {"type": "object", "properties": {
        "number": {"type": "string", "description": "invoice number"},
        "seller": {"type": "object", "properties": {"name": {"type": "string"}, "vat_id": {"type": ["string", "null"]}}},
        "method": {"enum": ["transfer", "card", "cash"]},
        "risk": {"oneOf": [{"const": "low", "description": "routine"}, {"const": "high", "description": "check"}]},
        "paid": {"type": "boolean"}, "total": {"type": "number"},
        "tags": {"type": "array", "items": {"enum": ["urgent", "foreign"]}}}}
    assert [(p, k) for p, k, _ in leaves(schema)] == [
        ("number", "string"), ("seller.name", "string"), ("seller.vat_id", "string"), ("method", "enum"),
        ("risk", "enum"), ("paid", "boolean"), ("total", "number"), ("tags", "enum_list")]
    with pytest.raises(ValueError, match="arrays are supported only with enum items"):
        list(leaves({"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "object"}}}}))


def test_classify_single_and_multi():
    srv = FakeServer(by_text(lambda s, o: 8 if "refund" in o else 1))
    a = run(srv, {"state": "I want my money back for the broken kettle.", "questions": {
        "intent": {"type": "classify", "labels": [{"name": "refund", "rubric": "asks for a refund"},
                                                  {"name": "track", "rubric": "asks where the parcel is"}], "none": True},
        "other": {"type": "noul", "instructions": "Is it polite?"}}})
    c = a["answers"]["intent"]
    assert c["label"] == "refund" and c["top_label"] == "refund" and set(c["scores"]) == {"refund", "track", "none"}
    assert abs(c["scores"]["refund"] - 0.8) < 1e-6 and c["experimental"] and "other" in a["answers"]
    assert c["automated"] == {"0.01": None, "0.05": None}  # no calibration file: no certificate
    assert a["usage"]["questions"] == 2
    srv = FakeServer(by_text(lambda s, o: (9 if "refund" in s.split("Does this apply")[-1] else 1) if o == "Yes" else 3))
    c = run(srv, {"state": "Refund the kettle.", "questions": {
        "tags": {"type": "classify", "labels": ["refund", "damage"], "multi_label": True}}})["answers"]["tags"]
    assert c["labels"] == ["refund"] and abs(c["scores"]["refund"] - 0.75) < 1e-6


def test_verify_three_way_and_binary():
    def pref(s, o):
        claim = re.search(r"Claim: (.*)", s).group(1)
        good = {"The total is 120 EUR.": "supports", "Payment is due in 30 days.": "does not say"}[claim]
        return 9 if good in o else 1
    srv = FakeServer(by_text(pref))
    v = run(srv, {"state": "Invoice 7: total 120 EUR.", "questions": {
        "check": {"type": "verify", "claims": ["The total is 120 EUR.", "Payment is due in 30 days."]}}})["answers"]["check"]
    assert v["claims"]["0"]["verdict"] == "supported" and v["claims"]["1"]["verdict"] == "not_stated"
    assert v["all_supported"] is False and v["not_supported"] == ["1"]
    srv = FakeServer(by_text(lambda s, o: 9 if o == "Yes" else 1))
    v = run(srv, {"state": "Invoice 7.", "questions": {"c": {"type": "verify", "claims": {"t": "Total 120."},
                                                            "binary": True}}})["answers"]["c"]
    assert v["claims"]["t"]["verdict"] == "supported" and abs(v["claims"]["t"]["probabilities"]["supported"] - 0.9) < 1e-6


def test_compress_threshold_and_keep_ratio():
    state = "The order was placed on Monday. The weather was sunny. The total is 120 EUR. Our office has a cat."
    srv = FakeServer(by_text(lambda s, o: (9 if any(w in s.split("Passage:")[-1].split("answer")[0]
                                                     for w in ("total", "placed")) else 1) if o == "Yes" else 3))
    q = {"type": "compress", "instructions": "When and for how much was the order placed?"}
    c = run(srv, {"state": state, "questions": {"ctx": q}})["answers"]["ctx"]
    assert c["kept"] == [0, 2] and c["compressed"] == "The order was placed on Monday. The total is 120 EUR."
    assert c["original_chars"] == len(state) and len(c["units"]) == 4 and abs(c["units"][0]["probability"] - 0.75) < 1e-6
    c = run(srv, {"state": state, "questions": {"ctx": dict(q, keep_ratio=0.2)}})["answers"]["ctx"]
    assert len(c["kept"]) == 1


def test_extract_choice_fields_and_refused_copy():
    schema = {"type": "object", "properties": {"method": {"enum": ["transfer", "card"]}, "paid": {"type": "boolean"},
                                               "number": {"type": "string", "description": "invoice number"}}}

    def pref(s, o):
        q = s.split("<user>")[-1]
        if "method" in q:
            return 9 if o.startswith("transfer") else 1
        if "paid" in q:
            return 9 if o == "No" else 1
        return 9 if o == "No" else 1  # number: not present
    srv = FakeServer(by_text(pref))
    e = run(srv, {"state": "Invoice. Pay by transfer.", "questions": {"x": {"type": "extract", "schema": schema}}})
    x = e["answers"]["x"]
    assert x["data"] == {"method": "transfer", "paid": False, "number": None}
    assert x["fields"]["number"]["present_probability"] < 0.5 and x["fields"]["method"]["source"] == "choice"
    srv = FakeServer(by_text(lambda s, o: 9 if o == "Yes" else 1))
    with pytest.raises(ValueError, match="eager backend"):
        run(srv, {"state": "Invoice FV/1.", "questions": {"x": {"type": "extract", "schema": {
            "type": "object", "properties": {"number": {"type": "string"}}}}}})


class CharTok:
    """Character-level tokenizer: one token per character (id = code point), offsets included."""
    eos_token_id, unk_token_id = 1, 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return type("E", (dict,), {"input_ids": property(lambda self: self["input_ids"])})(out)

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def convert_tokens_to_ids(self, t):
        return 0

    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<model>"


class Wants:
    """Stand-in backend whose model wants to write `target` and then close the quote."""
    lock = __import__("threading").Lock()
    CHUNK = 7

    def __init__(self, target):
        self.tok, self.dev, self.target, self.written, self._head_in = CharTok(), "cpu", target, "", torch.zeros(1, 1, 1)
        self.prefilled = False

    def model(self, input_ids, position_ids, past_key_values, use_cache, logits_to_keep):
        if self.prefilled:
            self.written += chr(int(input_ids[0, -1]))
        if position_ids[0, -1] + 1 >= len(self.full):
            self.prefilled = True
        return type("O", (), {"past_key_values": past_key_values})()

    def _codes(self, h, ids_list):
        want = self.target[len(self.written)] if len(self.written) < len(self.target) else '"'
        w = [9.0 if chr(i) == want else 1.0 for i in ids_list[0]]
        return [[x / sum(w) for x in w]]


def test_copy_decoding_is_exact_and_located():
    state = "Seller: ACME. Invoice no. FV/12/2026 of 3 Oct. Order FV/12/2025."
    b = Wants("FV/12/2026")
    cp = Copier(b)
    b.full = b.tok(cp.prompt(state, "Copy the invoice number."), add_special_tokens=False).input_ids
    out = cp(state, "Copy the invoice number.")
    assert out["text"] == "FV/12/2026" and state[out["start"]:out["end"]] == "FV/12/2026"
    assert 0 < out["probability"] < 1 and "truncated" not in out
    b = Wants("FV/99")  # the model wants text that is not in the state: the copy can only follow the state
    b.full = b.tok(cp.prompt(state, "x"), add_special_tokens=False).input_ids
    out = Copier(b)(state, "x")
    assert out["text"] in state and state[out["start"]:out["end"]] == out["text"]


def test_task_name_separator_refused():
    with pytest.raises(ValueError, match="may not contain"):
        tasks.expand({"state": "x", "questions": {"a::b": {"type": "classify", "labels": ["p", "q"]}}})
