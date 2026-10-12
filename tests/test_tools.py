"""The `tool` question type: options from OpenAI / MCP tool definitions plus the extra options, read as a choice (a
yes/no question per tool with "multi"); clear 422 errors for duplicate names and malformed definitions; ranking and
defer-to-human. Synthetic tools only."""
import asyncio
import re

import pytest

from basal.sets import select
from basal.tools import DESCRIPTION_CHARS, EXTRA, option_text, parse
from tests.test_prompt2 import fmt2_server
from tests.test_server import FakeServer

ORDER = {"name": "get_order", "description": "Look up an order by its number.\n  Returns status and items.",
         "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "fields": {"type": "array"}},
                        "required": ["order_id"]}}
REFUND = {"name": "refund", "description": "Refund a paid order.",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"},
                                                          "reason": {"type": "string"}},
                         "required": ["order_id", "amount"]}}
MAIL = {"name": "send_message", "description": "Send a message to the customer."}
TOOLS = [{"type": "function", "function": ORDER}, {"type": "function", "function": REFUND},
         {"type": "function", "function": MAIL}]
STATE = "User: Where is my order 5512? It was paid last week."


def lines(prompt):
    """The option texts of a format-1 prompt in their order."""
    text = "".join(map(chr, prompt))
    block = text.split("Options:\n")[-1].split("\n\nAnswer in")[0]
    return [re.sub(r"^[A-J]\. ", "", s) for s in block.split("\n")]


def choice_probs(weights):
    """Position probabilities from canonical weights {option prefix: weight}, in the prompt's option order."""
    def probs(prompt, ids):
        w = [next(v for k, v in weights.items() if s.startswith(k)) for s in lines(prompt)]
        return [x / sum(w) for x in w]
    return probs


def yes_probs(p_yes):
    """A yes/no branch per tool: P(yes) by the tool name in the branch's question."""
    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        label = text.split('Does this apply: "')[-1]
        py = next(v for k, v in p_yes.items() if label.startswith(k))
        return [py, 1 - py] if "A. Yes" in text else [1 - py, py]
    return probs


def ask(probs, q, state=STATE, cal=None):
    body = {"state": state, "questions": {"t": dict({"type": "tool", "tools": TOOLS}, **q)}}
    return asyncio.run(FakeServer(probs, cal=cal)._run(body))["answers"]["t"]


# -- definitions -------------------------------------------------------------------------------------------------------


def test_tool_definitions_in_every_form():
    """OpenAI Chat Completions, OpenAI Responses, MCP and Anthropic definitions give the same options."""
    forms = [TOOLS,
             [dict(f, type="function") for f in (ORDER, REFUND, MAIL)],
             [{("inputSchema" if k == "parameters" else k): v for k, v in f.items()} for f in (ORDER, REFUND, MAIL)],
             [{("input_schema" if k == "parameters" else k): v for k, v in f.items()} for f in (ORDER, REFUND, MAIL)]]
    want = (["get_order", "refund", "send_message", "answer_directly", "ask_user", "none"],
            ["get_order(order_id, fields?) — Look up an order by its number. Returns status and items.",
             "refund(order_id, amount, reason?) — Refund a paid order.",
             "send_message() — Send a message to the customer.",
             EXTRA["answer_directly"]["en"], EXTRA["ask_user"]["en"], EXTRA["none"]["en"]],
            ["get_order", "refund", "send_message"])
    for tools in forms:
        assert parse("t", {"tools": tools}, "en") == want
    assert parse("t", {"tools": TOOLS, "extra": ["ask_user"]}, "pl")[1][-1] == EXTRA["ask_user"]["pl"]
    assert parse("t", {"tools": TOOLS, "extra": []}, "en")[0] == want[2]
    assert parse("t", {"tools": TOOLS, "multi": True}, "en")[0] == want[2]   # sets: no extra options by default
    assert option_text("ping", None, None) == "ping()"


def test_long_descriptions_are_cut_on_a_word_boundary():
    words = " ".join(f"word{i:03d}" for i in range(100))                       # 799 characters, words of 7
    cut = option_text("t", words, None).removeprefix("t() — ")
    assert cut.endswith("…") and len(cut) <= DESCRIPTION_CHARS + 1 and words.startswith(cut[:-1])
    assert cut[:-1].split()[-1] in words.split()                                # no word is cut in two
    assert option_text("t", words, None, cap=0) == f"t() — {words}"            # 0: no cap
    assert option_text("t", "a" * 400, None, cap=10) == "t() — aaaaaaaaaa…"     # one long word: cut inside it
    assert option_text("t", "short, plain", None, cap=12) == "t() — short, plain"
    assert option_text("t", "one two, three", None, cap=9) == "t() — one two…"  # trailing comma dropped
    long_tool = {"name": "search", "description": words}
    assert parse("t", {"tools": [long_tool, MAIL]}, "en")[1][0] == "search() — " + cut
    assert parse("t", {"tools": [long_tool, MAIL], "description_chars": 0}, "en")[1][0] == f"search() — {words}"
    assert len(parse("t", {"tools": [long_tool, MAIL], "description_chars": 50}, "en")[1][0]) <= 11 + 50 + 1


@pytest.mark.parametrize("q, msg", [
    ({"tools": []}, "non-empty list of tool definitions"),
    ({"tools": None}, "non-empty list of tool definitions"),
    ({"tools": TOOLS + [{"type": "function", "function": ORDER}]}, r"duplicate tool names \['get_order'\]"),
    ({"tools": [ORDER, dict(MAIL, name="none")]}, r"tool names \['none'\] clash with the extra options"),
    ({"tools": [ORDER, "refund"]}, r"tools\[1\] must be an object"),
    ({"tools": [ORDER, {"type": "web_search"}]}, r"tools\[1\]: type 'web_search' is not supported"),
    ({"tools": [ORDER, {"type": "function", "function": "refund"}]}, r"tools\[1\].function must be an object"),
    ({"tools": [ORDER, dict(MAIL, name="send message")]}, "name must be a non-empty string without spaces"),
    ({"tools": [ORDER, {"description": "no name"}]}, "name must be a non-empty string"),
    ({"tools": [ORDER, dict(MAIL, description=["x"])]}, r"\(send_message\): description must be a string"),
    ({"tools": [ORDER, dict(MAIL, parameters="order_id")]}, r"\(send_message\).parameters must be a JSON schema"),
    ({"tools": [ORDER, dict(MAIL, inputSchema={"type": "array"})]}, r"inputSchema: type must be \"object\""),
    ({"tools": [ORDER, dict(MAIL, parameters={"properties": ["to"]})]}, r"properties must be an object"),
    ({"tools": [ORDER, dict(MAIL, parameters={"properties": {"to": "string"}})]}, r"\['to'\] are not schemas"),
    ({"tools": [ORDER, dict(MAIL, parameters={"properties": {"to": {}}, "required": "to"})]},
     "required must be a list of distinct parameter names"),
    ({"tools": [ORDER, dict(MAIL, parameters={"properties": {"to": {}}, "required": ["body"]})]},
     r"required: \['body'\] are not among the properties"),
    ({"tools": TOOLS, "extra": ["escalate"]}, r"unknown extra options \['escalate'\]"),
    ({"tools": TOOLS, "extra": "none"}, "extra must be a list"),
    ({"tools": TOOLS, "rank": 7}, r"rank must be an integer in 1..6"),
    ({"tools": TOOLS, "rank": True}, "rank must be an integer"),
    ({"tools": TOOLS, "multi": "yes"}, "multi must be true or false"),
    ({"tools": TOOLS, "multi": True, "max_error": 0.01}, "max_error is not supported with multi"),
    ({"tools": TOOLS, "max_error": 1}, "max_error must be a number between 0 and 1"),
    ({"tools": TOOLS, "criteria": {"a": "A"}}, "tool takes \"tools\""),
    ({"tools": TOOLS, "description_chars": -1}, "description_chars must be a non-negative integer"),
    ({"tools": TOOLS, "description_chars": "300"}, "description_chars must be a non-negative integer"),
])
def test_malformed_tool_questions_are_refused(q, msg):
    with pytest.raises(ValueError, match=msg):
        parse("t", q, "en")


def test_option_count_follows_the_prompt_format():
    many = [{"name": f"tool_{i}", "description": f"Tool number {i}."} for i in range(8)]
    with pytest.raises(ValueError, match=r"11 options \(tools and extra options\) \(supported: 2..10\)"):
        ask(lambda p, ids: [1 / len(ids)] * len(ids), {"tools": many})
    assert len(ask(lambda p, ids: [1 / len(ids)] * len(ids), {"tools": many, "extra": []})["probabilities"]) == 8

    seen = []

    def uniform(prompt, ids):
        seen.append(len(ids))
        return [1 / len(ids)] * len(ids)
    body = lambda n: {"state": STATE, "questions": {"t": {"type": "tool", "tools": [
        {"name": f"tool_{i}", "description": f"Tool number {i}."} for i in range(n)]}}}
    ans = asyncio.run(fmt2_server(uniform)._run(body(252)))["answers"]["t"]
    assert len(ans["probabilities"]) == 255 and seen == [255, 255]
    with pytest.raises(ValueError, match=r"256 options \(tools and extra options\) \(supported: 2..255\)"):
        asyncio.run(fmt2_server(uniform)._run(body(253)))


# -- single tool -------------------------------------------------------------------------------------------------------


def test_server_chooses_a_tool_and_ranks():
    probs = choice_probs({"get_order(": 6, "refund(": 1, "send_message(": 1, "Answer": 0.5, "Ask": 1, "No ": 0.5})
    ans = ask(probs, {"rank": 3})
    assert ans["tool"] == "get_order" and ans["call"] is True and abs(ans["confidence"] - 0.6) < 1e-6
    assert list(ans["probabilities"]) == ["get_order", "refund", "send_message", "answer_directly", "ask_user", "none"]
    assert abs(sum(ans["probabilities"].values()) - 1) < 1e-6
    assert [r["tool"] for r in ans["ranking"]] == ["get_order", "refund", "send_message"]   # ties in option order
    assert ans["ranking"][0]["probability"] == ans["probabilities"]["get_order"]
    assert "deferred" not in ans and "ranking" not in ask(probs, {})
    ans = ask(choice_probs({"get_order(": 1, "refund(": 1, "send_message(": 1, "Answer": 1, "Ask": 6, "No ": 1}), {})
    assert ans["tool"] == "ask_user" and ans["call"] is False


def test_prompt_shows_the_tools_and_the_extra_options_in_the_request_language():
    shown = []

    def probs(prompt, ids):
        shown.append("".join(map(chr, prompt)))
        return [1 / len(ids)] * len(ids)
    ask(probs, {"extra": ["none", "ask_user"]})
    assert lines([ord(c) for c in shown[0]]) == [
        "get_order(order_id, fields?) — Look up an order by its number. Returns status and items.",
        "refund(order_id, amount, reason?) — Refund a paid order.",
        "send_message() — Send a message to the customer.", EXTRA["none"]["en"], EXTRA["ask_user"]["en"]]
    assert "Question: What should the assistant do next: which tool should it call?" in shown[0]
    shown.clear()
    ask(probs, {"instructions": "Które narzędzie?"}, state="Użytkownik: gdzie jest moje zamówienie?")
    assert "Pytanie: Które narzędzie?" in shown[0] and EXTRA["answer_directly"]["pl"] in shown[0]


def test_max_error_defers_below_the_certified_threshold():
    cal = {"temperature_per_prim": {"choice": 1.0},
           "thresholds": {"0.01": {"confidence": 0.9}, "0.05": {"confidence": None}}}
    sure = choice_probs({"get_order(": 95, "refund(": 1, "send_message(": 1, "Answer": 1, "Ask": 1, "No ": 1})
    unsure = choice_probs({"get_order(": 6, "refund(": 1, "send_message(": 1, "Answer": 0.5, "Ask": 1, "No ": 0.5})
    ans = ask(sure, {"max_error": 0.01}, cal=cal)
    assert ans["tool"] == "get_order" and ans["deferred"] is False and ans["call"] is True
    ans = ask(unsure, {"max_error": 0.01, "rank": 2}, cal=cal)
    assert ans["tool"] is None and ans["deferred"] is True and ans["refused"] == "get_order" and ans["call"] is False
    assert ans["ranking"][0]["tool"] == "get_order"                       # the ranking is still the model's
    for c, target in ((cal, 0.05), (cal, 0.02), ({}, 0.01)):              # no certified threshold: always defer
        ans = ask(sure, {"max_error": target}, cal=c)
        assert ans["deferred"] is True and ans["refused"] == "get_order"
        assert ans["max_error"] == "no certified threshold for this type at this max_error"
    # a calibration file with per-type thresholds: tool questions use the choice threshold only, never the pooled one
    per = {"thresholds": {"0.01": {"confidence": 0.5, "per_type": {"noul": {"confidence": 0.6}}}}}
    ans = ask(sure, {"max_error": 0.01}, cal=per)                         # choice not certified: defers
    assert ans["deferred"] is True and ans["refused"] == "get_order" and ans["max_error"] == "no certified threshold for this type at this max_error"
    per["thresholds"]["0.01"]["per_type"]["choice"] = {"confidence": 0.999}
    ans = ask(sure, {"max_error": 0.01}, cal=per)                         # its own 0.999, not the pooled 0.5
    assert ans["deferred"] is True and "max_error" not in ans
    per["thresholds"]["0.01"]["per_type"]["choice"] = {"confidence": 0.9}
    assert ask(sure, {"max_error": 0.01}, cal=per)["deferred"] is False


def test_tool_question_shares_the_state_with_other_questions():
    srv = FakeServer(lambda p, ids: [1 / len(ids)] * len(ids))
    body = {"state": STATE, "questions": {"next": {"type": "tool", "tools": TOOLS},
                                          "done": {"type": "noul", "instructions": "Is the task complete?"}}}
    r = asyncio.run(srv._run(body))
    assert srv.entries == 1 and r["usage"]["branches"] == 2 and r["answers"]["next"]["type"] == "tool"


# -- set of tools ------------------------------------------------------------------------------------------------------


P_YES = {"get_order(": 0.9, "refund(": 0.6, "send_message(": 0.2, "Ask": 0.3}


def test_set_of_tools():
    ans = ask(yes_probs(P_YES), {"multi": True})
    assert ans["type"] == "tool" and ans["multi"] is True and ans["tools"] == ["get_order", "refund"]
    assert ans["call"] is True and list(ans["probabilities"]) == ["get_order", "refund", "send_message"]
    assert "constraints_applied" not in ans and abs(ans["set_confidence"] - 0.9 * 0.6 * 0.8) < 1e-6
    ans = ask(yes_probs(P_YES), {"multi": True, "objective": "iou", "rank": 2})
    probs = dict(zip(["get_order", "refund", "send_message"], [0.9, 0.6, 0.2]))
    sel, score, _ = select(probs, "iou")
    assert ans["tools"] == sel and abs(ans["expected_score"] - score) < 1e-6
    assert [r["tool"] for r in ans["ranking"]] == ["get_order", "refund"]
    assert ask(yes_probs({"get_order(": 0.1, "refund(": 0.2, "send_message(": 0.3}), {"multi": True})["call"] is False


def test_extra_options_of_a_set_are_chosen_alone():
    ans = ask(yes_probs(P_YES), {"multi": True, "extra": ["ask_user"]})
    assert ans["tools"] == ["get_order", "refund"] and ans["constraints_applied"]["exclusive"] == ["ask_user"]
    ans = ask(yes_probs(dict(P_YES, Ask=0.95)), {"multi": True, "extra": ["ask_user"]})
    assert ans["tools"] == ["ask_user"] and ans["call"] is False
    assert ans["constraints_applied"]["chosen"] == "exclusive"
    with pytest.raises(ValueError, match=r"exclusive labels \['email'\] are not among the criteria"):
        ask(yes_probs(P_YES), {"multi": True, "exclusive": ["email"]})
