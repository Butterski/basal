"""Set objectives of `multi`: the expected IoU / F1 / exact-match of every prefix equals brute force over all true
sets, and the chosen set is the best of ALL subsets (not only prefixes) within the size bounds."""
import asyncio
import itertools
import time

import numpy as np
import pytest

from basal.sets import expected_scores, select, validate
from tests.test_server import FakeServer


def score(S, Y, objective):
    if objective == "exact":
        return float(S == Y)
    if not S and not Y:
        return 1.0
    return len(S & Y) / len(S | Y) if objective == "iou" else 2 * len(S & Y) / (len(S) + len(Y))


def brute_expected(p, S, objective):
    e = 0.0
    for y in itertools.product([0, 1], repeat=len(p)):
        pr = np.prod([pi if yi else 1 - pi for pi, yi in zip(p, y)])
        e += pr * score(S, {i for i, yi in enumerate(y) if yi}, objective)
    return e


def cases():
    rng = np.random.default_rng(0)
    out = [[0.5], [0.9, 0.1], [1.0, 0.0, 0.5], [0.5, 0.5, 0.5, 0.5], [0.95, 0.9, 0.2, 0.2, 0.01]]
    for n in (3, 6, 8):
        out += [list(rng.random(n)), list(rng.random(n) ** 4), list(1 - rng.random(n) ** 4)]
    return out


@pytest.mark.parametrize("objective", ["iou", "f1", "exact"])
@pytest.mark.parametrize("p", cases())
def test_prefix_scores_equal_brute_force(p, objective):
    order = sorted(range(len(p)), key=lambda i: -p[i])
    got = expected_scores([p[i] for i in order], objective)
    for k in range(len(p) + 1):
        assert abs(got[k] - brute_expected(p, set(order[:k]), objective)) < 1e-12


@pytest.mark.parametrize("objective", ["iou", "f1", "exact"])
@pytest.mark.parametrize("p", cases())
@pytest.mark.parametrize("bounds", [(None, None), (1, None), (None, 2), (2, 3)])
def test_selected_set_is_the_best_subset(p, objective, bounds):
    lo, hi = bounds
    n = len(p)
    if lo is not None and lo > n:
        return
    probs = {f"l{i}": x for i, x in enumerate(p)}
    sel, exp, applied = select(probs, objective, min=lo, max=hi)
    assert applied is None
    sizes = range(lo or 0, min(n, n if hi is None else hi) + 1)
    best = max(brute_expected(p, set(S), objective) for k in sizes for S in itertools.combinations(range(n), k))
    assert len(sel) in sizes and abs(exp - best) < 1e-12
    assert abs(brute_expected(p, {int(k[1:]) for k in sel}, objective) - exp) < 1e-12
    if objective == "exact" and lo is None and hi is None:
        assert sorted(sel) == sorted(k for k, v in probs.items() if v >= 0.5)  # p >= 0.5 per label


def test_threshold_objective_is_the_15_rule():
    probs = {"a": 0.9, "b": 0.7, "c": 0.1}
    assert select(probs) == (["a", "b"], None, None)
    assert select(probs, max=1) == (["a"], None, None) and select(probs, threshold=0.95, min=1) == (["a"], None, None)


def test_many_labels_against_poisson_binomial_dp_and_fast():
    def pb(ps):
        d = np.array([1.0])
        for x in ps:
            d = np.convolve(d, [1 - x, x])
        return d
    p = np.sort(np.random.default_rng(1).random(255))[::-1]
    iou, f1 = expected_scores(p, "iou"), expected_scores(p, "f1")
    for k in (1, 7, 40, 128, 254, 255):
        A, B = pb(p[:k]), pb(p[k:])
        a, b = np.arange(len(A))[:, None], np.arange(len(B))[None, :]
        assert abs(iou[k] - (A[:, None] * B[None, :] * a / (k + b)).sum()) < 1e-11
        assert abs(f1[k] - (A[:, None] * B[None, :] * 2 * a / (k + a + b)).sum()) < 1e-11
    probs = {f"l{i}": float(x) for i, x in enumerate(p)}
    for objective in ("iou", "f1"):
        select(probs, objective)
        t0 = time.perf_counter()
        for _ in range(20):
            select(probs, objective)
        assert (time.perf_counter() - t0) / 20 < 5e-3  # about 0.2-0.3 ms on a laptop CPU


def test_request_validation():
    keys = ["a", "b", "c"]
    validate("q", "threshold", keys, threshold=0.7, min="2")  # 1.5 handling unchanged
    for bad, msg in [(dict(objective="best"), "objective must be one of"),
                     (dict(objective="iou", threshold=0.7), "threshold applies only"),
                     (dict(objective="f1", min=4), "more than the 3 labels"),
                     (dict(objective="f1", min=2, max=1), "min 2 > max 1"),
                     (dict(objective="iou", max=-1), "non-negative integer"),
                     (dict(objective="iou", exclusive=["z"]), "not among the criteria"),
                     (dict(objective="threshold", exclusive="a"), "list of label keys"),
                     (dict(objective="f1", fallback="z"), "fallback 'z' is not among"),
                     (dict(objective="f1", fallback_threshold=0.3), "needs a fallback"),
                     (dict(objective="f1", fallback="a", fallback_threshold=1.5), "number in 0..1")]:
        objective = bad.pop("objective")
        with pytest.raises(ValueError, match=msg):
            validate("q", objective, keys, **bad)


def multi_server(p_yes):
    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        py = next(v for k, v in p_yes.items() if f"„{k}”" in text)
        return [py, 1 - py] if "A. Tak" in text else [1 - py, py]
    return FakeServer(probs)


LABELS = {"d": "Uszkodzenie", "r": "Zwrot", "i": "Faktura", "k": "Kurier"}
P_YES = {"Uszkodzenie": 0.9, "Zwrot": 0.45, "Faktura": 0.1, "Kurier": 0.4}


def ask(q):
    body = {"state": "Paczka uszkodzona, chcę zwrotu.", "questions": {"t": dict(
        {"type": "multi", "instructions": "Które?", "criteria": LABELS}, **q)}}
    return asyncio.run(multi_server(P_YES)._run(body))["answers"]["t"]


def test_server_multi_objectives():
    old = ask({})
    assert set(old) == {"type", "selected", "probabilities", "threshold", "set_confidence"} and old["selected"] == ["d"]
    for objective in ("iou", "f1", "exact"):
        ans = ask({"objective": objective})
        p = [P_YES[LABELS[k]] for k in LABELS]
        sel, exp, _ = select(dict(zip(LABELS, p)), objective)
        assert ans["selected"] == sel and abs(ans["expected_score"] - exp) < 1e-6 and ans["objective"] == objective
        assert "threshold" not in ans
    assert ask({"objective": "f1"})["selected"] == ["d", "r", "k"]  # two near-0.5 labels pay off under F1
    assert ask({"objective": "exact"})["selected"] == ["d"]
    assert len(ask({"objective": "iou", "min": 2, "max": 2})["selected"]) == 2
    with pytest.raises(ValueError, match="objective must be one of"):
        ask({"objective": "jaccard"})


# -- label constraints -----------------------------------------------------------------------------------------------
def constrained_sets(n, exclusive, fallback, lo, hi):
    """All sets allowed with these constraints: subsets of the other labels within the size bounds, and each
    exclusive label alone (when one label may be the whole set). The fallback is handled by the rule, not here."""
    excl = set(exclusive) | ({fallback} if fallback is not None else set())
    normal = [i for i in range(n) if i not in excl]
    hi = n if hi is None else hi
    out = [set(S) for k in range(lo, min(len(normal), hi) + 1) for S in itertools.combinations(normal, k)]
    if lo <= 1 <= hi:
        out += [{e} for e in exclusive]
    return out


CONSTRAINED = [([0.9, 0.2, 0.7, 0.1], [2], None), ([0.3, 0.25, 0.85, 0.2, 0.1], [2], None),
               ([0.1, 0.05, 0.6, 0.15], [], 2), ([0.45, 0.4, 0.3, 0.2, 0.9], [4], 3),
               ([0.02, 0.03, 0.01, 0.2], [], 3), ([0.5, 0.5, 0.5, 0.5], [0, 1], None)]


@pytest.mark.parametrize("objective", ["iou", "f1", "exact"])
@pytest.mark.parametrize("p,exclusive,fallback", CONSTRAINED)
@pytest.mark.parametrize("bounds", [(None, None), (1, None), (None, 2)])
def test_constrained_selection_is_the_best_allowed_set(p, exclusive, fallback, objective, bounds):
    lo, hi = bounds
    probs = {f"l{i}": x for i, x in enumerate(p)}
    sel, exp, applied = select(probs, objective, min=lo, max=hi, exclusive=[f"l{e}" for e in exclusive],
                               fallback=None if fallback is None else f"l{fallback}")
    cands = constrained_sets(len(p), exclusive, fallback, lo or 0, hi)
    if objective == "exact":  # the normal labels by p >= 0.5, against each exclusive label alone
        normal = [i for i in range(len(p)) if i not in set(exclusive) | {fallback}]
        k = sorted(normal, key=lambda i: -p[i])
        m = int(np.clip(sum(p[i] >= 0.5 for i in normal), lo or 0, len(normal) if hi is None else min(hi, len(normal))))
        cands = [set(k[:m])] + ([{e} for e in exclusive] if (lo or 0) <= 1 <= (hi or 99) else [])
    best_S = max(cands, key=lambda S: brute_expected(p, S, objective))
    best = brute_expected(p, best_S, objective)
    if fallback is not None and not best_S and (lo or 0) <= 1 <= (hi or 99):  # nothing else: the fallback may win
        best = max(best, brute_expected(p, {fallback}, objective))
    got = {int(k[1:]) for k in sel}
    assert abs(exp - best) < 1e-12 and abs(brute_expected(p, got, objective) - exp) < 1e-12
    assert applied["chosen"] in ("labels", "exclusive", "fallback", "empty")
    if len(got & set(exclusive) | got & {fallback}):
        assert len(got) == 1  # a constrained label is only ever selected alone


def test_constraints_under_threshold():
    probs = {"damage": 0.4, "refund": 0.3, "general": 0.8, "other": 0.2}
    sel, score, applied = select(probs, exclusive=["general"])
    assert sel == ["general"] and score is None and applied["chosen"] == "exclusive"
    probs["damage"] = 0.9  # a more probable specific label beats the exclusive one
    assert select(probs, exclusive=["general"])[0] == ["damage"]
    probs = {"damage": 0.1, "refund": 0.2, "other": 0.35}
    assert select(probs, fallback="other")[0] == []                                  # 0.35 < threshold 0.5
    assert select(probs, fallback="other", fallback_threshold=0.3)[0] == ["other"]
    assert select(probs, fallback="other", min=1)[0] == ["refund"]                   # min 1: never empty
    probs = {"damage": 0.7, "other": 0.9}
    assert select(probs, fallback="other")[0] == ["damage"]                          # something else is selected


def test_fallback_threshold_decides_under_set_objectives():
    probs = {"a": 0.05, "b": 0.04, "general": 0.3}
    sel, exp, applied = select(probs, "iou", fallback="general")
    assert sel == [] and applied["chosen"] == "empty"            # P(nothing) beats general alone
    sel, exp, applied = select(probs, "iou", fallback="general", fallback_threshold=0.25)
    assert sel == ["general"] and applied["chosen"] == "fallback" and applied["fallback_threshold"] == 0.25
    assert abs(exp - brute_expected([0.05, 0.04, 0.3], {2}, "iou")) < 1e-12


def test_server_multi_constraints_and_long_descriptions():
    scope = ("Zgłoszenia dotyczące uszkodzeń towaru w transporcie, w tym pęknięć, wgnieceń i zalania, gdy klient "
             "zgłasza je w ciągu 14 dni od doręczenia i dołącza zdjęcia opakowania oraz produktu. ") * 3  # ~460 chars
    labels = {"d": scope, "r": "Zwrot", "g": "Ogólne"}
    p_yes = {scope: 0.3, "Zwrot": 0.2, "Ogólne": 0.7}

    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        py = next(v for k, v in p_yes.items() if f"„{k}”" in text)
        return [py, 1 - py] if "A. Tak" in text else [1 - py, py]
    body = {"state": "Paczka.", "questions": {"t": {"type": "multi", "instructions": "Które?", "criteria": labels,
                                                    "objective": "f1", "exclusive": ["g"]}}}
    ans = asyncio.run(FakeServer(probs)._run(body))["answers"]["t"]
    assert ans["selected"] == ["g"] and ans["constraints_applied"] == {"exclusive": ["g"], "fallback": None,
                                                                       "chosen": "exclusive"}
    body["questions"]["t"].update(objective="threshold", exclusive=None, fallback="g")
    ans = asyncio.run(FakeServer(probs)._run(body))["answers"]["t"]
    assert ans["selected"] == ["g"] and ans["constraints_applied"]["chosen"] == "fallback" and "threshold" in ans
    del body["questions"]["t"]["fallback"]
    assert "constraints_applied" not in asyncio.run(FakeServer(probs)._run(body))["answers"]["t"]
