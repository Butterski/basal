"""Task types (experimental, 2.1): classify, verify, compress, extract.

A task question is expanded into the decision types (choice, noul, multi) over the request's state, so it shares the
state (SOAM) with the request's other questions and uses their calibration; its answer is assembled from theirs. Text
fields of `extract` are then copied from the state by copy-only decoding (eager backend): the model may only continue a
run of the state's own tokens, so a value is always exact source text, with its character offsets.

  classify  {"labels": {key: rubric} | [keys], "multi_label": false, "none": false, "max_error": [0.01, 0.05]}
  verify    {"claims": {key: claim} | [claims], "binary": false}
  compress  {"instructions": "<query>", "unit": "sentence" | "line" | "paragraph", "keep_ratio": 0.4 | "threshold": 0.5}
  extract   {"schema": <JSON schema: object properties, enum / oneOf const, boolean, string, number, integer, arrays of enum>}
"""
import json
import math
import re

import torch

from .prompt import lang_of

TASKS = ("classify", "verify", "compress", "extract")
SEP = "::"  # expanded question names: <task question>::<part>
RISKS = (0.01, 0.05)
MAX_UNITS = 600

W = {
    "pl": dict(
        classify="Która kategoria najlepiej opisuje ten tekst?", none="Żadna z powyższych kategorii",
        classify_multi="Które kategorie opisują ten tekst?",
        verify="Twierdzenie: {claim}\nCo źródło mówi o tym twierdzeniu?", supported="Źródło to potwierdza",
        contradicted="Źródło temu przeczy", not_stated="Źródło tego nie mówi",
        verify_binary="Twierdzenie: {claim}\nCzy źródło potwierdza to twierdzenie?",
        compress="Fragment: „{unit}”\nCzy ten fragment zawiera informację potrzebną do odpowiedzi na pytanie: {q}",
        compress_generic="Fragment: „{unit}”\nCzy ten fragment zawiera ważną informację z dokumentu?",
        present="Czy dokument podaje wartość pola: {field}?", enum="Jaka jest wartość pola: {field}?",
        enum_multi="Które wartości ma pole: {field}?", missing="Nie podano w dokumencie", yes="Tak", no="Nie",
        copy="Przepisz z dokumentu dokładnie, znak po znaku, wartość pola: {field}. Odpowiedz w formacie JSON."),
    "en": dict(
        classify="Which label best describes this text?", none="None of the labels above",
        classify_multi="Which labels describe this text?",
        verify="Claim: {claim}\nWhat does the source say about this claim?", supported="The source supports it",
        contradicted="The source contradicts it", not_stated="The source does not say",
        verify_binary="Claim: {claim}\nDoes the source support this claim?",
        compress="Passage: \"{unit}\"\nDoes this passage hold information needed to answer this question: {q}",
        compress_generic="Passage: \"{unit}\"\nDoes this passage hold important information of the document?",
        present="Does the document give the value of the field: {field}?", enum="What is the value of the field: {field}?",
        enum_multi="Which values does the field have: {field}?", missing="Not stated in the document", yes="Yes",
        no="No", copy="Copy from the document, exactly and character by character, the value of the field: {field}. "
                      "Answer in JSON."),
}


def is_task(q):
    return isinstance(q, dict) and q.get("type") in TASKS


def _risks(q):
    r = q.get("max_error", list(RISKS))
    return [float(x) for x in (r if isinstance(r, list) else [r])]


def _labels(spec, name):
    """{key: rubric}, [keys] or [{"name": key, "rubric": ...}] (the ScaleDown-style list) -> {key: rubric or None}."""
    if isinstance(spec, list) and spec and all(isinstance(x, dict) and "name" in x for x in spec):
        return {str(x["name"]): x.get("rubric") or x.get("description") for x in spec}
    if isinstance(spec, list) and spec:
        return {str(k): None for k in spec}
    if isinstance(spec, dict) and spec:
        return {str(k): v for k, v in spec.items()}
    raise ValueError(f"question {name!r}: labels must be a non-empty object {{key: rubric}} or a list of keys")


# --- compress units -------------------------------------------------------------------------------------------------

ABBREV = {"np", "itd", "itp", "tzw", "ul", "nr", "sp", "o.o", "tj", "ok", "godz", "pkt", "art", "ust", "zł", "e.g",
          "i.e", "etc", "mr", "mrs", "dr", "prof", "inc", "ltd", "co", "no", "vs", "st", "jr", "fig", "al"}
SENT_END = re.compile(r"[.!?…]+[\"”»)\]]*\s+(?=[\"„«(\[]?[A-ZĄĆĘŁŃÓŚŹŻ0-9])")


def units(text, unit="sentence"):
    """The text's units as (start, end) character spans, whitespace trimmed: lines, paragraphs (blank-line separated) or
    sentences (sentence ends, blank lines and line breaks before a capital, digit or bullet; common abbreviations and
    single letters do not end a sentence)."""
    if unit not in ("sentence", "line", "paragraph"):
        raise ValueError('unit must be "sentence", "line" or "paragraph"')
    if unit == "line":
        cuts = [m.end() for m in re.finditer(r"\n", text)]
    elif unit == "paragraph":
        cuts = [m.end() for m in re.finditer(r"\n[ \t]*\n", text)]
    else:
        cuts = []
        for m in SENT_END.finditer(text):
            word = re.search(r"([\w.]+)[.!?…]*[\"”»)\]]*$", text[:m.start() + 1].rstrip())
            w = (word.group(1) if word else "").rstrip(".").lower()
            if m.group(0)[0] == "." and (w in ABBREV or len(w) == 1 and w.isalpha()):
                continue
            cuts.append(m.end())
        cuts += [m.end() for m in re.finditer(r"\n[ \t]*\n|\n(?=[ \t]*[A-ZĄĆĘŁŃÓŚŹŻ0-9•\-*–])", text)]
    out, s = [], 0
    for c in sorted(set(cuts)) + [len(text)]:
        a, b = s, c
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
        s = c
    return out


def join_units(text, spans):
    """Kept units in source order; neighbours in the source keep their separator, gaps become a line break or space."""
    parts, prev = [], None
    for k, (a, b) in spans:
        if prev is not None:
            pk, pb = prev
            parts.append(text[pb:a] if k == pk + 1 else ("\n" if "\n" in text[pb:a] else " "))
        parts.append(text[a:b])
        prev = (k, b)
    return "".join(parts)


# --- extract schema -------------------------------------------------------------------------------------------------

def _type(s):
    t = s.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    return t


def _consts(s):
    for key in ("oneOf", "anyOf"):
        alts = s.get(key)
        if alts and all(isinstance(a, dict) and "const" in a for a in alts):
            return [(a["const"], a.get("description") or a.get("title")) for a in alts]
    if "enum" in s:
        return [(v, None) for v in s["enum"] if v is not None]
    return None


def leaves(schema, path=""):
    """A JSON schema -> (path, kind, spec): kind is enum, enum_list, boolean, string, number or integer."""
    if not isinstance(schema, dict):
        raise ValueError(f"extract: schema at {path or 'root'!r} must be an object")
    t = _type(schema)
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        if not props:
            raise ValueError(f"extract: object {path or 'root'!r} has no properties")
        for k, sub in props.items():
            yield from leaves(sub, f"{path}.{k}" if path else str(k))
        return
    if _consts(schema):
        yield path, "enum", schema
    elif t == "boolean":
        yield path, "boolean", schema
    elif t == "array":
        items = schema.get("items") or {}
        if _consts(items):
            yield path, "enum_list", schema
        else:
            raise ValueError(f"extract: field {path!r}: arrays are supported only with enum items (experimental)")
    elif t in ("string", "number", "integer", None):
        yield path, t or "string", schema
    else:
        raise ValueError(f"extract: field {path!r}: type {t!r} is not supported")


def _field_text(path, spec):
    name = path.replace("_", " ").replace(".", " / ")
    d = spec.get("description") or spec.get("title")
    return f"{name} ({d})" if d else name


def _set_path(obj, path, value):
    *head, last = path.split(".")
    for k in head:
        obj = obj.setdefault(k, {})
    obj[last] = value


NUM = re.compile(r"[-+]?\d[\d\s .,']*")


def parse_number(text, integer=False):
    m = NUM.search(text or "")
    if not m:
        return None
    s = re.sub(r"[\s ']", "", m.group(0)).rstrip(".,")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) <= 2 and s.count(",") == 1 else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    if integer:
        return int(v) if v == int(v) else None
    return int(v) if v == int(v) and "." not in s else v


# --- expansion ------------------------------------------------------------------------------------------------------

def expand(body):
    """A request with task questions -> (the request with only decision questions, plans to assemble the answers)."""
    state = body.get("state")
    text = state if isinstance(state, str) else ""
    questions, plans = {}, {}
    for name, q in body["questions"].items():
        if not is_task(q):
            questions[name] = q
            continue
        if SEP in name:
            raise ValueError(f"question {name!r}: a task question's name may not contain {SEP!r}")
        t = q["type"]
        lang = body.get("lang") or lang_of(text + str(q.get("instructions") or ""))
        w = W[lang]
        sub = {}
        if t == "classify":
            labels = _labels(q.get("labels") or q.get("criteria"), name)
            multi = bool(q.get("multi_label"))
            if q.get("none") and not multi:
                labels["none"] = w["none"]
            spec = {"type": "multi" if multi else "choice", "criteria": labels,
                    "instructions": q.get("instructions") or w["classify_multi" if multi else "classify"],
                    "option_keys": q.get("option_keys", "show")}
            if multi:
                spec.update({k: q[k] for k in ("threshold", "min", "max", "objective") if k in q})
            sub["labels"] = spec
            plans[name] = dict(type=t, multi=multi, risks=_risks(q))
        elif t == "verify":
            claims = q.get("claims")
            claims = {str(i): c for i, c in enumerate(claims)} if isinstance(claims, list) else claims
            if not isinstance(claims, dict) or not claims:
                raise ValueError(f"question {name!r}: claims must be a non-empty object or list")
            binary = bool(q.get("binary"))
            pre = (str(q["instructions"]) + "\n") if q.get("instructions") else ""
            for k, c in claims.items():
                c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                if binary:
                    sub[f"c{SEP}{k}"] = {"type": "noul", "instructions": pre + w["verify_binary"].format(claim=c)}
                else:
                    sub[f"c{SEP}{k}"] = {"type": "choice", "instructions": pre + w["verify"].format(claim=c),
                                         "criteria": {"supported": w["supported"], "contradicted": w["contradicted"],
                                                      "not_stated": w["not_stated"]}, "option_keys": "hide"}
            plans[name] = dict(type=t, keys=list(claims), binary=binary, risks=_risks(q))
        elif t == "compress":
            if not text:
                raise ValueError(f"question {name!r}: compress needs a text state")
            unit = q.get("unit", "sentence")
            spans = units(text, unit)
            if not spans:
                raise ValueError(f"question {name!r}: the state has no text to compress")
            if len(spans) > MAX_UNITS:
                raise ValueError(f"question {name!r}: {len(spans)} {unit}s (at most {MAX_UNITS}); use a larger unit "
                                 "or split the document")
            query = str(q.get("instructions") or "").strip()
            for i, (a, b) in enumerate(spans):
                u = text[a:b]
                sub[f"u{SEP}{i}"] = {"type": "noul", "instructions": w["compress"].format(q=query, unit=u) if query
                                     else w["compress_generic"].format(unit=u)}
            keep_ratio, thr = q.get("keep_ratio"), float(q.get("threshold", 0.5))
            if keep_ratio is not None and not 0 < float(keep_ratio) <= 1:
                raise ValueError(f"question {name!r}: keep_ratio must be in (0, 1]")
            plans[name] = dict(type=t, spans=spans, unit=unit, keep_ratio=keep_ratio, threshold=thr,
                               min_keep=int(q.get("min_keep", 1)))
        elif t == "extract":
            fields = list(leaves(q.get("schema") or {}))
            pre = (str(q["instructions"]) + "\n") if q.get("instructions") else ""
            for path, kind, spec in fields:
                f = _field_text(path, spec)
                if kind in ("enum", "enum_list"):
                    vals = _consts(spec if kind == "enum" else spec["items"])
                    crit = {str(v): d for v, d in vals}
                    if kind == "enum":
                        crit["__missing__"] = w["missing"]
                        sub[f"f{SEP}{path}"] = {"type": "choice", "criteria": crit,
                                                "instructions": pre + w["enum"].format(field=f)}
                    else:
                        sub[f"f{SEP}{path}"] = {"type": "multi", "criteria": crit,
                                                "instructions": pre + w["enum_multi"].format(field=f)}
                elif kind == "boolean":
                    sub[f"f{SEP}{path}"] = {"type": "choice", "instructions": pre + w["enum"].format(field=f),
                                            "criteria": {"true": w["yes"], "false": w["no"], "__missing__": w["missing"]},
                                            "option_keys": "hide"}
                else:
                    if not text:
                        raise ValueError(f"question {name!r}: field {path!r} is copied from the state, which has no text "
                                         "(fields from media: enum and boolean only)")
                    sub[f"f{SEP}{path}"] = {"type": "noul", "instructions": pre + w["present"].format(field=f)}
            plans[name] = dict(type=t, fields=fields, lang=lang, pre=pre)
        for k, v in sub.items():
            questions[f"{name}{SEP}{k}"] = v
    return dict(body, questions=questions), plans


# --- assembly -------------------------------------------------------------------------------------------------------

def _automated(cal, q_type, conf, risks):
    from .server import certified
    out = {}
    for r in risks:
        thr = certified(cal, r, q_type)
        out[str(r)] = None if thr is None else bool(conf >= thr)
    return out


def finish(out, body, plans, cal, copy=None):
    """The expanded answers of a response -> one answer per task question. copy(state, field_text, lang) -> span dict,
    for extract's text fields (None: the server cannot decode, and those fields are refused)."""
    ans = out["answers"]
    state = body.get("state") if isinstance(body.get("state"), str) else ""
    for name, p in plans.items():
        part = {k[len(name) + len(SEP):]: ans.pop(k) for k in list(ans) if k.startswith(name + SEP)}
        t = p["type"]
        if t == "classify":
            a = part["labels"]
            if p["multi"]:
                res = {"type": "classify", "labels": a["selected"], "scores": a["probabilities"],
                       "set_confidence": a["set_confidence"]}
            else:
                res = {"type": "classify", "label": a["choice"], "top_label": a["choice"],
                       "scores": a["probabilities"], "confidence": a["confidence"],
                       "automated": _automated(cal, "choice", a["confidence"], p["risks"])}
        elif t == "verify":
            claims, flagged = {}, []
            for k in p["keys"]:
                a = part[f"c{SEP}{k}"]
                if p["binary"]:
                    pr = {"supported": a["noul"], "not_supported": 1 - a["noul"]}
                    verdict = "supported" if a["noul"] >= 0.5 else "not_supported"
                    q_type = "noul"
                else:
                    pr, verdict, q_type = a["probabilities"], a["choice"], "choice"
                conf = max(pr.values())
                claims[k] = {"verdict": verdict, "probabilities": pr, "confidence": conf,
                             "automated": _automated(cal, q_type, conf, p["risks"])}
                if verdict != "supported":
                    flagged.append(k)
            res = {"type": "verify", "claims": claims, "all_supported": not flagged, "not_supported": flagged}
        elif t == "compress":
            spans = p["spans"]
            pr = [part[f"u{SEP}{i}"]["noul"] for i in range(len(spans))]
            order = sorted(range(len(spans)), key=lambda i: -pr[i])
            total = sum(b - a for a, b in spans)
            if p["keep_ratio"] is not None:
                keep, n = set(), 0
                for i in order:
                    if n >= float(p["keep_ratio"]) * total:
                        break
                    keep.add(i)
                    n += spans[i][1] - spans[i][0]
            else:
                keep = {i for i in range(len(spans)) if pr[i] >= p["threshold"]}
            keep |= set(order[:p["min_keep"]])
            kept = sorted(keep)
            text = join_units(state, [(i, spans[i]) for i in kept])
            res = {"type": "compress", "compressed": text, "kept": kept, "unit": p["unit"],
                   "units": [{"start": a, "end": b, "probability": round(x, 6)} for (a, b), x in zip(spans, pr)],
                   "original_chars": len(state), "compressed_chars": len(text),
                   "ratio": round(len(text) / max(1, len(state)), 4)}
        else:  # extract
            data, fields = {}, {}
            for path, kind, spec in p["fields"]:
                a = part[f"f{SEP}{path}"]
                if kind == "enum":
                    vals = {str(v): v for v, _ in _consts(spec)}
                    c = a["choice"]
                    f = {"value": None if c == "__missing__" else vals[c], "probabilities": a["probabilities"],
                         "confidence": a["confidence"], "source": "choice"}
                elif kind == "enum_list":
                    vals = {str(v): v for v, _ in _consts(spec["items"])}
                    f = {"value": [vals[k] for k in a["selected"]], "probabilities": a["probabilities"],
                         "confidence": a["set_confidence"], "source": "choice"}
                elif kind == "boolean":
                    c = a["choice"]
                    f = {"value": None if c == "__missing__" else c == "true", "probabilities": a["probabilities"],
                         "confidence": a["confidence"], "source": "choice"}
                else:
                    present = a["noul"]
                    f = {"value": None, "present_probability": present, "source": "copy"}
                    if present >= 0.5:
                        if copy is None:
                            raise ValueError("extract: text fields need the eager backend (basal-serve --mode eager)")
                        sp = copy(state, p["pre"] + W[p["lang"]]["copy"].format(field=_field_text(path, spec)), p["lang"])
                        f.update(text=sp["text"], start=sp["start"], end=sp["end"],
                                 confidence=present * sp["probability"], copy_probability=sp["probability"])
                        if sp.get("truncated"):
                            f["truncated"] = True
                        if kind in ("number", "integer"):
                            f["value"] = parse_number(sp["text"], kind == "integer")
                            if f["value"] is None:
                                f["error"] = f"not a {kind}"
                        else:
                            f["value"] = sp["text"]
                    else:
                        f["confidence"] = 1 - present
                f["calibration"] = "validated" if f["source"] == "choice" else "unvalidated"
                fields[path] = f
                _set_path(data, path, f["value"])
            res = {"type": "extract", "data": data, "fields": fields}
        res["experimental"] = True
        ans[name] = res
    out["usage"]["questions"] = len(body["questions"])
    return out


# --- copy-only decoding ---------------------------------------------------------------------------------------------

class Copier:
    """Greedy decoding constrained to copy a contiguous run of the state's tokens (eager backend, one at a time).

    The prompt is the chat template over the state and the field instruction, with the assistant prefill
    {"value": " ; each next token must continue an occurrence of the copied run in the state's own tokenization, or
    close the value with a quote token (not before the first token). Probabilities are fp32 softmaxes over the allowed
    tokens; the span's probability is their product."""

    PREFILL = '{"value": "'

    def __init__(self, backend, max_tokens=64):
        self.b, self.max_tokens = backend, max_tokens
        tok = backend.tok
        stop = set()
        for s in ('"', '",', '"}', '" }', '"\n', '"}\n'):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                stop.add(ids[0])
        for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")):
            if isinstance(t, int) and t >= 0 and t != tok.unk_token_id:
                stop.add(t)
        self.stop = sorted(stop)

    def prompt(self, state, instruction):
        msgs = [{"role": "user", "content": f"{state}\n\n{instruction}"}]
        return self.b.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + self.PREFILL

    @torch.no_grad()
    def __call__(self, state, instruction, lang=None):
        from transformers import DynamicCache
        b, tok = self.b, self.b.tok
        enc = tok(state, add_special_tokens=False, return_offsets_mapping=True)
        src, offs = enc["input_ids"], enc["offset_mapping"]
        ids = tok(self.prompt(state, instruction), add_special_tokens=False).input_ids
        chunk = getattr(b, "CHUNK", 4096)
        with b.lock:
            cache, n = DynamicCache(), 0
            for s0 in range(0, len(ids), chunk):
                part = torch.tensor([ids[s0:s0 + chunk]], device=b.dev)
                pos = torch.arange(s0, s0 + part.shape[1], device=b.dev)[None]
                cache = b.model(input_ids=part, position_ids=pos, past_key_values=cache, use_cache=True,
                                logits_to_keep=1).past_key_values
                n = s0 + part.shape[1]
            run, starts, logp, truncated = [], list(range(len(src))), 0.0, True
            for _ in range(self.max_tokens):
                nxt = sorted({src[p + len(run)] for p in starts if p + len(run) < len(src)})
                allowed = nxt + (self.stop if run else [])
                if not allowed:
                    truncated = False
                    break
                pr = b._codes(b._head_in[:, -1, :].float(), [allowed])[0]
                j = max(range(len(allowed)), key=pr.__getitem__)
                t = allowed[j]
                logp += math.log(max(pr[j], 1e-30))
                if t in self.stop and t not in nxt:
                    truncated = False
                    break
                run.append(t)
                starts = [p for p in starts if p + len(run) <= len(src) and src[p + len(run) - 1] == t]
                cache = b.model(input_ids=torch.tensor([[t]], device=b.dev),
                                position_ids=torch.tensor([[n]], device=b.dev), past_key_values=cache, use_cache=True,
                                logits_to_keep=1).past_key_values
                n += 1
        p0 = starts[0]
        a, e = offs[p0][0], offs[p0 + len(run) - 1][1]
        while a < e and state[a].isspace():
            a += 1
        while e > a and state[e - 1].isspace():
            e -= 1
        out = {"text": state[a:e], "start": a, "end": e, "probability": math.exp(logp)}
        if truncated:
            out["truncated"] = True
        return out
