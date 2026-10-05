# Features beyond choice, yes/no and score

basal-1.5 adds request features on top of the three System One types. They are accepted on `POST /v1/systemone` and
`POST /v1/basal`; clients that use only the official fields are not affected.

| feature | what it does | status |
|---|---|---|
| [SOAM](SOAM.md) | all questions of a request over one shared state | stable, on by default |
| [`facts`](#facts-auto) | computed calendar and money facts appended to a Polish state | stable (the models are trained with it) |
| [`multi`](#multi-which-labels-apply) | independent probability per label and the selected set | stable API; labels are answered as yes/no questions |
| [`act`](#act-cost-aware-actions) | the action with the lowest expected cost, or defer to a person | **experimental** |
| [`evidence`](#evidence-spans) | spans of the state that support the answer | **experimental** |
| [Agents and tool use](#agents-and-tool-use) | next step, tool, ask the user, confirm | uses `choice`, `noul`, `multi`, `act` |

*Experimental* means: the API may change in a minor release, and the measured quality is below what we would call
production-ready; use it with a person in the loop and check it on your own data.

## `facts: "auto"`

Exact arithmetic is a weak spot of one-pass decisions: a model that answers in one forward pass cannot reliably add
14 days to a date or compare two gross amounts. With `"facts": "auto"` the server computes such facts from a Polish
state and appends them, under the header *Fakty pomocnicze (wyliczone automatycznie, bez oceny prawnej)*:

- every date: its weekday, and whether it is a Saturday or a statutory day off (with the holiday's name);
- gaps in days between consecutive dates, and "today minus N days" when the text states today's date;
- date + duration for the durations the text mentions ("14 dni", "2 tygodnie", "6 miesięcy");
- net / gross amounts for a VAT rate the text states (23%, 8%, 5% with "VAT" in the sentence);
- euro × a stated exchange rate, budget minus what was spent, quantity × unit price and list totals, the tier of a
  quantity-tiered price list and a stated delivery fee;
- comparisons of the amounts, as symbols only (`>`, `<`, `=`).

Only arithmetic and calendar facts are added, never a rule ("the deadline is shifted to Monday") or a verdict ("the
amount exceeds the limit"): which fact matters is the model's decision. The models were trained with and without the
block, so they work either way; in development tests on Polish date and amount decisions the block raised accuracy by
about 5 points. Evidence spans never point into the facts block.

```json
{"state": "Towar odebrano 3 marca 2025 r. Oświadczenie o odstąpieniu wysłano 20 marca 2025 r. ...",
 "facts": "auto",
 "questions": {"in_time": {"type": "noul", "instructions": "Czy odstąpienie od umowy złożono w terminie?"}}}
```

Polish only (dates in Polish formats, amounts in zł / PLN / euro). The same code is importable:
`from basal.facts import inject`.

## `multi`: which labels apply?

```json
{"state": "Klient: Laptop zamówiony 3 tygodnie temu przyszedł z pękniętym ekranem, a kurier rzucił paczką pod drzwi. Nie chcę naprawy, chcę zwrotu pieniędzy.",
 "questions": {"tags": {"type": "multi", "instructions": "Które kategorie dotyczą zgłoszenia?",
   "criteria": {"damage": "Towar uszkodzony", "delivery": "Problem z dostawą lub kurierem",
                "refund": "Żądanie zwrotu pieniędzy", "invoice": "Faktura", "repair": "Naprawa gwarancyjna"},
   "max": 3}}}
```

Response fields: `probabilities` (one independent P(applies) per label; they do not sum to 1), `selected` (labels with
probability ≥ `threshold`, default 0.5, at most `max`, at least `min`), `threshold` and `set_confidence` (the
probability that every selected label applies and every other label does not, assuming independence).

- **How it works.** Each label becomes a yes/no branch over the shared state ("Czy dotyczy: „Towar uszkodzony”?" /
  "Does this apply: ..."), answered with the yes/no calibration. With SOAM the cost is one state plus a short branch per
  label.
- **When to use it instead of `choice`.** When several answers can be right at once. A `choice` would have to split
  its probability between two correct labels, which looks like uncertainty but is not.
- **Limits.** The labels are independent questions: the engine does not enforce consistency between them (for example
  mutually exclusive labels). Not available with `evidence`.

## `act`: cost-aware actions

**Experimental.** A plain decision returns probabilities, and the caller still has to turn them into an action,
usually with a hand-picked threshold. `act` does that step with the caller's own costs.

```json
{"state": "Zamówienie #88213, 1 240 zł, dostarczone 6 dni temu. Klientka pisze: odstępuję od umowy, towar nieużywany, w oryginalnym opakowaniu. Regulamin: 14 dni na odstąpienie od umowy od dnia doręczenia.",
 "questions": {"refund": {"type": "act",
   "instructions": "Czy zwrot przysługuje zgodnie z regulaminem?",
   "criteria": {"true": "Przysługuje", "false": "Nie przysługuje"},
   "costs": {"approve": {"true": 0,   "false": 1240},
             "reject":  {"true": 300, "false": 0},
             "human":   {"true": 15,  "false": 15}}}}}
```

For every action the engine computes the expected cost under the calibrated probabilities,
E[cost(a)] = Σ_y p(y) · C[a, y], and returns the cheapest one, with all expected costs, the most likely outcome, the
probabilities and a `calibration` flag. A person (or `defer`) is just another action with a fixed cost, so the engine
automates exactly when automating is cheaper on average than asking someone. In this example an automatic approval
needs P(true) > 1 − 15 / 1,240 = 0.988 and an automatic rejection P(false) > 1 − 15 / 300 = 0.95; anything less
confident goes to the consultant.

- **Short form.** `"costs": {"wrong": 50, "defer": 2}`: answering with any outcome costs 50 when wrong, deferring
  costs 2 (the action names are then the outcome keys and `defer`).
- **`max_error`.** `"max_error": 0.01` also refuses automatic actions below the confidence threshold certified for 1%
  error in the calibration file; the response then carries `refused` (the action that was refused) and the defer
  action (`defer`, `human` or the one named in `"defer_action"`).
- **`calibration`.** `validated` when the calibration file has temperatures and the question is a yes/no question or
  uses `"option_keys": "hide"` (the formats whose calibration we measured); otherwise `unvalidated`, and the expected
  costs are only as good as the probabilities.
- **Measured.** On yes/no test decisions with three cost matrices fixed in advance, `act` was cheaper than a
  confidence threshold chosen on separate data on every matrix of the calibration test split, and on two of three on
  the test split (one tie). It was *not* better than a threshold tuned in hindsight on the evaluation items
  themselves, because the highest confidences are slightly overconfident. That is why it is experimental.

## Evidence spans

**Experimental.** `"evidence": true` on a question returns the spans of the state that support the model's answer:

```json
{"state": "Umowę najmu zawarto 1 marca 2024 r. na czas nieokreślony. § 7. Każda ze stron może wypowiedzieć umowę z zachowaniem trzymiesięcznego okresu wypowiedzenia. § 9. Czynsz jest płatny do 10. dnia miesiąca.",
 "questions": {"notice": {"type": "noul",
   "instructions": "Czy okres wypowiedzenia umowy jest dłuższy niż jeden miesiąc?",
   "evidence": true}}}
```

The answer gets an `evidence` list of up to three non-overlapping spans, sorted by probability, each
`{"text", "start", "end", "probability"}`. `start` / `end` are character offsets into the request's state (into the
compact JSON serialisation when the state is a JSON value), so `state[start:end] == text`: the quote is always
verbatim and can be highlighted in a UI or stored in an audit log.

- **How it works.** A small pointer head, shipped as `evidence_head.pt` next to the weights, scores the start and end
  of a span over the state's tokens, conditioned on the model's state at the answer position, so the span depends on
  the question. It was trained after the decision model on its frozen states, so it never changes a decision. It runs
  on one extra pass per question that asks for evidence (original option order).
- **Cost.** One extra forward pass over the prompt: with basal-1.5 on real documents on one H100 the request latency
  rises from 40 to 85 ms (p50, served) with evidence on (basal-1.5-mini: 25 to 51 ms; basal-1.5-max: 80 to 158 ms).
- **Measured quality** (basal-1.5 served, 200 held-out real documents): every returned span is verbatim; token-F1 of the top
  span against a verified quote is 0.53 (basal-1.5-mini: 0.50; basal-1.5-max: 0.58), so the span often covers only part of the quote or a neighbouring
  sentence.
  Documents also state the same fact several times, so removing one span frequently leaves the information in place;
  treat the span as *where the model looked*, not as proof that it is the only reason.
- **Use in RAG and document QA.** Put the retrieved passages into the state (numbered text or a JSON list), ask the
  typed question ("Do the passages support the claim?", "Which passage answers the question?", "Can it be answered from
  the passages?") with `"evidence": true`, and show or log the returned spans with the decision.
- **Limits.** Trained mostly on Polish documents; English spans are less tested. Spans are at most 80 tokens. Not
  available for `multi`, and only in the modes that run the model in process (`fast`, `fast-nocompile`, `fp8`,
  `nvfp4`, `fast-exit`, `eager`; not `vllm`, `sglang` or the ports). A model without `evidence_head.pt` answers HTTP 422.

## Agents and tool use

basal decides **what** an agent does next; the LLM or code writes free-text arguments. Typical questions over one
state (the user request, the conversation so far, tool results, the tool catalogue or a UI tree):

| question | type |
|---|---|
| Which tool (or step) next? Options: the tools, plus "ask the user" and "answer now" | `choice` |
| Which element of the page to click or fill next? | `choice` (options may be JSON objects) |
| Is a required argument missing? Does the call need human confirmation? Did the tool result complete the task? | `noul` |
| Which tools will the task need? | `multi` |
| Execute, ask for confirmation or hand over, given the cost of a wrong call | `act` |
| How risky is the call? | `score` |

The 1.5 models are trained for action selection: the next step of an agent, a user interface or a workflow under an
explicit operating policy. Werdykt's *action* category measures it: basal-1.5 0.630 (best other system: GPT-6 Astra 1.000).

```json
{"state": {"policy": "Refunds above 500 EUR need the user's confirmation before the refund tool is called.",
           "conversation": ["user: Please refund the 740 EUR I paid for order 5512.", "tool order_lookup: {\"id\": 5512, \"paid\": 740, \"status\": \"delivered\"}"],
           "tools": ["order_lookup", "refund", "send_message"]},
 "questions": {"next": {"type": "choice", "instructions": "What should the agent do next?",
   "criteria": {"refund": "Call refund(5512, 740)", "confirm": "Ask the user to confirm the refund of 740 EUR",
                "lookup": "Call order_lookup again", "answer": "Answer the user without a tool call"}}}}
```

Up to 10 options per question in this release; for larger tool catalogues, shortlist first (for example by embedding
search) and ask basal to choose among the shortlist.

## Availability per engine

All modes run behind the same `basal-serve` HTTP API, so the server-level features work everywhere; what differs is
how the shared state is reused, the calibration file and evidence.

| | basal engine (`fast`, `fp8`, …) | `eager` (CUDA / MPS / CPU) | `vllm` | `sglang` | `mlx` | `ollama` / `llamacpp` |
|---|---|---|---|---|---|---|
| choice / noul / score, `multi`, `act`, `facts` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| two option orders | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| calibration file | `CALIBRATION.json` (validated) | `CALIBRATION.json` | `CALIBRATION.vllm.json` if shipped, else `CALIBRATION.json` (1.5: agreement 0.997) | `CALIBRATION.sglang.json` if shipped, else `CALIBRATION.json` | `CALIBRATION.mlx.json` if shipped, else the bf16 file (not validated) | `CALIBRATION.<mode>.json` if shipped, else the bf16 file (not validated) |
| shared state (SOAM) | one packed row | – (same answers, no speed-up) | prefix caching | RadixAttention | shared-prefix KV cache | the runtime's prompt cache (sequential requests) |
| `evidence` | ✓ | ✓ | – | – | – | – |
| longest prompt | model context | model context | 4,096 tokens by default (`--max-len`) | 4,096 tokens by default (`--max-len`) | model context | `num_ctx` / `-c` (4,096 by default) |
