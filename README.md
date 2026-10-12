# basal 2.0

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basal--2.0%20models-yellow)](https://huggingface.co/collections/Remek/basal-20-6acd1712a8e443c86cd576fc)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)
[![basal-1.0 technical report](https://img.shields.io/badge/basal--1.0%20technical%20report-PDF-b31b1b.svg)](docs/basal-1.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022986.svg)](https://doi.org/10.5281/zenodo.23022986)

Inference engine and models for **basal-2.0** and **basal-omni-1.0**: small, fast, calibrated *typed-decision* models
for Polish and English. The basal-omni models also take **images and audio** in a request (experimental).

**What it is.** basal is inspired by *System 1* (fast, intuitive) decision models such as Jev: instead of a chatbot
that writes an answer, the model reads a **state** (a message, a document, a case file, retrieved passages, an agent
trace, a web page as JSON) and answers a **typed question** about it by returning a **probability for each allowed
answer**, in one forward pass, without generating text. The answer can never fall outside the options you gave, and
the probability says how sure the model is.

**What it is for: a dynamic classifier.** You describe the classes *in the request*, in plain language, per call,
instead of training a classifier for them. The same model routes tickets today, checks a filing deadline tomorrow and
scores the urgency of an incident report next week, each time with new options and no retraining. Typical uses:

- ticket, e-mail and document **routing** with categories that change often, up to 255 options per question;
- **rule and policy checks** on a document ("is the claim covered?", "was the appeal filed in time?");
- **questions about long real documents** (judgments, contracts, statutes, reports);
- **RAG decisions**: is a passage relevant, do the passages suffice to answer, do sources conflict, is a claim
  supported, or should the system say "cannot be determined";
- **scoring** on ordered scales (urgency, risk, severity, quality of an answer);
- **agent decisions**: the next tool from OpenAI / MCP tool definitions, whether to ask the user, whether a call needs
  confirmation, with **cost-aware actions** that defer to a person when automation is too risky;
- **judging**: is a candidate answer correct and complete, which of two answers is better;
- **triage with a confidence threshold**: accept confident decisions automatically and send the rest to a person.

> The name comes from the *basal ganglia*, the part of the brain that selects one action among competing options.

## What is new in basal-2.0

**Models**
- A family of three: **basal-2.0-mini** (1.5B, fastest), **basal-2.0** (4.5B, main model, with an experimental
  thinking mode) and **basal-2.0-max** (11B, most accurate). Weights in bf16, plus **FP8** (NVIDIA Hopper and
  Blackwell), **GGUF** (Ollama, llama.cpp) and **MLX 8-bit** (Apple Silicon).
- **Prompt format 2**: up to **255 options per question** (basal-1.5: 10), read in one forward pass with **one option
  order** by default (basal-1.5 read two orders). See [docs/FORMATS.md](docs/FORMATS.md).
- On **Werdykt**, the hidden Polish–English decision benchmark, every tier is above its basal-1.5 counterpart (+1.2 /
  +0.7 / +1.1 points for mini / 4.5B / max) with a lower calibration error (0.035 / 0.060 / 0.044 against
  0.086 / 0.068 / 0.074), and faster. Not everything improved: on a panel of 10 public English decision tasks all three
  are below basal-1.5 (−2.1 / −3.6 / −1.8 points). All numbers: [results against basal-1.5](#results-against-basal-15).
- **Calibration** with a temperature per question type and option-count bucket, fitted on question families the model
  was not trained on. It is cautious on new tasks: the shipped files certify a 5% error target but no 1% target (one
  port file excepted), so `max_error: 0.01` defers every decision ([using the confidence](#using-the-confidence)).
- **Thinking mode** (4.5B, *experimental, off by default*): `--think auto` writes a short reasoning note for uncertain
  questions before deciding. On Werdykt it lowered accuracy and calibration and cut throughput; we recommend leaving it
  off ([thinking mode](#thinking-mode-experimental)).

**basal-omni-1.0 (experimental): images and audio**
- Two models on Google Gemma 4: **basal-omni-1.0-mini** (4B, Gemma 4 E4B) and **basal-omni-1.0-max** (12B). They answer
  the same typed questions over the same API, and with `--media on` a request may also carry **images and audio**
  (`"media"`).
- On Werdykt: 0.769 (mini) and 0.823 (max), against 0.721 for basal-1.5 (4.5B) and 0.773 for basal-1.5-max.
- Trained on text decisions only: Gemma 4's own encoders read the images and audio zero-shot, and media answers use
  the text calibration. `--mode eager` only for now. See [images and audio](#images-and-audio-basal-omni-experimental)
  and [docs/OMNI.md](docs/OMNI.md).

**Engine v2.0.0**
- **Faster serving on one GPU**: compact batching (single prompts packed into one row with FlashAttention 3 varlen),
  the prompt template computed once at start-up, a scheduler that keeps long requests from delaying short ones,
  merged projections and FlashAttention 3 on Hopper, native ModelOpt FP8 checkpoints in `--mode fp8`.
- **One engine for text and media**: the eager backend serves Gemma 4 models (basal-omni) and, with `--media on` and
  the `basal[media]` extra, requests with images and audio (`"media"`, `"detail"`, `"lang"`). In every model it now
  reads option codes in fp32, reads left-padded prompts at their own positions and prefills long prompts in
  2,048-token chunks.
- **New question type `tool`**: the next tool (or answer directly / ask the user / no suitable tool) from OpenAI, MCP
  or Anthropic tool definitions, also as a set of tools (*experimental*).
- **MCP for agents** (*experimental*): `basal-serve --mcp on` also serves the decisions as an MCP server at `/mcp`
  (tools `decide` and `decide_many`), and `basal-mcp` connects a running server to agent hosts over stdio
  ([MCP](#mcp-basal-as-a-tool-for-agents)).
- **`multi`** gains set objectives (`iou`, `f1`, `exact`) and `exclusive` / `fallback` labels.
- **`basal-calibrate`**: fit the temperatures and certified confidence thresholds on your own labelled requests.
- **`--deterministic`** serving (answers independent of load), **`--orders 1 | 2 | auto`**, long states truncated in
  the middle instead of refused (`--context-overflow`), option-code logits in fp32 in every backend.
- The engine still serves basal-1.5 and basal-1.0 models (prompt format 1) with the same API.

Full list: [release notes](https://github.com/rkinas/basal/releases/tag/v2.0.0). Details of the experimental features:
[docs/FEATURES.md](docs/FEATURES.md).

## Models

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-2.0-mini** | 1.5B | fastest; high-volume routing, laptops and small GPUs; context 8,192 tokens | [`Remek/basal-2.0-mini`](https://huggingface.co/Remek/basal-2.0-mini) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-mini-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-mini-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-mini-MLX-8bit) |
| **basal-2.0** | 4.5B | main model: best balance of accuracy and speed; experimental thinking mode | [`Remek/basal-2.0-4.5B`](https://huggingface.co/Remek/basal-2.0-4.5B) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-4.5B-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-4.5B-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-4.5B-MLX-8bit) |
| **basal-2.0-max** | 11B | most accurate; long documents, judging, hard English decisions | [`Remek/basal-2.0-max`](https://huggingface.co/Remek/basal-2.0-max) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-max-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-max-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-max-MLX-8bit) |

- **bf16** weights run in this engine (`fast`, `eager`, `mps`, `mlx`) and in vLLM / SGLang. Each repository ships
  `basal.json` (prompt format and readout), `CALIBRATION.json` (one option order) and `CALIBRATION.o2.json` (two
  orders): per-type temperatures and confidence thresholds. The 4.5B repository also ships `think/` and
  `CALIBRATION.think.json` for the thinking mode. The 2.0 models ship no evidence head.
- **`-FP8`**: NVIDIA ModelOpt FP8 checkpoints, served by the basal engine itself (`--mode fp8`, sm89+).
- **`-GGUF`** (`Q8_0`, `Q5_K_M`) for Ollama and llama.cpp, **`-MLX-8bit`** for Apple Silicon.
- Every port is checked against its bf16 model on the development set (agreement of decisions, accuracy, calibration
  error); the FP8, GGUF and MLX ports of all three tiers pass their bars. Each port's card has its numbers, its
  calibration files and, where a port misses a bar, an *experimental* label.
- The basal-2.0 models are fine-tuned from Apache-2.0 Bielik v3 base models (see [NOTICE](NOTICE)) and released under
  Apache-2.0.

**basal-omni-1.0** (experimental, Google Gemma 4; text, images and audio):

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-omni-1.0-mini** | 4B (Gemma 4 E4B) | decisions about text, images and audio; about 16 GB of GPU memory | [`Remek/basal-omni-1.0-mini`](https://huggingface.co/Remek/basal-omni-1.0-mini) |
| **basal-omni-1.0-max** | 12B | highest Werdykt score of the basal models; text, images and audio; about 24 GB of GPU memory | [`Remek/basal-omni-1.0-max`](https://huggingface.co/Remek/basal-omni-1.0-max) |

- bf16 weights, served by `basal-serve --mode eager` on an NVIDIA GPU (`--media on` for images and audio). No FP8,
  NVFP4, GGUF or MLX ports and no thinking mode yet. Fine-tuned from Apache-2.0 Gemma 4 models (see [NOTICE](NOTICE))
  and released under Apache-2.0.

Which one to choose: start with **basal-2.0 (4.5B)**. Use **max** when accuracy on long or hard inputs matters more
than latency, and **mini** for high volumes, laptops and small GPUs (its context is 8,192 tokens). All three use the
same API, prompt format and calibration procedure. Use **basal-omni** (experimental) when a request carries images or
audio, or when the highest accuracy matters more than speed and an eager-mode server is enough.

## Results against basal-1.5

Measured once on the frozen release weights and published as measured, including where basal-2.0 is not better. Each
basal-2.0 model is compared with the basal-1.5 model of the same size. Full tables per model, with intervals and notes,
are on the model cards.

### Werdykt: hidden Polish–English decision benchmark

[Werdykt](https://basal.si5.pl/benchmark/) has 10 categories in Polish and English (5,000 decisions), one call per
item and one protocol for every model; its items are not published. Macro accuracy over the 20 cells:

| | basal-1.5-mini | **basal-2.0-mini** | basal-1.5 (4.5B) | **basal-2.0 (4.5B)** | basal-1.5-max | **basal-2.0-max** |
|---|---|---|---|---|---|---|
| macro [95% CI] | 0.607 [0.594, 0.619] | **0.618** [0.605, 0.631] | 0.721 [0.710, 0.733] | **0.729** [0.717, 0.739] | 0.773 [0.762, 0.783] | **0.784** [0.774, 0.795] |
| 2.0 − 1.5 | | +1.2 pt | | +0.7 pt | | +1.1 pt |
| PL / EN | 0.619 / 0.594 | 0.628 / 0.609 | 0.724 / 0.718 | 0.729 / 0.728 | 0.777 / 0.769 | 0.787 / 0.781 |
| calibration error (ECE, lower is better) | 0.086 | **0.035** | 0.068 | **0.060** | 0.074 | **0.044** |
| automated at 5% target error | 21.9% (4.33% err.) | 25.4% (5.12%) | 50.1% (5.04%) | 51.3% (5.08%) | 65.6% (4.99%) | 65.6% (4.51%) |
| automated at 1% target error | 11.9% (1.66% err.) | 15.2% (1.82%) | 22.5% (1.23%) | 11.9% (0.66%) | 42.5% (0.93%) | 35.3% (1.34%) |
| decisions/s at 16 clients, bf16, one H100 | 20.5 | **106.4** (×5.2) | 23.0 (SGLang) | **32.2** (×1.4) | 10.2 (SGLang) | **16.2** (×1.6) |
| latency at one client (p50) | 14 ms | 16 ms | 34 ms (SGLang) | 25 ms | 67 ms (SGLang) | 37 ms |

basal-1.5 rows are the public board's rows: basal-1.5 (4.5B) and basal-1.5-max served on SGLang (their fastest
configuration), basal-1.5-mini in basal-serve's default mode; basal-2.0 is served by basal-serve v2.0.0 in its default
mode. With both in basal-serve's default mode the 4.5B and max speed-ups are ×4.0 and ×3.5. Each release uses its own
calibration file; accuracy with `--deterministic`. *Automated at X% target error*: a confidence threshold chosen on one
half of the items for that error, applied to the other half (share automated, observed error in brackets). At the 1%
target basal-2.0 (4.5B) and max automate fewer decisions than basal-1.5.

**Where basal-2.0 stands.** On the public board hosted API models and larger open decision systems score above
basal-2.0; the [board](https://basal.si5.pl/benchmark/) lists every tested system with its accuracy, latency and cost.

### Other benchmarks

Both models served by engine v2.0.0 on the same items; differences in points, paired, with 95% bootstrap intervals on
the model cards (* = the interval excludes 0).

| benchmark | mini: 1.5 → 2.0 | 4.5B: 1.5 → 2.0 | max: 1.5 → 2.0 |
|---|---|---|---|
| Decision Index (JDI 0.2.1) | 16.41 → 26.69 | 27.65 → 39.86 | – → 44.07 (no 1.5 index) |
| JDI requests left unanswered | 17.2% → 0.02% | 17.2% → 0.00% | – → 0.00% |
| LEK, Polish medical licensing exam, PL / EN (% correct) | 47.7 / 48.4 → 47.7 / 49.2 | 60.5 / 59.7 → 62.0 / 63.0 | 70.0 / 67.6 → 75.6 / 74.1 |
| PL-Holdout (4,000 items) | −0.8 | +0.6 | −0.6 |
| Polish decisions (7,081 items) | −1.4* | +0.1 | −0.2 |
| PL-Real-300, doc-clean subset (158 items) | 0.0 | 0.0 | +1.3 |
| PL-Real-300, all 300 items | −2.0 | −2.0 | 0.0 |
| English panel (10 public tasks, mean accuracy) | −2.1* | −3.6* | −1.8* |
| Polish general (3,079 items; development) | +2.2* | +1.6* | +1.4* |
| English corpus test (1,479 items; development) | +1.1 | +0.1 | +1.2 |

basal-1.5 answers questions with at most 10 options, so it leaves 17.2% of JDI requests unanswered. Polish general and
the English corpus test were used to choose between training runs, so their numbers are development numbers; the other
benchmarks were not used to select basal-2.0. LEK questions: NASK-PIB Reassess, 2008–2026 sessions (CC BY-NC-SA 4.0),
shown for the basal models only. The PL-Real primary figure is the subset whose source documents no basal training row
came from.

### Speed

One decision = one question, one option order; bf16 unless noted.

| | mini | 4.5B | max |
|---|---|---|---|
| bf16, one H100, Werdykt speed pass: decisions/s at 16 clients; p50 at one client | 106.4; 16 ms | 32.2; 25 ms | 16.2; 37 ms |
| `-FP8`, one H100, port check: decisions/s at 16 clients; p50 at one client | 103.5; 14 ms | 38.6; 29 ms | 19.9; 50 ms |
| `-MLX-8bit`, Apple M4 Pro 24 GB: one question / five questions over one state (p50) | 93 ms / 223 ms | 278 ms / 698 ms | 622 ms / 1,506 ms |

The bf16 and FP8 rows come from different workloads (the Werdykt items and the port check's development items), so
compare within a row, not across them.

Engine v2.0.0 items, each left out in turn (one H100, bf16): compact batching +19% decisions/s at 16 clients, +17% at
32, +8% at one client; the prompt-template cache +20% at 8 clients (mini and max; off for the 4.5B model, see the quick
start). In a mixed load at 32 clients the scheduler answers short requests in 0.81 s instead of 1.38 s (p50) while long
requests take about 7% longer (1.96 s instead of 1.83 s); `--scheduler off` serves in arrival order.

basal-1.5 also shipped FP8 checkpoints served by vLLM, which on long, repeated documents under mixed traffic were much
faster than basal-1.5 in bf16 (model cards, "basal-1.5's other configuration"). basal-2.0 was not measured on that
workload: for such traffic, measure both on your own load before you switch.

## Engines

basal runs on seven engines, all behind the same `basal-serve` HTTP API (`POST /v1/systemone`). Commands below use
basal-2.0 (4.5B); replace the repository name for mini or max.

<!-- engines -->
| engine | `basal-serve` mode | hardware | what it is for | quick start |
|---|---|---|---|---|
| **basal engine** (PyTorch) | `fast` (also `fast-nocompile`); `mps`; `eager --device mps\|cpu` | NVIDIA GPUs (sm80+); Apple Silicon (`mps`); CPU (`eager`) | the primary server: packed requests (SOAM), compact batching, `--deterministic` | `basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off` |
| **basal engine, FP8** | `fp8` | NVIDIA sm89+ (Hopper, Blackwell) | the `-FP8` checkpoints in the same engine (bf16 weights are quantised on load) | `basal-serve --model Remek/basal-2.0-4.5B-FP8 --mode fp8 --template-prefix off` |
| **vLLM** | `vllm` | NVIDIA GPUs | an alternative CUDA server for bf16 weights (the `-FP8` checkpoints are checked in `--mode fp8` only) | `basal-serve --model Remek/basal-2.0-4.5B --mode vllm` (extra `basal[vllm]`) |
| **SGLang** | `sglang` | NVIDIA GPUs | an alternative CUDA server (measured with basal-1.5) | `basal-serve --model Remek/basal-2.0-4.5B --mode sglang` (own environment) |
| **MLX** | `mlx` (default on a Mac with `basal[mlx]`) | Apple Silicon | Macs: the `-MLX-8bit` ports | `basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx` (extra `basal[mlx]`) |
| **Ollama** | `ollama` | CPU, Apple Silicon, consumer GPUs | local and desktop use with the `-GGUF` files (up to 20 options per question) | `basal-serve --model ./basal-2.0-4.5B-GGUF --mode ollama --ollama-model basal-2.0-4.5b:q8_0` |
| **llama.cpp** | `llamacpp`; `gguf` | CPU, Apple Silicon, consumer GPUs | `llama-server` with the `-GGUF` files, or the file in process (`--mode gguf`, extra `basal[gguf]`) | `basal-serve --model ./basal-2.0-4.5B-GGUF --mode llamacpp` |

**`basal-serve` is the decision layer in every row.** vLLM, SGLang, MLX, Ollama and llama.cpp only run the weights;
the typed, calibrated answers (a probability for each of your options, the state shared by several questions,
`multi`, `act`, `tool` and `facts`) come from `basal-serve --mode <engine>` in front of them, with the same HTTP API
in every mode. A plain `ollama run`, `llama-cli` or `mlx_lm.generate` gives free text only. vLLM and SGLang start with
a 4,096-token context (longer states are cut or refused, see `--context-overflow`); raise it with `--max-len` (e.g.
32768 for long documents). Install and first request for each engine: [quick start](#quick-start).
<!-- /engines -->

The basal-omni models (Gemma 4) run in the basal engine's `eager` mode only, on an NVIDIA GPU; `--media on` adds images
and audio ([images and audio](#images-and-audio-basal-omni-experimental)).

## Quick start

### NVIDIA GPU: the basal engine (recommended)

Install with [uv](https://docs.astral.sh/uv/) into a fresh environment (no git needed; `pip install uv` if it is
missing):

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --port 8000
```

- **`--template-prefix off` for the 4.5B model** (bf16 and `-FP8`): the prompt-template cache failed this model's
  check, and its numbers were measured without it. mini and max run with the default (`auto`, on).
- **Install torch first, from the CUDA 12.8 index.** A plain `pip install basal` takes the newest torch from PyPI,
  which may be built for a newer CUDA than your GPU driver ("The NVIDIA driver on your system is too old"). The cu128
  build needs a driver that supports CUDA 12.8 or newer (`nvidia-smi` shows it top right). DGX Spark and B300: use the
  cu130 build ([docs/HARDWARE.md](docs/HARDWARE.md)).
- **Use a fresh environment on cloud GPU images.** Their system Python often ships a `torchvision` built for another
  torch, which breaks `transformers` ("operator torchvision::nms does not exist"). The text models do not need
  torchvision (the `basal[media]` extra for basal-omni installs it).
- The first start in mode `fast` compiles the model and captures CUDA graphs (several minutes); `--mode
  fast-nocompile` starts in seconds. On Hopper, FlashAttention 3 is used when its wheel is installed
  (`uv pip install flash_attn_3 --extra-index-url https://download.pytorch.org/whl/cu128`); compact batching and
  the template cache need it.
- Other models: `--model Remek/basal-2.0-max` or `--model Remek/basal-2.0-mini`.
- From a clone instead: `git clone https://github.com/rkinas/basal && cd basal && uv pip install -e ".[fp8]"`
  (after the torch line above).

Ask questions with curl. Several questions about one state go into one request and share the state:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

Each answer has a probability for every option, the chosen key and the confidence:

```jsonc
{
 "model": "basal-2.0-4.5B",
 "answers": {
  "dept":   {"type": "choice", "choice": "<key>", "probabilities": {"cards": p, "online": p, "loans": p}, "confidence": p},
  "urgent": {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": p}, "confidence": p}
 },
 "usage": {"input_tokens": n, "output_tokens": 0, "questions": 2, "branches": 2, "latency_ms": t}
}
```

To send uncertain decisions to a person instead, ask the same question as an `act` question with a defer action and
`"max_error": 0.05` ([question types](#question-types)), for example with `"option_keys": "hide"` and
`"costs": {"wrong": 19, "defer": 1}`; the answer then says `"deferred": true` when the model is not sure enough.

Python (the package includes a small client):

```python
from basal.client import Basal
b = Basal("http://127.0.0.1:8000")
state = ("Zgłoszenie z oddziału w Gdańsku: od 7:40 nie działa żaden terminal płatniczy, klienci odchodzą od kas, "
         "kolejka na 30 osób. Obejście: tylko gotówka.")
a = b.score(state, "Jak pilne jest to zgłoszenie?",
            ["niska — można zaplanować", "średnia — w ciągu kilku dni", "wysoka — dziś", "krytyczna — natychmiast"])
print(round(a["score"], 2), a["probabilities"])          # expected level 0-3 and the distribution
print(b.yes_no(state, "Czy problem dotyczy płatności kartą?")["noul"])   # P(yes)
```

### FP8 checkpoints (basal engine)

The `-FP8` repositories are NVIDIA ModelOpt FP8 checkpoints that the basal engine loads natively (the same install as
above; GPU with FP8, sm89+):

```bash
basal-serve --model Remek/basal-2.0-mini-FP8 --mode fp8 --port 8000
basal-serve --model Remek/basal-2.0-4.5B-FP8 --mode fp8 --template-prefix off --port 8000
basal-serve --model Remek/basal-2.0-max-FP8 --mode fp8 --port 8000
```

In FP8 the batch can change near-tie decisions; `--deterministic` runs every request alone with full FP8
accumulation. With a bf16 model, `--mode fp8` quantises the weights at load time (torchao) instead. FP8 keeps
FlashAttention 2 (no compact batching or template prefix) also where FlashAttention 3 is installed: the FP8 ports were
checked on that path. `--attn fa3` selects FlashAttention 3 anyway: it is 21–26% faster on an H100 (4.5B, 16 clients:
48.6 vs 39.5 decisions/s) but is not the checked path: 3–8% of decisions then differ from the checked rows, mostly near
ties, and accuracy moved by −0.8 to +1.4 points against them on a development sample.

### vLLM or SGLang with bf16 weights

```bash
# vLLM brings its own torch: a separate environment. vLLM 0.31 brings torch built for CUDA 13.0: NVIDIA driver >= 580
uv venv --python 3.12 --allow-existing ~/basal-vllm && source ~/basal-vllm/bin/activate
uv pip install "basal[vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode vllm --port 8000          # --max-len 32768 for long documents

# SGLang (own environment; its kernels need a CUDA 12.9+ toolkit; it also brings torch for CUDA 13.0: driver >= 580)
uv venv --python 3.12 --allow-existing ~/basal-sglang && source ~/basal-sglang/bin/activate
uv pip install --prerelease=allow "sglang[srt]==0.5.21"
uv pip install --no-deps "basal @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
uv pip install starlette uvicorn httpx accelerate safetensors   # if missing
basal-serve --model Remek/basal-2.0-4.5B --mode sglang --port 8000
```

The basal server, API and calibration stay the same; the backend reads the option codes' log-probabilities (vLLM up
to 255 codes, see [docs/FORMATS.md](docs/FORMATS.md#engines-and-the-number-of-options)). A per-engine calibration file
(`CALIBRATION.<mode>.json`) is used when a model ships one, else `CALIBRATION.json`. The basal-2.0 release numbers
were measured with the basal engine, not with these backends; their agreement with the basal engine was measured on
basal-1.5 ([docs/QUANTIZATION.md](docs/QUANTIZATION.md)).

### Apple Silicon (MLX)

```bash
uv venv --python 3.12 --allow-existing ~/basal-mlx && source ~/basal-mlx/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000
```

The request and response are the same as above. `-MLX-8bit` holds affine 8-bit decoder weights; the token embeddings
and the output head stay in bf16, because the decision is read from the output head. The state of a request is
computed once into a KV cache shared by all its questions. With `basal[mlx]` installed, `mlx` is the default mode on a
Mac; `--mode mlx` also loads a bf16 model directory, and `--mode mps` runs the PyTorch engine on the Mac's GPU
instead ([docs/HARDWARE.md](docs/HARDWARE.md#apple-silicon)). How the ports work: [docs/PORTS.md](docs/PORTS.md).

### Ollama and llama.cpp (GGUF)

The basal server runs in front of Ollama or llama.cpp: it renders basal's own prompt, sends it raw and reads the
log-probabilities of the option codes of one generated token. Ollama 0.12.11 or newer:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install "basal @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"   # also installs the hf CLI
hf download Remek/basal-2.0-4.5B-GGUF --local-dir basal-2.0-4.5B-GGUF
cd basal-2.0-4.5B-GGUF && ollama create basal-2.0-4.5b:q8_0 -f Modelfile.Q8_0 && cd ..
basal-serve --model ./basal-2.0-4.5B-GGUF --mode ollama --ollama-model basal-2.0-4.5b:q8_0 --port 8000
```

`--model` points at the downloaded folder for the tokenizer, chat template and calibration files; Ollama serves the
weights. Ollama returns at most 20 log-probabilities per position, so questions with more than 20 options need
llama.cpp. The llama.cpp server takes the engine's token ids directly:

```bash
llama-server -m basal-2.0-4.5B-GGUF/basal-2.0-4.5B-Q8_0.gguf -c 4096 --port 8080
basal-serve --model ./basal-2.0-4.5B-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --port 8000
```

Without a separate server, `--mode gguf` loads the file in process through llama-cpp-python (Metal, CUDA or CPU; for
CUDA build it first, see [docs/GGUF.md](docs/GGUF.md)):

```bash
uv pip install "basal[gguf] @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
basal-serve --model ./basal-2.0-4.5B-GGUF --mode gguf --gguf basal-2.0-4.5B-GGUF/basal-2.0-4.5B-Q8_0.gguf --port 8000
```

`--mode gguf` uses `CALIBRATION.json` (the bf16 fit); add `--calibration basal-2.0-4.5B-GGUF/CALIBRATION.llamacpp.json`
to use the llama.cpp fit. Context: Ollama mode reads at most 4,096 tokens per prompt (basal sends `num_ctx` 4096, which
overrides the Modelfile) and llama.cpp mode reads `llama-server -c`; longer states are cut in the middle
(`usage.truncated`). For long documents use `llama-server -c 32768` or `--mode gguf`.

Quantisations: `Q8_0` (recommended) and `Q5_K_M` (smaller). Requests of one state are sent one after another, so the
prompt cache of Ollama / llama.cpp computes the state once; `--http-parallel N` (with `OLLAMA_NUM_PARALLEL` /
`llama-server -np`) serves concurrent clients. GGUF is the slow path for batches: it is for desktops and laptops, not
servers. Details: [docs/PORTS.md](docs/PORTS.md), [docs/GGUF.md](docs/GGUF.md).

### CPU

`basal-serve --model Remek/basal-2.0-mini --mode eager --device cpu` runs everywhere, for tests and low volumes.

### Images and audio: basal-omni (experimental)

Install the `media` extra (pillow, soundfile, scipy, torchvision) and serve a basal-omni model in `eager` mode with
`--media on`:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
basal-serve --model Remek/basal-omni-1.0-mini --mode eager --media on --port 8000   # or Remek/basal-omni-1.0-max
```

A request may then carry images and audio with its state; they go where the state text goes, before it:

```bash
IMG=$(base64 < paragon.jpg | tr -d '\n')
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "", "lang": "pl",
  "media": [{"type": "image", "data": "'"$IMG"'"}],
  "questions": {
    "card": {"type": "noul", "instructions": "Czy zapłacono kartą?"},
    "vat":  {"type": "choice", "instructions": "Jaka jest najwyższa stawka VAT na paragonie?",
             "criteria": {"23": "23%", "8": "8%", "5": "5%", "0": "0% lub zwolniony"}}}}'
```

- Images: any format Pillow reads; `"detail": "high"` reads small print (1,120 image tokens instead of 280; slower).
  Audio: WAV, FLAC, OGG or MP3, at most 30 s a clip. At most 8 items and 20 MB each per request.
- Answers carry `"calibration": "text"`: the temperatures and thresholds were fitted on text, so validate them on your
  own media before automating with them. Text requests to the same server take the usual text path.
- Details, memory and what the engine does differently for Gemma 4: [docs/OMNI.md](docs/OMNI.md). Media results:
  the model cards.

## Thinking mode (experimental)

**basal-2.0 (4.5B) only. Off by default, and we recommend leaving it off.** With `--think auto` the model writes a
short reasoning note before it decides, but only for questions whose calibrated confidence is below the type's
certified 5% threshold; the decision is then read out again over the state and the note, with its own calibration file
(`CALIBRATION.think.json`). The note is generated by a vLLM server that basal-serve starts in its own process on the
think weights in the model's `think/` folder, so the server needs vLLM:

```bash
uv pip install "basal[fp8,vllm] @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
# vLLM 0.31 replaces this environment's torch with one built for CUDA 13.0, which needs an NVIDIA driver >= 580;
# or, since vLLM brings its own torch, keep it in a separate environment and pass --think-python <that env's python>

basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --think auto --port 8000                   # on
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --think auto --think-tokens 128 --port 8000   # shorter notes (default 256)
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --port 8000                                 # off (the default: --think off)
```

On a server started without `--think auto` there is no think path, and the server is exactly the one-pass server.
Per request, on a `--think auto` server:

- `"think": "auto"` (the default), `"off"` (a one-pass answer for this request) or `"force"` (every question of a think
  type thinks); `"think_text": true` returns the note;
- a question that thought answers with the second readout's probabilities and confidence plus `"think": {"tokens",
  "ms", "calibration", "no_think": {...}}` (the one-pass answer); every answer carries `"meta": {"think": {"used",
  "tokens"}}`;
- `multi` questions, tool sets and requests whose state was truncated never think.

Measured once on Werdykt, both rows on the same pod (one H100 80GB, default mode, 16 clients for the speed column):

| server | macro | PL / EN | ECE | automated at 5% | questions that thought | latency p50, one client | decisions/s |
|---|---|---|---|---|---|---|---|
| `--think off` | **0.729** | 0.729 / 0.728 | **0.060** | **51.3%** | – | **23 ms** | **31.2** |
| `--think auto` | 0.719 | 0.718 / 0.720 | 0.099 | 47.1% | 56.7% (mean 142 tokens) | 499 ms | 11.5 |

Thinking lowered the macro score by 0.9 points, raised the calibration error and cut throughput to about a third.
Every other number in this README is the default one-pass mode. Try `--think auto` only with your own measurement on
your own data.

## Inference on a JSONL file

`basal-run` sends every line of a JSONL file to a running server (concurrently; the server batches the requests) and
writes one answer per line, in the same order. Start a server first (`basal-serve ...`), then:

```bash
EX=$(python -c "import basal, pathlib; print(pathlib.Path(basal.__file__).parent / 'examples')")   # installed examples
basal-run --input $EX/questions.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone
```

Each input line is either a **simple item** or a **full request** (the formats can be mixed):

```jsonc
// simple item: one question, options as a list; "type" defaults to "choice", "gold" (index) is optional
{"id": "t1", "state": "Klient: od wczoraj nie mogę zalogować się do bankowości ...", "question": "Do którego działu skierować zgłoszenie?",
 "options": ["Reklamacje kart", "Wsparcie bankowości elektronicznej", "Kredyty"], "gold": 1}
// yes/no item: options are [yes-text, no-text]
{"id": "t2", "type": "noul", "state": "...", "question": "Czy odstąpienie złożono w terminie?", "options": ["Tak", "Nie"]}
// full /v1/systemone request: several typed questions about one state
{"id": "t3", "state": {"ticket": "..."}, "questions": {"category": {"type": "choice", "instructions": "...", "criteria": {"complaint": "...", "other": "..."}},
                                                      "urgent": {"type": "noul", "instructions": "..."}}}
```

Simple items are sent with placeholder keys and `"option_keys": "hide"`, so the model sees exactly the option texts.
Each output line contains `id`, the full `answers` (probabilities, confidence) and the server `latency_ms`; simple items
also get `prediction` (index of the chosen option), `option`, `confidence` and, when `gold` is given, `correct`. At the
end `basal-run` prints a summary (items per second, median latency and accuracy if gold labels are present).
Ready-to-run examples (Polish and English) in `basal/examples/` of the repository, also installed with the package
(`$EX` above):

| file | what it shows |
|---|---|
| `choice.jsonl` | routing, document type, amounts, policy rules, sentiment, next step — one option out of 3–4 (with `gold`) |
| `noul.jsonl` | yes/no decisions: deadlines, approval thresholds, phishing, refunds, missing information, alerts (with `gold`) |
| `score.jsonl` | ordered scales: urgency, satisfaction, fraud risk, answer correctness; the answer includes the expected level |
| `complex.jsonl` | full requests: several typed questions about one JSON state (fan-out), a web-agent step with structured options, a deadline decomposed into simple questions, loan triage, a prompt-injection guard |
| `questions.jsonl` | 20 mixed items; with `choice`, `noul` and `score` the default set of `basal-bench` and `basal-loadtest` |
| `types_demo.py` | one request per decision type, sent to a running server, request and response printed |

```bash
for f in choice noul score complex; do basal-run --input $EX/$f.jsonl --output answers_$f.jsonl; done
```

## Serving modes

`basal-serve --mode <mode>`:

| mode | what it does | hardware |
|---|---|---|
| `fast` *(default on CUDA)* | bf16 + `torch.compile` + CUDA graphs + SOAM packing + compact batching | any CUDA GPU (sm80+) |
| `fast-nocompile` | same without compilation (start-up in seconds instead of minutes) | any CUDA GPU |
| `fp8` | the `-FP8` ModelOpt checkpoints natively; bf16 weights quantised to FP8 at load time (torchao) | GPUs with FP8 (sm89+): Hopper, Blackwell |
| `nvfp4` | `fast` with bf16 weights quantised to NVFP4 at load time (torchao, experimental; costs accuracy; not the basal-1.5 / 1.0 `-NVFP4` checkpoints) | Blackwell |
| `vllm` | vLLM backend, readout from the option codes' log-probabilities; bf16 weights (the `-FP8` checkpoints are checked in `--mode fp8` only), and the basal-1.5 / 1.0 `-NVFP4` checkpoints | CUDA GPUs supported by vLLM |
| `sglang` | SGLang backend, same readout | CUDA GPUs supported by SGLang |
| `eager` | plain PyTorch reference; `--device cuda\|mps\|cpu` (default: `BASAL_DEVICE`, else CUDA, else Apple Silicon, else CPU) | any GPU, Apple Silicon (MPS) or CPU |
| `fast-exit` | `fast` + trained early-exit heads, exit policy chosen per request | models that ship `exit_heads/` (basal-1.0 only) |
| `mlx` *(default on Apple Silicon with `basal[mlx]`)* | Apple Silicon with MLX / mlx-lm: the `-MLX-8bit` ports or bf16 weights; shared-prefix KV cache | Apple Silicon |
| `mlx-q8` | `mlx` with 8-bit weights made from bf16 at load time (about half the memory, not faster) | Apple Silicon |
| `mps` | the PyTorch engine on the Mac's GPU: SOAM packing + token-budget batching, no CUDA graphs | Apple Silicon |
| `gguf` | a GGUF file in process through llama.cpp (`--gguf <file>`, extra `basal[gguf]`) | Apple Silicon (Metal), CUDA, CPU |
| `ollama` | the basal server in front of Ollama (GGUF ports; `--ollama-model`, `--ollama-url`) | anything Ollama runs on |
| `llamacpp` | the basal server in front of `llama-server` (GGUF ports; `--llamacpp-url`) | anything llama.cpp runs on |

- Without `--mode`: `fast` on CUDA, `mlx` on Apple Silicon (`mps` without `basal[mlx]`), `eager` otherwise.
- `--calibration file.json` overrides the calibration file (default: `CALIBRATION.<mode>.json` when the model ships
  one for that engine, else `CALIBRATION.json`; `--mode gguf` uses `CALIBRATION.json`; with `--orders 2`,
  `CALIBRATION.o2.json` when shipped). A file applies its thresholds only to the option orders it was fitted with. `basal-calibrate` fits one on your labelled requests
  ([docs/CALIBRATION.md](docs/CALIBRATION.md)). A missing file is refused at start-up.
- **Option orders** (`--orders 1 | 2 | auto`): `1` asks the order given (the default for prompt-format-2 models such as
  basal-2.0, which are trained with random option orders). `2` also asks the reversed order and averages the
  probabilities (the default for basal-1.5 and basal-1.0); both orders share the state and the question text, so the
  second order costs only its options. `auto` asks order 1 and the reversed order only where order 1 is uncertain
  (`--auto-tau`, `--auto-band`). See [docs/FORMATS.md](docs/FORMATS.md#option-orders).
- **SOAM** (`--soam on`, default): every question and option order of a request is one packed row over the state.
  `--soam off` restores the per-question path of basal-1.0 (same answers in fp32). See [docs/SOAM.md](docs/SOAM.md).
- **Start-up**: `fast` compiles and captures CUDA graphs for all input shapes before accepting requests (several
  minutes the first time, about half with a warm compile cache). A request with several branches (option orders,
  questions, `multi` labels) runs in two exact passes: the state once, keeping its keys and values, then all branches
  over them ([docs/SOAM.md](docs/SOAM.md#which-kernel-runs)).
- **Batching**: concurrent requests are batched into the same forward passes and every request is answered as soon as
  its passes are read out. Under load the server waits up to `--batch-window-ms` (default 3) for as many requests as
  the previous batch had; an idle server answers at once. `--max-batch` and `--wait-ms` control batching further.
- **`--deterministic`**: answers do not depend on load or on the other requests: every request runs in its own
  forward passes, never batched with others, with deterministic kernels (FP8 checkpoints: GEMMs with full
  accumulation). Costs about 4% of the throughput under load with one option order; in-process engines only
  (`basal-loadtest --determinism N` checks a running server, including decision flips under load).
- **Long states**: `--context-overflow truncate` (the default) cuts the middle of a state that does not fit the
  context (the first 1/3 and the last 2/3 of the kept tokens, `--truncate-head`; the question and options stay whole)
  and says what was cut in `usage.truncated`; `--context-overflow refuse` returns HTTP 422 instead. The context is
  32,768 tokens for basal-2.0 (4.5B) and max, 8,192 for mini (Ollama mode: 4,096; llama.cpp mode: `llama-server -c`;
  vLLM / SGLang: `--max-len`).
- **Engine v2.0.0 items** (bf16 weights in `--mode fast` on Hopper with FlashAttention 3; each item has a flag,
  `auto` = on):
  - `--batches compact`: single prompts packed back to back into one row (FA3 varlen), graphs per token count, no
    padding to bucket lengths or row counts.
  - `--template-prefix on`: the chat template, system prompt and user header of every prompt language are computed
    once at start-up; compact rows run only their own tokens over them (off for basal-2.0 4.5B, see the quick start).
  - `--scheduler on`: requests longer than `--long-tokens` (2,048) wait in a backlog and join engine calls in
    highest-response-ratio order, at most 8,192 of their tokens per call. It trades long-request latency (about +7% p50)
    for short-request latency; `--scheduler off` serves in arrival order.
  - Answers are not bit-identical to engine v1.5.0's (other GEMM row counts); `--deterministic` stays reproducible on a
    given GPU.
- **Kernels of the CUDA graph engine** (`fast`, `fast-nocompile`, `fp8`, `nvfp4`, `fast-exit`):
  - `--attn auto|fa2|fa3|fa4|cudnn` is the attention of causal forwards. `auto` uses FlashAttention 3 on Hopper when
    its wheel is installed, else torch's built-in FlashAttention 2; FP8 checkpoints and `--mode fp8` stay on
    FlashAttention 2 unless `--attn fa3`.
  - `--merge auto|on|off` merges the q/k/v and gate/up projections into one GEMM each (`auto`: on for bf16 weights,
    off for FP8 checkpoints).
  - `--fp8-accum fast|full`: FP8 GEMM accumulation in the default mode (`--deterministic` always uses `full`).
  - Opt-in, off by default: `--lean on|pass2` (one compiled layer function: faster start-up, slower under load),
    `--tree flex` (a FlexAttention mask for the branches of a request), `--readout-layer on`, `--state-cache-gb N`
    (a state cache across requests), `--prefix-cache`, `--long-lane`. Measurements:
    [docs/HARDWARE.md](docs/HARDWARE.md#kernels-of-the-cuda-graph-engine-one-h100).
- `--max-len N` sets the context length of the `vllm` and `sglang` modes (default 4096): `--max-len 32768` for long
  documents. The `fast` mode is not affected.
- `--log-decisions file.jsonl` appends one line per request with the served probabilities (a propensity log, for later
  use of production feedback).

## Benchmark your GPU

Two tools are installed with the package. Both run out of the box on the bundled examples (44 Polish and English items
with gold answers in `basal/examples/`):

```bash
# offline: latency, throughput and agreement of serving modes (no HTTP); the first mode is the reference
basal-bench --model Remek/basal-2.0-4.5B --modes eager-fp32 fast --out bench.json

# end to end over HTTP against a running server (basal-serve ...)
basal-loadtest --url http://127.0.0.1:8000/v1/systemone
basal-loadtest --url http://127.0.0.1:8000/v1/systemone --synthetic 1500 --concurrency 16   # ~1.5k-token states

# identity gate: every prefix-reuse path of the fast engine against the eager per-prompt reference
basal-bench --identity --model Remek/basal-2.0-4.5B                    # fp32; --dtype bfloat16 as served
```

`basal-loadtest` reports the median and p95 latency of sequential requests and the decisions per second with
concurrent clients (`--concurrency`); `--synthetic N` uses generated items with states of about N tokens. `basal-bench
--identity` runs single prompts, option orders and multi-question requests over one state, alone and batched, and
compares every readout with an eager forward of its own prompt (argmax agreement, max |Δp|).

**Your own data.** 44 examples are enough to check that everything works, not for precise numbers. For numbers that
describe *your* workload, write a JSONL file with a few hundred items in the same simple format (`{"state": ...,
"question": ..., "options": [...], "gold": <index>}`, `gold` optional) with realistic state lengths, and pass it with
`--questions my_items.jsonl`.

## API

The HTTP API implements the *System One* JSON interface (`POST /v1/systemone`, `GET /v1/models`); responses validate
against the official OpenAPI schema (`tests/test_server.py`). Clients that use this subset work by changing the base
URL (the official SDK was not tested). `POST /v1/basal` is the same endpoint; the extensions below (`multi`, `act`,
`tool`, `evidence`, `facts`, `option_keys`, `think`, `early_exit`) are accepted on both.

### Request

`POST /v1/systemone`

```jsonc
{
  "state": "text or JSON",                         // JSON values are serialised compactly into the prompt
  "questions": {
    "<name>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "<description>", ...}},   // or a list of keys
    "<name>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},  // criteria optional
    "<name>": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]},      // lowest first
    "<name>": {"type": "multi",  "instructions": "...", "criteria": {...}, "threshold": 0.5, "min": 1, "max": 3},
    "<name>": {"type": "multi",  "instructions": "...", "criteria": {...}, "objective": "iou", "max": 3,  // or "f1", "exact"
               "exclusive": ["<label>"], "fallback": "<label>", "fallback_threshold": 0.4},              // optional
    "<name>": {"type": "act",    "instructions": "...", "criteria": {...}, "costs": {...}, "max_error": 0.05},
    "<name>": {"type": "tool",   "instructions": "...", "tools": [<OpenAI or MCP tool definitions>],      // instructions optional
               "extra": ["answer_directly", "ask_user", "none"], "rank": 3, "max_error": 0.05,        // all optional
               "description_chars": 300},
    "<name>": {"type": "tool",   "tools": [...], "multi": true, "objective": "f1"},                   // the set of tools
    // optional on every question:
    //   "option_keys": "show" | "hide"   (default "show": options are shown as "key: description")
    //   "evidence": true                 (models with an evidence head, basal-1.5; not for multi; experimental)
  },
  "facts": "off" | "auto",                          // optional, default "off" (Polish calendar / money facts)
  "think": "auto" | "off" | "force",                // optional, --think auto servers only (basal-2.0 4.5B)
  "think_text": true,                               // optional, returns the reasoning note
  "early_exit": "off" | "0.999" | "0.995" | "0.99" | "0.98",  // optional, --mode fast-exit only (basal-1.0)
  "media": [{"type": "image" | "audio", "data": "<base64>"}],  // optional, --media on servers only (basal-omni)
  "detail": "default" | "high",                     // optional, image tokens 280 / 1,120 (with media)
  "lang": "pl" | "en"                               // optional, the prompt language when the state text does not show it
}
```

2–255 options per question for prompt-format-2 models (basal-2.0), 2–10 for prompt-format-1 models (basal-1.0 to 1.5);
`ollama` mode: at most 20 (see [docs/FORMATS.md](docs/FORMATS.md)). Several questions per request are answered over
one shared state (SOAM); asking them in one request is faster than separate requests and gives the same answers.

### Response

```jsonc
{
  "model": "basal-2.0-4.5B",
  "answers": {
    "<choice>": {"type": "choice", "choice": "<key>", "probabilities": {"<key>": p, ...}, "confidence": p},
    "<noul>":   {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": p}, "confidence": p},
    "<score>":  {"type": "score", "score": expected_level, "legend": {"0": "level 0", ...},
                 "probabilities": {"0": p, ...}, "confidence": p},
    "<multi>":  {"type": "multi", "selected": ["<key>", ...], "probabilities": {"<key>": p_applies, ...},
                 "threshold": 0.5, "set_confidence": p},          // with "objective": "objective" and "expected_score"
    "<act>":    {"type": "act", "action": "<action>", "expected_costs": {"<action>": c, ...}, "answer": "<outcome>",
                 "probabilities": {...}, "confidence": p, "calibration": "validated" | "unvalidated"},
                 // with max_error: "deferred" (true: the defer action, "refused" names the automatic one)
    "<tool>":   {"type": "tool", "tool": "<tool name or extra option>", "call": true, "probabilities": {...},
                 "confidence": p, "ranking": [{"tool": "<name>", "probability": p}, ...]},  // ranking with "rank"
                 // with max_error: "deferred" (true: "tool" is null, "refused" names the model's choice)
                 // with "multi": true: "tools": [...] instead of "tool", and the fields of a multi answer
    // with "evidence": true, any of the above (except multi) also has
    //   "evidence": [{"text": "...", "start": 102, "end": 153, "probability": 0.93}, ...]
  },
  "usage": {"input_tokens": n, "output_tokens": 0, "questions": n, "branches": n, "latency_ms": t}
  // "truncated" when the state was cut to the context; with --orders auto also "second_order_branches": n
}
```

- `probabilities` are calibrated (temperatures per question type and option-count bucket from the calibration file);
  `confidence` is the probability of the chosen answer.
- `score` is the expected level (0 = first level); `legend` maps level keys to their texts.
- `branches` counts the question branches of the request: one per question, one per label of a `multi` and one per
  tool of a `tool` question with `"multi": true` (times the option orders).
- Errors (unknown type, too many options, a field not available in the server's mode or model) return HTTP 422 with
  `{"error": "..."}`.
- `GET /v1/models` returns `{"models": [{"name", "description", "release_date", "mode", "early_exit",
  "temperature_per_prim"}]}` (`release_date` is the engine's release date, not the model's); `GET /health` returns
  `{"status": "ok"}`.

### Question types

| type | answer | example |
|---|---|---|
| `choice` | distribution over named options (up to 255) | route a ticket, pick the applicable rule, the intent among 150 |
| `noul` | probability of *yes* (with optional descriptions of both sides) | "was the appeal filed on time?" |
| `score` | distribution over ordered levels + expected level | urgency 0–3, severity, grade of an answer |
| `multi` | independent probability per label + the selected set | "which of these issues does the message raise?" |
| `act` | the action with the lowest expected cost under your cost matrix | approve / reject / send to a person |
| `tool` | the next tool from OpenAI / MCP tool definitions, or answer directly / ask the user / no suitable tool | the next step of an agent |

**`multi`**: each label is asked as a yes/no branch over the shared state ("does label X apply?"). `selected` holds
the labels with probability ≥ `threshold` (default 0.5), at most `max` and at least `min`; `set_confidence` estimates
that the whole set is exactly right. With `"objective": "iou"`, `"f1"` or `"exact"` the engine instead returns the set
with the highest expected IoU, F1 or exact-match probability under the label probabilities (`min` / `max` bound its
size) and its `expected_score`. An `exclusive` label is selected only alone; a `fallback` label ("none of these") is
selected only when no other label is. See [docs/FEATURES.md](docs/FEATURES.md#choosing-the-set-objective).

**`act`**: `criteria` are the possible outcomes (`{"true": ..., "false": ...}` for a yes/no question, or named options),
`costs` gives the cost of every action for every outcome (`{"approve": {"true": 0, "false": 1240}, "reject":
{"true": 300, "false": 0}, "human": {"true": 15, "false": 15}}`), or the short form `{"wrong": 99, "defer": 1}`
(answer with any outcome, wrong costs 99; deferring costs 1). The engine returns the action with the minimum expected
cost under the calibrated probabilities. With `"max_error"` it also refuses automatic actions below the confidence
threshold certified for that error and question type (the response then names the `refused` action and says
`"deferred": true`) and needs a `defer` / `human` action (or `"defer_action": "<name>"`). Without a threshold certified
for the type and target, every automatic action is refused. See [docs/FEATURES.md](docs/FEATURES.md#act-cost-aware-actions).

**`tool`** (*experimental*): `tools` takes OpenAI (Chat Completions or Responses), MCP (`inputSchema`) or Anthropic
(`input_schema`) tool definitions. Each tool becomes an option `name(a, b, c?) — description`, where `?` marks an
optional parameter; descriptions are cut at 300 characters (`"description_chars"`, 0 = no cap). The extra options
(`answer_directly`, `ask_user`, `none`; by default all three) follow the tools. `"rank": k` returns the k most probable
options. `"multi": true` returns the set of tools to call, using the `multi` objectives and constraints. `max_error`
defers to a person, as in `act`. See [docs/FEATURES.md](docs/FEATURES.md#tool-which-tool-next), which also shows a
[step of an agent loop](docs/FEATURES.md#a-step-of-an-agent-loop) in one request.

**Option keys.** By default every described option is shown to the model as `key: description`
(`{"B": "Dispatch office: Krakow"}` → `B: Dispatch office: Krakow`), because only you know whether a key is meaningful
(a service code, a decision name); the server never guesses. If your keys are placeholders (`option_1`, `0`, …) and
the descriptions alone define the options, send `"option_keys": "hide"`: the model then sees the descriptions only.
A key without a description (`"criteria": {"approve": null}` or a list of keys) is shown by itself.

**Evidence** (*experimental*): `"evidence": true` on a question returns up to three spans of the state that support the
answer, with character offsets and a probability. It needs a model that ships `evidence_head.pt` (basal-1.5); the
basal-2.0 models do not ship one, and the server refuses evidence requests for them. See
[docs/FEATURES.md](docs/FEATURES.md#evidence-spans).

**Facts**: `"facts": "auto"` appends a block of computed facts to a Polish state: the weekday of every date and
whether it is a Saturday or a statutory day off, gaps between dates, date + stated durations, net/gross amounts for
a stated VAT rate, euro × a stated exchange rate, budget remainders, quantity × price and list totals, and
comparisons of amounts (as symbols only, never a verdict). Only arithmetic and calendar facts are added, never a rule
or an answer.

### Using the confidence

The server applies the per-type temperatures of the calibration file (for basal-2.0 also per option-count bucket:
2–10, 11–26, 27–255 options). The file also contains confidence thresholds chosen on calibration data, before testing,
for a target error of 1% and 5% among accepted decisions (`thresholds` in `CALIBRATION.json`): accept decisions above
the threshold automatically and route the rest to a person, either in your code or with an `act` / `tool` question and
`"max_error": 0.05`, which defers below the certified threshold of the question's type.

The basal-2.0 calibration was fitted on question families the models were not trained on, so it holds on new tasks
but is cautious. Held out by question family (leave one family out):

| | mini | 4.5B | max |
|---|---|---|---|
| calibration error (ECE) | 0.017 | 0.015 | 0.022 |
| 5% target threshold: share automated (observed error) | 10.5% (2.94%) | 33.2% (2.39%) | 41.9% (2.51%) |
| 1% target threshold in the shipped file | not certified | not certified | not certified |

No 1% threshold is certified in the shipped files, except in `CALIBRATION.ollama.json` of `basal-2.0-max-GGUF`,
which certifies one for `choice` (confidence 0.989), so `"max_error": 0.01` defers every other decision: use 0.05, or
certify your own thresholds on your data. The ports use per-engine files where the card lists one
(`CALIBRATION.<engine>.json`), else the bf16 file, which is not validated for them.

**Calibrate on your own data.** On your own domain, refit the temperatures and thresholds on your labelled requests
with `basal-calibrate`. It runs the engine you serve, refits by the same procedure, and prints coverage and error at
the shipped and the refitted thresholds on a held-out part of your items. A few hundred labelled items usually certify
the 5% target; the 1% target needs at least 299 accepted decisions without an error in the fitting part, so in
practice about 1,000 items or more. A target that cannot be certified is marked in the file, and `max_error` then
defers every `act` / `tool` decision of that type. Input format and method: [docs/CALIBRATION.md](docs/CALIBRATION.md).

```bash
# labelled.jsonl: one request per line, each question with its correct answer in "gold"
basal-calibrate labelled.jsonl --model Remek/basal-2.0-4.5B --mode fast     # writes CALIBRATION.custom.json
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --calibration CALIBRATION.custom.json
```

### How to ask

- **One condition per yes/no question.** "Does the message mention health, finances or home address?" is harder than
  three questions or one `multi` with three labels.
- **Prefer `choice` with described sides** for a binary decision whose two outcomes have names ("approve: the invoice
  matches the order" / "hold: amounts or items differ").
- **Ask in the language of the state.** The prompt language is chosen from the text: Polish when the state or the
  question contains Polish diacritics, English otherwise.
- **Put the facts and rules in the state.** The model decides from what it reads; a small model's world knowledge is
  limited, and legal rules change.
- **Let code compute, let basal decide.** Exact arithmetic and thresholds are a weak spot of one-pass models: compute
  numbers in code (or send `"facts": "auto"` for Polish dates and amounts) and ask the typed question on top.
- **Offer "cannot be determined"** (or "none of the options") when the state may not contain the answer.
- **Use `criteria` descriptions** that say what each option means, not only a label; keep keys stable and send
  `"option_keys": "hide"` when keys are placeholders.

### MCP: basal as a tool for agents

*Experimental.* With the `basal[mcp]` extra, `--mcp on` also serves the decisions over the
[Model Context Protocol](https://modelcontextprotocol.io) at `/mcp` (streamable HTTP, stateless), next to
`/v1/systemone` and through the same queue. An agent host (Claude Code, Claude Desktop, Cursor, an agent SDK) then
calls basal as a tool:

- `decide`: one `choice`, `noul`, `score` or `multi` question about a state (`state`, `question`, `type`, `options`,
  `lang`). Returns that question's answer as in the HTTP API, plus `model`.
- `decide_many`: several questions over one state in one pass, every question type, including `act` with
  `max_error` and `tool` (`state`, `questions` as in the HTTP API, `lang`, `facts`). Returns the HTTP API response.

Both tools are read-only. The server's MCP instructions give the agent the rules of [how to ask](#how-to-ask), and an
invalid request (for example a question with one option) returns a tool error with the server's message.

```bash
uv pip install "basal[fp8,mcp] @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --host 127.0.0.1 --port 8000 --mcp on
claude mcp add --transport http basal http://127.0.0.1:8000/mcp     # Claude Code; other hosts take the same URL
```

For hosts that start MCP servers as local processes, `basal-mcp` is a stdio bridge to a running server (use the full
path of `basal-mcp` in your environment):

```json
{"mcpServers": {"basal": {"command": "/home/me/basal-env/bin/basal-mcp", "args": ["--url", "http://127.0.0.1:8000"]}}}
```

The MCP endpoint has no authentication, like the HTTP API. Bind `--host 127.0.0.1` for agents on the same machine
(the Host and Origin headers are then checked against DNS rebinding), or put the server behind your own gateway.

## Limitations

- Evaluated on held-out and hidden test sets and on public benchmarks; claims about a specific document collection
  need validation on your own data.
- On a panel of 10 public English decision tasks all three basal-2.0 models are below basal-1.5 (−2.1 / −3.6 / −1.8
  points for mini / 4.5B / max); basal-2.0-mini is also below basal-1.5-mini on Polish decisions (−1.4 points).
- The calibration is cautious on unseen tasks: `max_error: 0.01` defers every decision with the shipped files (except
  `choice` with `basal-2.0-max-GGUF`'s `CALIBRATION.ollama.json`, confidence 0.989); use 0.05 or fit thresholds on your
  data. At the 1% target on Werdykt, basal-2.0 (4.5B) and max automate fewer decisions than basal-1.5.
- The thinking mode is experimental: on Werdykt it is less accurate, less calibrated and slower than the default.
- Polish world knowledge of a small model is limited; supply the relevant facts and rules in the state. Legal rules
  change; the models do not know rules introduced after their training.
- Exact arithmetic and thresholds remain a weak spot of one-pass decisions; compute them in code or use `"facts":
  "auto"` for Polish dates and amounts.
- basal-2.0-mini has an 8,192-token context; longer states are cut in the middle (or refused with
  `--context-overflow refuse`).
- `act`, `tool`, `multi` over many labels and evidence spans are experimental; `multi` asks each label as an
  independent yes/no question.
- basal-omni is experimental: trained on text decisions only, with images and audio read zero-shot by Gemma 4's
  encoders and answered with the text calibration (`"calibration": "text"`); validate it on your own media. It runs
  in `eager` mode only, without the ports and the thinking mode of basal-2.0.
- Decisions with serious consequences for people should be reviewed by a person.

## Previous releases

**basal-1.5** (October 2026; mini 1.5B, 4.5B, max 11B, with FP8 / NVFP4, MLX 4-bit / 8-bit and GGUF ports) remains
available on Hugging Face ([collection](https://huggingface.co/collections/Remek/basal-15-6ac0de0e9199d121c6adac20))
and is served unchanged by this engine (`--model Remek/basal-1.5-4.5B`; two option orders by default). Its release
README, with the full Werdykt board of that release: [v1.5.0](https://github.com/rkinas/basal/tree/v1.5.0).

**basal-1.0** (September 2026; 4.5B and 1.5B, plus FP8 / NVFP4 ModelOpt checkpoints) remains available on Hugging Face
(`Remek/basal-1.0-4.5B`, `Remek/basal-1.0-1.5B`). Its full tables, methods and limitations are in the
[basal-1.0 technical report](docs/basal-1.pdf) and the [v1.0.1 README](https://github.com/rkinas/basal/tree/v1.0.1).
The engine serves basal-1.0 models unchanged; `fast-exit` and the `vllm` mode with the ModelOpt checkpoints work as
before. The Apple Silicon engine modes (`mps`, `gguf`, `mlx-q8`) and the Apple Silicon engine study were contributed
by Paweł Kiszczak ([#4](https://github.com/rkinas/basal/pull/4)).

## Citation

```bibtex
@software{kinas2026basal20,
  title   = {basal-2.0: typed-decision models and inference engine for Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/rkinas/basal}
}

@techreport{kinas2026basal,
  title       = {basal-1.0: Reliable, Highly Optimized Typed Decisions for Polish},
  author      = {Kinas, Remigiusz},
  institution = {ai5},
  year        = {2026},
  type        = {Technical report},
  doi         = {10.5281/zenodo.23022986},
  url         = {https://doi.org/10.5281/zenodo.23022986}
}
```

## License

Apache-2.0. The models are derivatives of Apache-2.0 base models (Bielik v3, SpeakLeash and ACK Cyfronet AGH, for
basal-2.0; Gemma 4, Google, for basal-omni); see [NOTICE](NOTICE).
