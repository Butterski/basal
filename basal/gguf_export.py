"""Export a basal checkpoint to GGUF whose tokenizer matches the Hugging Face one, with decision metadata for Ollama.

llama.cpp's converter writes basal's BPE vocabulary as a SentencePiece-style vocabulary with one constant score for
every token and with a word-boundary prefix, so llama.cpp (and Ollama, LM Studio, llama-server with text prompts)
splits text differently from the tokenizer the models were trained with. This tool runs the converter and then fixes
the file as the basal-1.5 -GGUF files are fixed (docs/PORTS.md, "Tokenizer fix"). With --gguf it fixes an existing
GGUF instead (a stock conversion, a llama-quantize output, an older basal GGUF); the model then only supplies
tokenizer.json and CALIBRATION.json, downloaded without the weights:

- tokenizer.ggml.scores: minus the rank of the first BPE merge that creates each token (the first merge scores 0), so
  llama.cpp's greedy highest-score-first merging follows the BPE merge order; the added tokens of tokenizer.json keep
  their score, every other token (single characters, byte tokens) scores 0;
- tokenizer.ggml.add_space_prefix = false: the Hugging Face tokenizer adds no word-boundary marker after special
  tokens such as <|im_start|>;
- tokenizer.ggml.token_type: every added token of tokenizer.json is CONTROL (special) or USER_DEFINED, as the Hugging
  Face tokenizer matches them whole; the converter reads special tokens from tokenizer_config.json only, and
  basal-1.0-4.5B's and basal-1.5-max's do not list <|im_start|>, so llama.cpp split it into characters;
- with --ollama-decision: <arch>.decision.type = "basal" and <arch>.decision.temperature.{choice,noul,score} from
  CALIBRATION.json (without it, such keys of the input are removed). Ollama's /v1/systemone then renders basal's own
  prompt format, scores both option orders and calibrates. Such a file is decision-only in Ollama (/api/generate and
  /api/chat refuse it) and needs an Ollama with the basal decision encoding; other Ollama versions reject it.

Fixing a fixed file changes nothing (the basal-1.5 -GGUF files come out byte-identical). The tokenizer matches the
Hugging Face one for text that starts with a special token (every basal prompt starts with <s>); for text without one,
Hugging Face adds a word-boundary marker at the very start and llama.cpp does not.

    basal-export-gguf Remek/basal-1.5-4.5B basal-1.5-4.5B-F16.gguf --outtype f16 --llama-cpp ~/llama.cpp
    basal-export-gguf Remek/basal-1.0-1.5B fixed/basal-1.0-1.5B-Q8_0.gguf --gguf basal-1.0-1.5B-Q8_0.gguf
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DECISION_TYPES = ("choice", "noul", "score")


def merge_rank_scores(tokenizer_json, tokens, scores):
    """Token scores as in the -GGUF files: -(rank of the first BPE merge producing the token); the added tokens of
    tokenizer.json keep their score; 0 for the rest (single characters, byte tokens: llama.cpp only compares the scores
    of merge results, and basal's vocabulary has no other multi-character token). Returns (scores, tokens ranked)."""
    rank = {}
    for i, m in enumerate(tokenizer_json["model"]["merges"]):
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        rank.setdefault(a + b, i)
    added = {t["id"] for t in tokenizer_json.get("added_tokens", [])}
    return ([float(s) if i in added else -float(rank[t]) if t in rank else 0.0
             for i, (t, s) in enumerate(zip(tokens, scores))], sum(t in rank for t in tokens))


def added_token_types(tokenizer_json, token_types):
    """GGUF token types with the added tokens of tokenizer.json marked CONTROL (special) or USER_DEFINED."""
    import gguf

    types = list(token_types)
    for t in tokenizer_json.get("added_tokens", []):
        if 0 <= t["id"] < len(types):
            types[t["id"]] = int(gguf.TokenType.CONTROL if t.get("special") else gguf.TokenType.USER_DEFINED)
    return types


def decision_metadata(arch, calibration):
    """{key: (type, value)} for the decision encoding and the per-type temperatures of CALIBRATION.json."""
    temps = calibration["temperature_per_prim"]
    meta = {f"{arch}.decision.type": ("string", "basal")}
    for typ in DECISION_TYPES:
        meta[f"{arch}.decision.temperature.{typ}"] = ("float32", float(temps[typ]))
    return meta


def fix_gguf(raw, outfile, model_dir, ollama_decision=False):
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    model_dir = Path(model_dir)
    reader = gguf.GGUFReader(raw)
    tokens = reader.fields[gguf.Keys.Tokenizer.LIST].contents()
    tokenizer_json = json.loads((model_dir / "tokenizer.json").read_text())
    if any(i >= len(tokens) or tokens[i] != t for t, i in tokenizer_json["model"]["vocab"].items()):
        raise ValueError(f"the vocabulary of {raw} differs from {model_dir / 'tokenizer.json'}")
    scores, ranked = merge_rank_scores(tokenizer_json, tokens, reader.fields[gguf.Keys.Tokenizer.SCORES].contents())
    token_types = added_token_types(tokenizer_json, reader.fields[gguf.Keys.Tokenizer.TOKEN_TYPE].contents())
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    types = {"string": gguf.GGUFValueType.STRING, "float32": gguf.GGUFValueType.FLOAT32}
    new = {gguf.Keys.Tokenizer.ADD_PREFIX: MetadataDetails(gguf.GGUFValueType.BOOL, False),
           gguf.Keys.Tokenizer.SCORES: MetadataDetails(gguf.GGUFValueType.ARRAY, scores,
                                                       sub_type=gguf.GGUFValueType.FLOAT32),
           gguf.Keys.Tokenizer.TOKEN_TYPE: MetadataDetails(gguf.GGUFValueType.ARRAY, token_types,
                                                           sub_type=gguf.GGUFValueType.INT32)}
    if ollama_decision:
        calibration = json.loads((model_dir / "CALIBRATION.json").read_text())
        for key, (typ, value) in decision_metadata(arch, calibration).items():
            new[key] = MetadataDetails(types[typ], value)
    stale = [] if ollama_decision else [k for k in reader.fields if k.startswith(f"{arch}.decision.")]
    writer = gguf.GGUFWriter(outfile, arch=arch, endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, new, stale)
    return ranked, len(tokens)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="local directory or Hugging Face repo id (with --gguf: only its tokenizer.json and "
                                  "CALIBRATION.json are read)")
    ap.add_argument("outfile")
    ap.add_argument("--gguf", default=None, help="fix this GGUF file instead of converting the model (a stock "
                                                "conversion, a llama-quantize output, an older basal GGUF)")
    ap.add_argument("--outtype", default="f16", help="converter output type: f16, q8_0, bf16, f32")
    ap.add_argument("--llama-cpp", default=None, help="llama.cpp checkout containing convert_hf_to_gguf.py "
                                                      "(not needed with --gguf)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--ollama-decision", action="store_true",
                    help="write decision metadata for Ollama's /v1/systemone (decision-only file; needs an Ollama with "
                         "the basal decision encoding)")
    a = ap.parse_args(argv)
    out = Path(a.outfile)
    if a.gguf:
        if not Path(a.gguf).is_file():
            ap.error(f"{a.gguf} not found")
        if out.exists() and out.samefile(a.gguf):
            ap.error("the output must be a new file: the input is read while the output is written")
    else:
        converter = Path(a.llama_cpp or "").expanduser() / "convert_hf_to_gguf.py"
        if not a.llama_cpp or not converter.is_file():
            ap.error(f"{converter} not found; clone https://github.com/ggml-org/llama.cpp and pass it as --llama-cpp "
                     "(or fix an existing file with --gguf)")
    try:
        import gguf  # noqa: F401
    except ImportError:
        ap.error('the gguf package is missing; install it with: uv pip install -e ".[gguf-export]"')
    from huggingface_hub import snapshot_download

    model_dir = Path(a.model) if Path(a.model).exists() else Path(snapshot_download(
        a.model, revision=a.revision, allow_patterns=["tokenizer.json", "CALIBRATION.json"] if a.gguf else None))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        raw = Path(a.gguf or Path(tmp) / "converted.gguf")
        if not a.gguf:
            done = subprocess.run([sys.executable, str(converter), str(model_dir), "--outtype", a.outtype,
                                   "--outfile", str(raw)])
            if done.returncode != 0:
                sys.exit(f"basal-export-gguf: llama.cpp's converter failed (exit {done.returncode}); "
                         "see its output above")
        try:
            ranked, total = fix_gguf(raw, out, model_dir, a.ollama_decision)
        except ValueError as e:
            sys.exit(f"basal-export-gguf: {e}")
    extra = ", decision.type=basal" if a.ollama_decision else ""
    print(f"wrote {out}: {ranked}/{total} token scores from BPE merges, add_space_prefix=false{extra}")


if __name__ == "__main__":
    main()
