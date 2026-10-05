"""basal-export-gguf: BPE merge ranks become llama.cpp token scores as in the -GGUF files, the word-boundary prefix is
switched off, the decision metadata for Ollama is written, and --gguf fixes an existing file (on a tiny GGUF; the
converter itself is not run)."""
import json
import math

import pytest

gguf = pytest.importorskip("gguf")

from basal.gguf_export import decision_metadata, fix_gguf, main, merge_rank_scores

TOKENS = ["<s>", "a", "▁a", "bc", "▁abc"]
TOKENIZER = {"model": {"merges": [["▁", "a"], ["b", "c"], ["▁a", "bc"], ["b", "c"]],
                       "vocab": {t: i for i, t in enumerate(TOKENS)}},
             "added_tokens": [{"id": 0, "content": "<s>", "special": True}]}


def write_model(d, tokenizer=TOKENIZER):
    """A tiny stock conversion (one score for every token, no special types) and its model folder."""
    w = gguf.GGUFWriter(str(d / "raw.gguf"), arch="llama")
    w.add_tokenizer_model("llama")
    w.add_token_list(TOKENS)
    w.add_token_scores([-1000.0] * 5)
    w.add_token_types([1] * 5)  # converter output when tokenizer_config.json lists no special tokens
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    (d / "tokenizer.json").write_text(json.dumps(tokenizer))
    (d / "CALIBRATION.json").write_text(json.dumps({"temperature_per_prim": {"choice": 0.96, "noul": 1.33,
                                                                            "score": 0.96}}))
    return d / "raw.gguf"


def tokenizer_fields(path):
    return {k: f.contents() for k, f in gguf.GGUFReader(path).fields.items() if k.startswith("tokenizer.")}


def test_merge_rank_scores_follow_merge_order():
    scores, ranked = merge_rank_scores(TOKENIZER, TOKENS, [-1000.0] * 5)
    assert (scores, ranked) == ([-1000.0, 0.0, 0.0, -1.0, -2.0], 3)  # added token keeps its score, unranked "a": 0
    assert math.copysign(1, scores[2]) == -1 and math.copysign(1, scores[1]) == 1  # -0.0 / 0.0 as in the -GGUF files
    assert merge_rank_scores({"model": {"merges": ["▁ a", "b c"]}}, ["bc", "▁a"], [0, 0]) == ([-1.0, 0.0], 2)  # "a b"


def test_decision_metadata_from_calibration():
    meta = decision_metadata("llama", {"temperature_per_prim": {"choice": 0.9, "noul": 1.3, "score": 0.9}})
    assert meta["llama.decision.type"] == ("string", "basal")
    assert meta["llama.decision.temperature.noul"] == ("float32", 1.3)


def test_fix_gguf_rewrites_tokenizer_and_adds_decision_metadata(tmp_path):
    raw = write_model(tmp_path)
    out = tmp_path / "fixed.gguf"
    plain = tmp_path / "plain.gguf"
    assert fix_gguf(raw, plain, tmp_path) == (3, 5)
    assert "llama.decision.type" not in gguf.GGUFReader(plain).fields  # usable for generation in any Ollama
    assert fix_gguf(raw, out, tmp_path, ollama_decision=True) == (3, 5)
    f = gguf.GGUFReader(out).fields
    assert f["tokenizer.ggml.scores"].contents() == [-1000, 0, 0, -1, -2]
    assert f["tokenizer.ggml.add_space_prefix"].contents() is False
    assert f["tokenizer.ggml.token_type"].contents() == [3, 1, 1, 1, 1]
    assert f["llama.decision.type"].contents() == "basal"
    assert f["llama.decision.temperature.noul"].contents() == pytest.approx(1.33)


def test_fix_gguf_is_idempotent_and_drops_decision_metadata_without_the_flag(tmp_path):
    raw = write_model(tmp_path)
    once, twice, plain = tmp_path / "once.gguf", tmp_path / "twice.gguf", tmp_path / "plain.gguf"
    fix_gguf(raw, once, tmp_path, ollama_decision=True)
    fix_gguf(once, twice, tmp_path, ollama_decision=True)
    assert twice.read_bytes() == once.read_bytes()
    fix_gguf(once, plain, tmp_path)
    assert tokenizer_fields(plain) == tokenizer_fields(once) != tokenizer_fields(raw)
    assert not [k for k in gguf.GGUFReader(plain).fields if ".decision." in k]


def test_fix_gguf_rejects_another_vocabulary(tmp_path):
    raw = write_model(tmp_path, {**TOKENIZER, "model": {**TOKENIZER["model"], "vocab": {"<s>": 0, "b": 1}}})
    with pytest.raises(ValueError, match="vocabulary"):
        fix_gguf(raw, tmp_path / "out.gguf", tmp_path)
    assert not (tmp_path / "out.gguf").exists()


def test_cli_repairs_an_existing_gguf_without_the_converter(tmp_path, monkeypatch):
    raw = write_model(tmp_path)
    monkeypatch.setattr("basal.gguf_export.subprocess.run", lambda *a, **k: pytest.fail("converter run"))
    out = tmp_path / "new" / "dir" / "fixed.gguf"
    main([str(tmp_path), str(out), "--gguf", str(raw), "--ollama-decision"])
    assert gguf.GGUFReader(out).fields["tokenizer.ggml.scores"].contents() == [-1000, 0, 0, -1, -2]
    assert gguf.GGUFReader(out).fields["llama.decision.type"].contents() == "basal"


def test_cli_repair_downloads_only_tokenizer_and_calibration(tmp_path, monkeypatch):
    raw = write_model(tmp_path)
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda r, **kw: calls.append((r, kw)) or str(tmp_path))
    main(["Remek/never-downloaded-GGUF", str(tmp_path / "fixed.gguf"), "--gguf", str(raw), "--revision", "v1"])
    assert calls == [("Remek/never-downloaded-GGUF", {"revision": "v1",
                                                      "allow_patterns": ["tokenizer.json", "CALIBRATION.json"]})]


def test_cli_refuses_to_write_over_its_input(tmp_path, capsys):
    raw = write_model(tmp_path)
    before = raw.read_bytes()
    (tmp_path / "link.gguf").symlink_to(raw)
    for out in (raw, tmp_path / "link.gguf"):
        with pytest.raises(SystemExit) as e:
            main([str(tmp_path), str(out), "--gguf", str(raw)])
        assert e.value.code == 2 and "must be a new file" in capsys.readouterr().err
    assert raw.read_bytes() == before


def test_cli_rejects_missing_converter_before_download(tmp_path, capsys):
    for llama_cpp in (["--llama-cpp", str(tmp_path)], []):
        with pytest.raises(SystemExit) as e:
            main(["Remek/never-downloaded", str(tmp_path / "x.gguf"), *llama_cpp])
        assert e.value.code == 2 and "convert_hf_to_gguf.py not found" in capsys.readouterr().err


def test_cli_creates_missing_output_directory(tmp_path):
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("import sys; sys.exit(3)\n")  # stops right after the temp dir
    out = tmp_path / "new" / "dir" / "model.gguf"
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path), str(out), "--llama-cpp", str(llama)])
    assert "converter failed (exit 3)" in str(e.value.code) and out.parent.is_dir()
