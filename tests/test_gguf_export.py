"""basal-export-gguf: BPE merge ranks become llama.cpp token scores as in the -GGUF files, the word-boundary prefix is
switched off, the decision metadata for Ollama is written, and --gguf fixes an existing file (on a tiny GGUF; the
converter is a stand-in script)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

gguf = pytest.importorskip("gguf")

from basal.gguf_export import decision_metadata, fix_gguf, main, merge_rank_scores

TOKENS = ["<s>", "a", "▁a", "bc", "▁abc"]
TOKENIZER = {"model": {"merges": [["▁", "a"], ["b", "c"], ["▁a", "bc"], ["b", "c"]],
                       "vocab": {t: i for i, t in enumerate(TOKENS)}},
             "added_tokens": [{"id": 0, "content": "<s>", "special": True}]}
CAL = {"choice": 1.14, "noul": 1.39, "score": 1.54}  # one temperature per type (basal-1.5-mini's, rounded)
CONFIG = {"num_hidden_layers": 2, "hidden_size": 8, "vocab_size": 5}  # the tiny GGUF's model


def write_model(d, tokenizer=TOKENIZER, scores=(-1000.0,) * 5):
    """A tiny stock conversion (one score for every token, no special types) and its model folder."""
    d.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(d / "raw.gguf"), arch="llama")
    w.add_block_count(2)
    w.add_embedding_length(8)
    w.add_tokenizer_model("llama")
    w.add_token_list(TOKENS)
    w.add_token_scores(list(scores))
    w.add_token_types([1] * 5)  # converter output when tokenizer_config.json lists no special tokens
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    (d / "tokenizer.json").write_text(json.dumps(tokenizer))
    (d / "CALIBRATION.json").write_text(json.dumps({"temperature_per_prim": CAL}))
    return d / "raw.gguf"


def tokenizer_fields(path):
    return {k: f.contents() for k, f in gguf.GGUFReader(path).fields.items() if k.startswith("tokenizer.")}


def test_merge_rank_scores_follow_merge_order():
    scores, ranked = merge_rank_scores(TOKENIZER, TOKENS)
    assert (scores, ranked) == ([-1000.0, 0.0, 0.0, -1.0, -2.0], 3)  # added token: the converter's -1000, "a": 0
    assert math.copysign(1, scores[2]) == -1 and math.copysign(1, scores[1]) == 1  # -0.0 / 0.0 as in the -GGUF files
    assert merge_rank_scores({"model": {"merges": ["▁ a", "b c"]}}, ["bc", "▁a"]) == ([-1.0, 0.0], 2)  # "a b"


def test_added_tokens_score_as_in_a_fresh_conversion_whatever_the_input(tmp_path):
    """An earlier basal-export-gguf scored every unranked token, the added ones included, -(1 + number of merges) (the
    published basal-1.0 Ollama files); fixing such a file gives the bytes of a fixed stock conversion."""
    old = write_model(tmp_path / "old", scores=[-5.0, -5.0, -0.0, -1.0, -2.0])
    fix_gguf(old, tmp_path / "from-old.gguf", tmp_path / "old")
    fix_gguf(write_model(tmp_path / "stock"), tmp_path / "from-stock.gguf", tmp_path / "stock")
    assert gguf.GGUFReader(tmp_path / "from-old.gguf").fields["tokenizer.ggml.scores"].contents()[0] == -1000.0
    assert (tmp_path / "from-old.gguf").read_bytes() == (tmp_path / "from-stock.gguf").read_bytes()


def test_decision_metadata_from_calibration():
    assert decision_metadata("llama", {"temperature_per_prim": CAL}) == {
        "llama.decision.type": ("string", "basal"), "llama.decision.temperature.choice": ("float32", 1.14),
        "llama.decision.temperature.noul": ("float32", 1.39), "llama.decision.temperature.score": ("float32", 1.54)}


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
    assert {t: f[f"llama.decision.temperature.{t}"].contents() for t in CAL} == pytest.approx(CAL)
    T = gguf.GGUFValueType  # the types llama.cpp reads (llama-vocab.cpp refuses float64 scores) and Ollama expects
    assert f["tokenizer.ggml.scores"].types == [T.ARRAY, T.FLOAT32]
    assert f["tokenizer.ggml.token_type"].types == [T.ARRAY, T.INT32]
    assert f["tokenizer.ggml.add_space_prefix"].types == [T.BOOL]
    assert f["llama.decision.type"].types == [T.STRING]
    assert [f[f"llama.decision.temperature.{t}"].types for t in CAL] == [[T.FLOAT32]] * 3


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
    # another token at an id; a token beyond the GGUF's vocabulary
    for vocab in ({"<s>": 0, "b": 1}, {**TOKENIZER["model"]["vocab"], "x": 5}):
        raw = write_model(tmp_path, {**TOKENIZER, "model": {**TOKENIZER["model"], "vocab": vocab}})
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


def test_cli_repair_downloads_only_tokenizer_calibration_and_config(tmp_path, monkeypatch):
    raw = write_model(tmp_path)
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda r, **kw: calls.append((r, kw)) or str(tmp_path))
    main(["Remek/never-downloaded-GGUF", str(tmp_path / "fixed.gguf"), "--gguf", str(raw), "--revision", "v1"])
    assert calls == [("Remek/never-downloaded-GGUF", {"revision": "v1", "allow_patterns": [
        "tokenizer.json", "CALIBRATION.json", "CALIBRATION.ollama.json", "config.json", "basal.json"]})]


def test_ollama_decision_prefers_the_ollama_calibration(tmp_path, capsys):
    raw = write_model(tmp_path)  # basal-serve --mode ollama reads CALIBRATION.ollama.json when the model ships it
    (tmp_path / "CALIBRATION.ollama.json").write_text(json.dumps({"temperature_per_prim": {
        "choice": 2.0, "noul": 2.5, "score": 3.0}}))
    main([str(tmp_path), str(tmp_path / "out.gguf"), "--gguf", str(raw), "--ollama-decision"])
    f = gguf.GGUFReader(tmp_path / "out.gguf").fields
    assert [f[f"llama.decision.temperature.{t}"].contents() for t in ("choice", "noul", "score")] == [2.0, 2.5, 3.0]
    assert "temperatures from CALIBRATION.ollama.json of " in capsys.readouterr().out


def test_ollama_decision_refuses_the_config_of_another_model(tmp_path):
    raw = write_model(tmp_path)
    for change, error in (({"num_hidden_layers": None}, "config.json is missing num_hidden_layers"),
                          ({"num_hidden_layers": 50}, "block_count is 2, but the model's config.json specifies "
                                                      "num_hidden_layers=50"),
                          ({"hidden_size": 4096}, "embedding_length is 8"), ({"vocab_size": 32000}, "vocabulary is 5")):
        (tmp_path / "config.json").write_text(json.dumps({k: v for k, v in {**CONFIG, **change}.items() if v}))
        with pytest.raises(ValueError, match=error):
            fix_gguf(raw, tmp_path / "out.gguf", tmp_path, ollama_decision=True)
        assert not (tmp_path / "out.gguf").exists()
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path), str(tmp_path / "out.gguf"), "--gguf", str(raw), "--ollama-decision"])
    assert str(e.value.code).startswith("basal-export-gguf: ") and "another model" in str(e.value.code)
    fix_gguf(raw, tmp_path / "plain.gguf", tmp_path)  # without the flag nothing depends on the model's size


def test_ollama_decision_names_the_calibration_and_the_model(tmp_path, capsys):
    raw = write_model(tmp_path)  # a -GGUF folder: basal.json, no config.json
    (tmp_path / "basal.json").write_text(json.dumps({"name": "basal-tiny"}))
    main([str(tmp_path), str(tmp_path / "a.gguf"), "--gguf", str(raw), "--ollama-decision"])
    assert (f"temperatures from CALIBRATION.json of {tmp_path} (basal-tiny), no config.json to check: choice 1.1400, "
            "noul 1.3900, score 1.5400") in capsys.readouterr().out
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    main([str(tmp_path), str(tmp_path / "b.gguf"), "--gguf", str(raw), "--ollama-decision"])
    out = capsys.readouterr().out
    assert "(basal-tiny), config.json has the GGUF's shape (2 layers, 8 wide, 5 tokens): choice" in out


def test_cli_rejects_a_model_without_tokenizer_or_calibration(tmp_path, monkeypatch, capsys):
    raw = write_model(tmp_path / "m")
    (tmp_path / "m" / "CALIBRATION.json").unlink()
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "m"), str(tmp_path / "out.gguf"), "--gguf", str(raw), "--ollama-decision"])
    assert e.value.code == 2 and "has no CALIBRATION.json, which --ollama-decision needs" in capsys.readouterr().err
    main([str(tmp_path / "m"), str(tmp_path / "out.gguf"), "--gguf", str(raw)])  # not needed without the flag
    snapshot = tmp_path / "hfcache" / "snapshots" / "fe6346d"  # a third-party GGUF repo: only .gguf files
    snapshot.mkdir(parents=True)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda r, **kw: str(snapshot))
    with pytest.raises(SystemExit) as e:
        main(["someone/basal-1.0-4.5B-GGUF", str(tmp_path / "x.gguf"), "--gguf", str(raw)])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "someone/basal-1.0-4.5B-GGUF has no tokenizer.json" in err and "hfcache" not in err


def test_cli_names_both_installs_when_gguf_is_missing(tmp_path, monkeypatch, capsys):
    raw = write_model(tmp_path)
    monkeypatch.setitem(sys.modules, "gguf", None)  # import gguf raises ImportError
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path), str(tmp_path / "x.gguf"), "--gguf", str(raw)])
    err = capsys.readouterr().err
    assert e.value.code == 2 and 'uv pip install -e ".[gguf-export]"' in err and 'uv pip install "gguf>=0.19"' in err


def test_help_without_docstrings():
    done = subprocess.run([sys.executable, "-OO", "-m", "basal.gguf_export", "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--ollama-decision" in done.stdout, done.stderr


def test_cli_refuses_to_write_over_its_input(tmp_path, capsys):
    raw = write_model(tmp_path)
    before = raw.read_bytes()
    (tmp_path / "link.gguf").symlink_to(raw)
    for out in (raw, tmp_path / "link.gguf"):
        with pytest.raises(SystemExit) as e:
            main([str(tmp_path), str(out), "--gguf", str(raw)])
        assert e.value.code == 2 and "must be a new file" in capsys.readouterr().err
    assert raw.read_bytes() == before


def test_cli_rejects_a_missing_input_or_a_directory_as_output_before_any_work(tmp_path, monkeypatch, capsys):
    raw = write_model(tmp_path)
    monkeypatch.setattr("basal.gguf_export.subprocess.run", lambda *a, **k: pytest.fail("converter run"))
    (tmp_path / "outdir").mkdir()
    missing = str(tmp_path / "missing.gguf")
    for argv, error in (([str(tmp_path / "out.gguf"), "--gguf", missing], "missing.gguf not found"),
                        ([str(tmp_path / "outdir"), "--gguf", str(raw)], "is a directory"),
                        ([str(tmp_path / "outdir"), "--llama-cpp", str(tmp_path)], "is a directory")):
        with pytest.raises(SystemExit) as e:
            main([str(tmp_path), *argv])
        assert e.value.code == 2 and error in capsys.readouterr().err


def test_cli_rejects_missing_converter_before_download(tmp_path, capsys):
    for llama_cpp in (["--llama-cpp", str(tmp_path)], []):
        with pytest.raises(SystemExit) as e:
            main(["Remek/never-downloaded", str(tmp_path / "x.gguf"), *llama_cpp])
        assert e.value.code == 2 and "convert_hf_to_gguf.py not found" in capsys.readouterr().err


def test_cli_creates_missing_output_directory(tmp_path):
    write_model(tmp_path)
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("import sys; sys.exit(3)\n")  # stops right after the temp dir
    out = tmp_path / "new" / "dir" / "model.gguf"
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path), str(out), "--llama-cpp", str(llama)])
    assert "converter failed (exit 3)" in str(e.value.code) and out.parent.is_dir()


def test_cli_converts_a_local_model_then_fixes_it(tmp_path, monkeypatch, capsys):
    raw = write_model(tmp_path / "model")  # the stand-in converter writes this tiny GGUF as its output
    (tmp_path / "model" / "config.json").write_text(json.dumps(CONFIG))
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text(
        "import json, shutil, sys\n"
        f"json.dump(sys.argv[1:], open({str(tmp_path / 'argv.json')!r}, 'w'))\n"
        f"shutil.copy({str(raw)!r}, sys.argv[sys.argv.index('--outfile') + 1])\n")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: pytest.fail("download"))
    out = tmp_path / "out" / "model-Q8_0.gguf"
    main([str(tmp_path / "model"), str(out), "--outtype", "q8_0", "--llama-cpp", str(llama), "--ollama-decision"])
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[:4] == [str(tmp_path / "model"), "--outtype", "q8_0", "--outfile"]
    assert Path(argv[4]).parent.parent == out.parent and sorted(out.parent.iterdir()) == [out]  # temp dir removed
    f = gguf.GGUFReader(out).fields
    assert f["tokenizer.ggml.scores"].contents() == [-1000, 0, 0, -1, -2] and f["llama.decision.type"].contents()
    assert "config.json has the GGUF's shape" in capsys.readouterr().out


def test_cli_conversion_from_a_repo_downloads_the_whole_model(tmp_path, monkeypatch):
    raw = write_model(tmp_path / "model")
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text(
        f"import shutil, sys; shutil.copy({str(raw)!r}, sys.argv[sys.argv.index('--outfile') + 1])\n")
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda r, **kw: calls.append((r, kw)) or str(tmp_path / "model"))
    main(["Remek/basal-tiny", str(tmp_path / "out.gguf"), "--llama-cpp", str(llama)])
    # the converter needs the weights: no allow_patterns
    assert calls == [("Remek/basal-tiny", {"revision": None, "allow_patterns": None})]
