"""Models with recurrent-state layers (Qwen3.5: Gated DeltaNet / linear attention beside full attention). The CUDA
graph engine (--mode fast, fast-nocompile, fast-exit, fp8, nvfp4) and mps share prefixes as attention keys and values
only, so they refuse such a model from its config.json, before any weights are read; without --mode it is served in
eager, which stays exact. --mode fp8 stops before loading on GPUs below compute capability 8.9. No GPU needed."""
import json

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

import basal.engine as engine_module
import basal.server as server_module
from basal.engine import (EagerBackend, ExitGraphBackend, GraphBackend, MPSBackend, fp8_refused, recurrent_layers)
from basal.server import Server, default_mode, parser

PICO08_LAYERS = (["linear_attention"] * 3 + ["full_attention"]) * 6  # Qwen3.5-0.8B: 18 Gated DeltaNet, 6 attention


def write_config(d, cfg):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


@pytest.fixture
def qwen35(tmp_path):
    return write_config(tmp_path / "pico08", {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
                                              "layer_types": PICO08_LAYERS, "num_hidden_layers": 24})


@pytest.fixture
def llama(tmp_path):
    return write_config(tmp_path / "llama", {"model_type": "llama", "num_hidden_layers": 2})


def test_recurrent_layers_from_config(tmp_path, qwen35, llama):
    assert recurrent_layers(qwen35) == "18 linear_attention layers"
    assert recurrent_layers(llama) is None
    # Gemma 4 (basal-omni): sliding and full attention in text_config, no recurrent state
    omni = write_config(tmp_path / "omni", {"model_type": "gemma4", "text_config": {
        "model_type": "gemma4_text", "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}})
    assert recurrent_layers(omni) is None
    # the multimodal Qwen3.5 wrapper: its decoder in text_config; a config without layer_types: by model_type
    vl = write_config(tmp_path / "vl", {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text",
                                                                                 "layer_types": PICO08_LAYERS}})
    assert recurrent_layers(vl) == "18 linear_attention layers"
    bare = write_config(tmp_path / "bare", {"model_type": "qwen3_5_text"})
    assert recurrent_layers(bare) == "model_type qwen3_5_text"
    assert recurrent_layers(tmp_path / "missing") is None


@pytest.mark.parametrize("cls, kw", [(GraphBackend, {}), (GraphBackend, {"quant": "fp8"}),
                                     (GraphBackend, {"compile": True, "batches": "compact", "prefix": True,
                                                     "state_cache_gb": 1}),
                                     (ExitGraphBackend, {}), (MPSBackend, {})])
def test_shared_prefix_engines_refuse_before_loading(monkeypatch, qwen35, cls, kw):
    def no_load(*a, **k):
        raise AssertionError("the refusal must come before the tokenizer and the weights are loaded")
    monkeypatch.setattr(engine_module.AutoTokenizer, "from_pretrained", no_load)
    monkeypatch.setattr(engine_module.AutoModelForCausalLM, "from_pretrained", no_load)
    with pytest.raises(ValueError, match="--mode eager") as e:
        cls(str(qwen35), **kw)
    assert "18 linear_attention layers" in str(e.value) and "--mode vllm" in str(e.value)


@pytest.mark.parametrize("mode", ["fast", "fast-nocompile", "fast-exit", "fp8", "nvfp4"])
def test_server_refuses_graph_modes_before_download(monkeypatch, qwen35, mode):
    def resolve(name, revision=None, metadata_only=False):
        assert metadata_only, "the weights must not be downloaded for a refused model"
        return str(qwen35)
    monkeypatch.setattr(server_module, "resolve", resolve)
    with pytest.raises(SystemExit, match="--mode eager"):
        Server(parser().parse_args(["--model", "Remek/pico08", "--mode", mode]))


def test_default_mode_is_eager_for_recurrent_models(monkeypatch, qwen35, llama):
    """Without --mode: eager for a Qwen3.5 model on CUDA (fast refuses it) and on Apple Silicon (mps refuses it, mlx is
    not checked on it); attention-only models keep their defaults."""
    monkeypatch.setattr(server_module, "resolve", lambda name, revision=None, metadata_only=False: name)
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: "fa3")
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    assert default_mode(str(qwen35)) == "eager"
    assert default_mode(str(llama)) == default_mode() == "fast"
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    assert default_mode(str(qwen35)) == "eager"
    assert default_mode(str(llama)) in ("mlx", "mps")


def test_default_mode_keeps_fast_when_the_config_cannot_be_read(monkeypatch):
    def offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(server_module, "resolve", offline)
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: "fa3")
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    assert default_mode("Remek/basal-2.0-pico") == "fast"  # the model load then reports it


def tiny_qwen35(d):
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})).eval()
    m.save_pretrained(d)
    return m


class Tok:
    pad_token = eos_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, add_special_tokens=False, **kw):
        return type("E", (), {"input_ids": texts})()


def test_fast_refuses_a_tiny_qwen35_and_eager_serves_it(monkeypatch, tmp_path):
    """The same checkpoint directory: GraphBackend refuses it, EagerBackend answers every prompt as its own fp32
    forward does (exact)."""
    m = tiny_qwen35(tmp_path / "tiny")
    monkeypatch.setattr(engine_module.AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())
    with pytest.raises(ValueError, match="1 linear_attention layers"):
        GraphBackend(str(tmp_path / "tiny"))
    be = EagerBackend(str(tmp_path / "tiny"), "float32", device="cpu")
    toks, lids = [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [5, 6, 7, 8, 9, 11]], [[11, 12, 13], [14, 15], [14, 15]]
    got = be.run(toks, lids)
    with torch.no_grad():
        for g, t, ids in zip(got, toks, lids):
            ref = torch.softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float()[ids], -1)
            assert torch.allclose(torch.tensor(g), ref, atol=1e-5)


@pytest.mark.parametrize("cc, refused", [((8, 6), True), ((8, 0), True), ((8, 9), False), ((9, 0), False),
                                         ((12, 0), False)])
def test_fp8_needs_compute_capability_8_9(monkeypatch, cc, refused):
    monkeypatch.setattr(engine_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_capability", lambda *a: cc)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3090")
    why = fp8_refused()
    assert bool(why) == refused
    if refused:
        assert f"sm_{cc[0]}{cc[1]}" in why and "8.9" in why and "--mode fast" in why


@pytest.mark.parametrize("args", [["--mode", "fp8"], ["--mode", "fast", "--quant", "fp8"],
                                  ["--mode", "fast-exit", "--quant", "fp8"]])
def test_server_refuses_fp8_on_ampere_before_loading(monkeypatch, args):
    monkeypatch.setattr(engine_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_capability", lambda *a: (8, 6))
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(server_module, "_recurrent", lambda *a: None)

    def resolve(name, revision=None, metadata_only=False):
        raise AssertionError("refused before the model is resolved")
    monkeypatch.setattr(server_module, "resolve", resolve)
    with pytest.raises(SystemExit, match="compute capability 8.9"):
        Server(parser().parse_args(["--model", "Remek/basal-2.0"] + args))


def test_fp8_refused_in_the_engine_too(monkeypatch, llama):
    """basal-bench and basal-calibrate build the engine without the server's checks."""
    monkeypatch.setattr(engine_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_capability", lambda *a: (8, 6))
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(engine_module.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: pytest.fail("refused before loading"))
    with pytest.raises(ValueError, match="sm_86"):
        GraphBackend(str(llama), quant="fp8")
