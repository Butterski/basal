"""The eager backend's own Gemma 4 forward (basal/gemma4.py): shared prefixes once (SOAM), the layers from the last
one with its own keys and values at the answer rows only, media towers once per request. Every readout equals the
transformers forward of the whole prompt (fp32, tiny random models shaped like basal-omni: E4B-like with KV sharing
and per-layer inputs, 31B-like with keys = values and no sharing, a MoE block, and a multimodal model with images,
audio and the larger models' bidirectional image attention)."""
import json

import pytest
import torch
from transformers import (Gemma4AudioConfig, Gemma4Config, Gemma4ForCausalLM, Gemma4ForConditionalGeneration,
                          Gemma4TextConfig, Gemma4VisionConfig)

from basal import engine
from basal.engine import EagerBackend
from basal.gemma4 import Gemma4Forward

ROPE = {"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}
IMG, AUD, VID, BOI, EOI, BOA, EOA = 60, 61, 62, 58, 59, 57, 56


def text_config(**kw):
    base = dict(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, global_head_dim=32, sliding_window=16,
                layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
                vocab_size_per_layer_input=64, hidden_size_per_layer_input=8, num_kv_shared_layers=2,
                final_logit_softcapping=30.0, max_position_embeddings=1024, rope_parameters=ROPE)
    return Gemma4TextConfig(**{**base, **kw})


CONFIGS = {
    "e4b": {},  # KV sharing of the last layers, per-layer inputs, wider global heads
    "31b": dict(num_kv_shared_layers=0, hidden_size_per_layer_input=0, attention_k_eq_v=True,
                use_bidirectional_attention="vision"),
    "moe": dict(num_kv_shared_layers=0, hidden_size_per_layer_input=0, enable_moe_block=True, num_experts=4,
                top_k_experts=2, moe_intermediate_size=32),
}


def tiny(name):
    torch.manual_seed(0)
    m = Gemma4ForCausalLM(text_config(**CONFIGS[name])).eval()
    with torch.no_grad():
        m.model.embed_tokens.weight.mul_(30.0)  # tied head: code logits large enough for the softcap to bend them
    return m


def groups_of():
    """A state longer than the sliding window and CHUNK with three questions, a short single, a long single, a short
    group, and a group whose prompts are prefixes of each other."""
    g = torch.Generator().manual_seed(1)
    r = lambda n: torch.randint(1, 56, (n,), generator=g).tolist()  # noqa: E731
    state = r(50)
    return [[state + [3, 4, 5], state + [3, 9], state + [7] * 7 + [1]], [r(10)], [r(70)], [[1, 2, 3, 4], [1, 2, 3, 5, 6]],
            [[1, 2, 3], [1, 2, 3, 4]], [r(30) + [5], r(12)]]


def last_hidden(m, t):
    with torch.no_grad():
        return m.model(input_ids=torch.tensor([t])).last_hidden_state[0, -1]


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("answer_rows", [True, False])
def test_forward_equals_transformers(name, answer_rows):
    m = tiny(name)
    g = Gemma4Forward(m, answer_rows=answer_rows)
    g.CHUNK = 24  # the state and the long single run in parts
    groups = groups_of()
    for grp, hs in zip(groups, g.hidden(groups)):
        for t, h in zip(grp, hs):
            r = last_hidden(m, t)
            assert (h - r).abs().max() <= 1e-4 * r.abs().max(), (name, t)


def test_kv_sharing_layout():
    g = Gemma4Forward(tiny("e4b"))
    assert g.last_kv == 3 and g.src == {4: 3, 5: 2} and g.store == {2, 3}  # E4B: the last 18 of 42 layers share
    g = Gemma4Forward(tiny("31b"))
    assert g.last_kv == 5 and g.src == {} and g.bidir  # 12B / 31B: the last layer at the answer rows only


def test_answer_rows_skip_work():
    """Layers after the last one with its own keys and values see one row per prompt; a prefix stops at that layer."""
    m = tiny("e4b")
    seen = []
    for i, layer in enumerate(m.model.layers):
        layer.mlp.register_forward_hook(lambda mod, inp, out, i=i: seen.append((i, inp[0].shape[1])))
    g = Gemma4Forward(m)
    g.hidden([[list(range(1, 41)) + [5], list(range(1, 41)) + [6, 7]]])
    prefix = [(i, n) for i, n in seen if n == 40]
    assert [i for i, _ in prefix] == [0, 1, 2]  # the prefix: layers before the last own one (3), MLPs only there
    assert all(n == 1 for i, n in seen if i >= 3)  # the suffix pass from layer 3 on: the answer row of each prompt
    seen.clear()
    Gemma4Forward(m, answer_rows=False).hidden([[list(range(1, 41)) + [5], list(range(1, 41)) + [6, 7]]])
    assert all(n == 2 for i, n in seen if n != 40)  # without: every suffix token in every layer


def test_budgets_split_forwards(monkeypatch):
    m = tiny("e4b")
    groups = groups_of()
    g = Gemma4Forward(m)
    g.CHUNK = 24
    ref = g.hidden(groups)
    calls = []
    orig = Gemma4Forward._pass
    monkeypatch.setattr(Gemma4Forward, "_pass", lambda self, h, *a, **k: calls.append(h.shape) or orig(self, h, *a, **k))
    g.MAX_BATCH_TOKENS, g.ATTN_PAIRS = 40, 2000
    got = g.hidden(groups)
    assert max(s[0] * s[1] for s in calls) <= 40 and len(calls) > 8
    calls.clear()
    g.MAX_BATCH_TOKENS, g.ATTN_PAIRS, g.KV_BYTES = 16384, 2048 * 16384, 60 * g.kv_row_bytes
    assert g.hidden(groups[:1]) and [s[0] for s in calls[-2:]] == [1, 1]  # 3 suffixes over 50 keys: one per forward
    got2 = g.hidden(groups)
    for a, b, c in zip(ref, got, got2):
        for x, y, z in zip(a, b, c):
            assert torch.allclose(x, y, atol=1e-5) and torch.allclose(x, z, atol=1e-5)


class Stub(EagerBackend):
    def __init__(self, model, g4=True, proc=None):  # the real __init__ minus loading
        self.model, self.dev, self.prefill, self.proc = model, torch.device("cpu"), "", proc
        self.tok = type("Tok", (), {"pad_token_id": 0, "apply_chat_template": lambda _s, msgs, **_k: json.dumps(msgs)})()
        self.head = model.get_output_embeddings()
        self.softcap = model.config.get_text_config().final_logit_softcapping
        self._head_in = None
        self.head.register_forward_hook(self._grab)
        self.g4 = Gemma4Forward(model) if g4 else None

    def encode(self, prompts):
        return prompts


def test_eager_backend_soam_equals_transformers_path():
    m = tiny("e4b")
    groups = groups_of()
    lids = [[[10, 11, 12]] * len(g) for g in groups]
    new = Stub(m).run_shared(list(zip(groups, lids)))
    old = Stub(m, g4=False).run_shared(list(zip(groups, lids)))
    for a, b in zip(new, old):
        for x, y in zip(a, b):
            assert max(abs(p - q) for p, q in zip(x, y)) < 1e-5
    flat = [p for g in groups for p in g]
    for x, y in zip(Stub(m).run(flat, [[10, 11]] * len(flat)), Stub(m, g4=False).run(flat, [[10, 11]] * len(flat))):
        assert max(abs(p - q) for p, q in zip(x, y)) < 1e-5


def tiny_mm(bidir, shared, ple):
    torch.manual_seed(0)
    t = text_config(sliding_window=8, hidden_size_per_layer_input=ple, num_kv_shared_layers=shared,
                    use_bidirectional_attention="vision" if bidir else None)
    v = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=2, head_dim=16, patch_size=2, pooling_kernel_size=2,
                           position_embedding_size=64)
    a = Gemma4AudioConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, subsampling_conv_channels=[128, 4],
                          output_proj_dims=32)
    c = Gemma4Config(text_config=t, vision_config=v, audio_config=a, image_token_id=IMG, audio_token_id=AUD,
                     video_token_id=VID, boi_token_id=BOI, eoi_token_id=EOI, boa_token_id=BOA, eoa_token_index=EOA)
    return Gemma4ForConditionalGeneration(c).eval()


class Proc:
    """The processor of one request: an image (4 x 4 patches -> 4 soft tokens) and 40 audio frames (10 soft tokens),
    placed where the state goes, before the prompt's own tokens (here the chat template's output is the JSON of the
    messages, a list of token ids)."""
    image_token_id, audio_token_id = IMG, AUD

    def __init__(self, n_audio=10):
        g = torch.Generator().manual_seed(2)
        self.px = torch.randn(1, 16, 12, generator=g)
        self.yx = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), -1).reshape(1, 16, 2)
        self.feats, self.n_audio = torch.randn(1, 40, 128, generator=g), n_audio

    def __call__(self, text, images=None, audio=None, **_):
        ids = [2, 5, 6, BOI] + [IMG] * 4 + [EOI, 7, BOA] + [AUD] * self.n_audio + [EOA, 9] + json.loads(text[0])
        ids = torch.tensor([ids])
        return dict(input_ids=ids, pixel_values=self.px, image_position_ids=self.yx, input_features=self.feats,
                    input_features_mask=torch.ones(1, 40, dtype=torch.bool),
                    mm_token_type_ids=torch.where(ids == IMG, 1, torch.where(ids == AUD, 3, 0)))


@pytest.mark.parametrize("bidir,shared,ple", [(True, 0, 0), (False, 2, 8)])
def test_media_once_equals_transformers_path(bidir, shared, ple, monkeypatch):
    m = tiny_mm(bidir, shared, ple)
    towers = []
    vt = m.model.vision_tower
    vt.register_forward_hook(lambda *_: towers.append(1))
    state = torch.randint(1, 50, (20,), generator=torch.Generator().manual_seed(3)).tolist()
    msgs = [state + [11, 12, 13], state + [11, 14], state + [15, 16, 17, 18]]
    lids = [[20, 21], [20, 21, 22], [23, 24]]
    for chunk in (2048, 6):  # 6: the prefix in parts around the image (never inside it)
        new = Stub(m, proc=Proc())
        new.g4.CHUNK = chunk
        towers.clear()
        got = new.run_media(msgs, lids, ["img"], ["aud"], 280)
        assert len(towers) == 1  # the image tower once for the three prompts
        ref = Stub(m, g4=False, proc=Proc()).run_media(msgs, lids, ["img"], ["aud"], 280)
        assert len(towers) == 4  # the transformers path: once per prompt
        for (p, n, nm), (q, n2, nm2) in zip(got, ref):
            assert (n, nm) == (n2, nm2) == (n, 14)
            assert max(abs(a - b) for a, b in zip(p, q)) < 1e-5
    if bidir:  # the image's soft tokens attend to each other in the sliding-window layers
        new.g4.bidir = False
        off = new.run_media(msgs, lids, ["img"], ["aud"], 280)
        assert max(abs(a - b) for (p, *_), (q, *_) in zip(off, got) for a, b in zip(p, q)) > 1e-4


def test_eager_forward_choice(tmp_path, monkeypatch):
    tiny("e4b").save_pretrained(tmp_path)
    tok = type("Tok", (), {"pad_token": "<pad>", "eos_token": "<eos>", "pad_token_id": 0})
    monkeypatch.setattr(engine.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tok())
    assert isinstance(EagerBackend(tmp_path, "float32", device="cpu").g4, Gemma4Forward)
    assert EagerBackend(tmp_path, "float32", device="cpu", forward="hf").g4 is None
    monkeypatch.setenv("BASAL_EAGER_FORWARD", "hf")
    assert EagerBackend(tmp_path, "float32", device="cpu").g4 is None
    with pytest.raises(ValueError):
        EagerBackend(tmp_path, "float32", device="cpu", forward="fast")
