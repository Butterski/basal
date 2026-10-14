"""Media in a request (basal-omni, --media on): decoding and limits (basal/media.py), the prompt with media where the
state goes (prompt.media_messages, as the basal-omni evaluation reads media), and Server.decide's media path with a stub eager
backend (answers, usage, errors; the text path untouched)."""
import asyncio
import base64
import io

import pytest

from basal import media
from basal.prompt import FORMATS, media_messages
from basal.server import models_payload
from test_server import FakeServer

CODES = FORMATS[2]["codes"]


def test_media_messages_put_the_media_where_the_state_goes():
    msgs = media_messages("Paragon nr 7", "Czy zapłacono kartą?", ["Tak", "Nie"], ["image", "audio"], "pl", 2)
    assert msgs[0] == {"role": "system", "content": FORMATS[2]["templates"]["pl"]["system"]}
    assert msgs[1]["content"] == [
        {"type": "text", "text": "Stan:\n"}, {"type": "image"}, {"type": "audio"},
        {"type": "text", "text": f"\nParagon nr 7\n\nPytanie: Czy zapłacono kartą?\nOpcje:\n{CODES[0]}. Tak\n{CODES[1]}. Nie"}]
    only = media_messages("", "Which topic?", ["sport", "politics", "science"], ["audio"], "en", 2)[1]["content"]
    assert only[-1]["text"].startswith("\n\nQuestion: Which topic?")  # no state text: the media alone


def test_media_messages_without_media_is_the_text_template():
    t = FORMATS[2]["templates"]["pl"]
    parts = media_messages("", "Pytanie?", ["a", "b"], [], "pl", 2)[1]["content"]
    opts = f"{CODES[0]}. a\n{CODES[1]}. b"
    assert "".join(p["text"] for p in parts) == t["user"].format(state="", q="Pytanie?", opts=opts)


def test_kinds_and_audio_window_from_the_model_files():
    assert media.kinds_of({"vision_config": {"a": 1}, "audio_config": {"b": 2}}) == ("image", "audio")
    assert media.kinds_of({"vision_config": {"a": 1}}) == ("image",)
    assert media.kinds_of({}) == ()
    assert media.audio_window({"audio_seq_length": 750, "audio_ms_per_token": 40}) == 30.0
    assert media.audio_window({}) is None
    assert media.detail_tokens("default") == 280 and media.detail_tokens("high") == 1120
    with pytest.raises(ValueError):
        media.detail_tokens("ultra")


def png(w=24, h=16, mode="RGBA"):
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new(mode, (w, h), (200, 10, 10, 255) if mode == "RGBA" else 128).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def wav(seconds=1.0, sr=44100, channels=2):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy.signal")
    t = np.arange(int(seconds * sr)) / sr
    x = np.stack([np.sin(2 * np.pi * 440 * t)] * channels, axis=1).astype("float32") * 0.1
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_image_and_audio():
    imgs, auds, order = media.decode([{"type": "image", "data": png()}, {"type": "audio", "data": wav()},
                                      {"type": "image", "data": "data:image/png;base64," + png(8, 8, "L")}],
                                     ("image", "audio"), 30.0)
    assert order == ["image", "audio", "image"]
    assert [im.mode for im in imgs] == ["RGB", "RGB"] and imgs[0].size == (24, 16)
    assert auds[0].dtype.name == "float32" and auds[0].ndim == 1 and abs(len(auds[0]) - media.SAMPLE_RATE) <= 1


@pytest.mark.parametrize("entries,kinds,msg", [
    ([], ("image",), "non-empty list"),
    ([{"type": "video", "data": "AAAA"}], ("image",), 'type must be "image" or "audio"'),
    ([{"type": "audio", "data": "AAAA"}], ("image",), "does not take audio"),
    ([{"type": "image", "data": "not base64!"}], ("image",), "not valid base64"),
    ([{"type": "image", "data": base64.b64encode(b"plain text, not an image").decode()}], ("image",),
     "not a readable image"),
    ([{"type": "image", "data": "AAAA"}] * (media.MAX_ITEMS + 1), ("image",), "at most"),
])
def test_decode_refuses(entries, kinds, msg):
    pytest.importorskip("PIL")
    with pytest.raises(ValueError, match=msg):
        media.decode(entries, kinds, 30.0)


def test_decode_limits(monkeypatch):
    big = png(20, 20)
    monkeypatch.setattr(media, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="pixels|not a readable image"):
        media.decode([{"type": "image", "data": big}], ("image",))
    with pytest.raises(ValueError, match="more than the model's 0.5 s"):
        media.decode([{"type": "audio", "data": wav(1.0)}], ("audio",), 0.5)
    with pytest.raises(ValueError, match="not readable audio"):
        media.decode([{"type": "audio", "data": base64.b64encode(b"RIFF....junk").decode()}], ("audio",), 30.0)
    monkeypatch.setattr(media, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="MB"):
        media.decode([{"type": "image", "data": png()}], ("image",))


class MediaServer(FakeServer):
    """FakeServer in prompt format 2 with a stub eager backend that takes images and audio."""
    def __init__(self, pos_probs, kinds=("image", "audio")):
        super().__init__(pos_probs)
        self.fmt, self.max_options, self.orders = 2, 255, 1
        self.calls = []
        srv = self

        class B:
            policies, media_kinds, max_audio_s = {}, kinds, 30.0

            def run_media(self, msgs, ids, images, audio, soft, max_tokens=None):
                srv.calls.append(dict(msgs=msgs, n_images=len(images), n_audio=len(audio), soft=soft))
                return [(list(srv.pos_probs), 300, 280) for _ in msgs]
        self.backend = B()

    def code_ids(self, prompt, k):
        return list(range(k))


def test_decide_with_media_reads_the_media_path():
    srv = MediaServer([0.8, 0.2])
    body = {"state": "", "lang": "pl", "media": [{"type": "image", "data": png()}], "detail": "high",
            "questions": {"q": {"type": "noul", "instructions": "Czy na zdjęciu jest paragon?"}}}
    r = asyncio.run(srv._run(body))
    assert srv.entries == 0  # nothing went through the text queue
    (call,) = srv.calls
    assert call["n_images"] == 1 and call["n_audio"] == 0 and call["soft"] == 1120
    user = call["msgs"][0][1]["content"]
    assert user[0] == {"type": "text", "text": "Stan:\n"} and user[1] == {"type": "image"}
    a = r["answers"]["q"]
    assert a["type"] == "noul" and abs(a["noul"] - 0.8) < 1e-6 and a["calibration"] == "text"
    assert r["usage"]["media"] == {"images": 1, "audio": 0, "media_tokens": 280, "detail": "high"}
    assert r["usage"]["input_tokens"] == 300


def test_media_refused_without_media_support_and_with_early_exit():
    srv = FakeServer([0.7, 0.3])
    body = {"state": "x", "media": [{"type": "image", "data": "AAAA"}],
            "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    with pytest.raises(ValueError, match="does not take media"):
        asyncio.run(srv._run(body))
    srv = MediaServer([0.7, 0.3])
    with pytest.raises(ValueError, match="early_exit"):
        asyncio.run(srv._run(dict(body, early_exit="0.99")))
    with pytest.raises(ValueError, match="lang"):
        asyncio.run(srv._run(dict(body, media=None, lang="de")))


def test_text_request_unchanged_on_a_media_server():
    srv = MediaServer([0.6, 0.4])
    r = asyncio.run(srv._run({"state": "Paczka dotarła uszkodzona.",
                              "questions": {"q": {"type": "noul", "instructions": "Czy to reklamacja?"}}}))
    assert srv.calls == [] and srv.entries == 1
    assert "calibration" not in r["answers"]["q"] and "media" not in r["usage"]


def test_model_list_names_the_media():
    assert models_payload("basal-omni", "eager", {}, media=("image", "audio"))["models"][0]["media"] == ["image", "audio"]
    assert "media" not in models_payload("basal-omni", "eager", {})["models"][0]
