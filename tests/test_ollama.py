"""Ollama and llama.cpp server backends read the letter log-probabilities from the engines' HTTP responses (mocked
HTTP server), and the GGUF / Apple Silicon modes are registered in the server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from basal.engine import LlamaCppBackend, OllamaBackend


class _Tok:
    pad_token_id = 0
    bos_token = "<s>"

    @staticmethod
    def decode(ids):
        return "".join("ABCDEFGHIJ"[i - 10] if 10 <= i < 20 else f"<{i}>" for i in ids)


class _Engine(BaseHTTPRequestHandler):
    """Mock of Ollama's /api/generate and llama.cpp's /completion: letter log-probabilities by number of options."""
    seen = []
    LP = {"A": -0.2, "B": -1.9, "C": -3.5}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, body))
        top = [{"token": t, "logprob": v, "id": 10 + "ABC".index(t)} for t, v in self.LP.items()]
        top.append({"token": "x", "logprob": -4.0, "id": 50})
        if self.path == "/api/generate":
            out = {"model": body["model"], "response": "A", "done": True,
                   "logprobs": [{"token": "A", "logprob": -0.2, "top_logprobs": top}]}
        else:
            out = {"content": "A", "completion_probabilities": [{"id": 10, "token": "A", "logprob": -0.2,
                                                                  "top_logprobs": top}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def engine_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Engine.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def http_stub(cls, url, **kw):
    import httpx
    be = cls.__new__(cls)
    be.tok, be.http, be.pool, be.parallel = _Tok(), httpx.Client(base_url=url), None, 1
    for k, v in kw.items():
        setattr(be, k, v)
    return be


def _softmax(xs):
    return torch.softmax(torch.tensor(xs), -1).tolist()


def test_ollama_backend(engine_url):
    be = http_stub(OllamaBackend, engine_url, name="basal-test", num_ctx=4096, keep_alive="5m", letter_text={},
                   checked=True)
    out = be.run_shared([(["<s>prompt one", "prompt two"], [[10, 11, 12], [10, 11, 12, 13]])])[0]
    assert out[0] == pytest.approx(_softmax([-0.2, -1.9, -3.5]), abs=1e-6)
    assert out[1][3] < 1e-6 and sum(out[1]) == pytest.approx(1.0)  # D is not in the top list
    path, body = _Engine.seen[0]
    assert path == "/api/generate" and body["raw"] is True and body["prompt"] == "prompt one"  # BOS added by Ollama
    assert body["options"]["num_predict"] == 1 and body["logprobs"] is True and body["top_logprobs"] >= 10
    assert OllamaBackend.takes_text


def test_llamacpp_backend(engine_url):
    be = http_stub(LlamaCppBackend, engine_url)
    out = be.run([[1, 2, 3, 4]], [[10, 11, 12]])[0]
    assert out == pytest.approx(_softmax([-0.2, -1.9, -3.5]), abs=1e-6)
    path, body = _Engine.seen[0]
    assert path == "/completion" and body["prompt"] == [1, 2, 3, 4] and body["n_predict"] == 1
    assert body["n_probs"] >= 10 and body["post_sampling_probs"] is False


def test_server_modes_registered():
    from basal.server import MODES, parser
    assert {"mlx", "mlx-q8", "mps", "gguf", "ollama", "llamacpp"} <= set(MODES)
    a = parser().parse_args(["--mode", "ollama", "--ollama-model", "basal:q8_0", "--calibration", "C.json"])
    assert a.ollama_model == "basal:q8_0" and a.calibration == "C.json"
    a = parser().parse_args(["--mode", "gguf", "--gguf", "basal-1.5-mini-Q8_0.gguf"])
    assert a.gguf == "basal-1.5-mini-Q8_0.gguf"
