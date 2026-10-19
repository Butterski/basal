"""The Gemma 4 (basal-omni) decoder forward of the eager backend (basal 2.2.1). It runs the layers of the transformers
model (the same modules and weights) in its own loop, for three things the transformers forward cannot do:

- **SOAM (state once, ask many):** the prompts of a group (the option orders of a question; with SOAM every question
  and option order of a request) share a token prefix: the chat template, the state and, with media, the image /
  audio soft tokens. The prefix runs once into keys and values; the rest of every prompt (its suffix) then runs as one
  right-padded batch over them, positions continuing from the prefix. Each suffix sees the prefix and its own earlier
  tokens only, so its readout equals the forward of its whole prompt (up to float rounding). Prefixes of several groups
  are batched together, a long one runs alone in CHUNK-token parts; a single prompt longer than CHUNK runs its parts
  the same way (the transformers path's chunked prefill). Sliding-window layers keep the last window - 1 keys of a
  prefix only (the most a later token can see), global layers all of them.
- **Answer rows only (G1):** the layers from the last one with its own keys and values onwards matter only at the
  readout positions: Gemma 4 E2B / E4B reuse the keys and values of earlier layers in their last layers (E4B: 18 of
  42), so those layers produce nothing a later token reads. Such layers, and the attention and MLP of the last layer
  with its own keys and values (on 12B / 31B the last layer), run at the answer rows only: one row per prompt, its
  query attending the stored keys. A prefix stops after the keys and values of that layer. Same maths.
- **Media once per request:** the image and audio towers run once for all prompts of a request (the prompts carry the
  same media), and the soft tokens are part of the shared prefix. On the larger Gemma 4 models
  (use_bidirectional_attention "vision") the soft tokens of one image attend to each other in the sliding-window
  layers, as transformers builds the mask; a prefix part never ends inside an image.

Attention is torch SDPA with an explicit boolean mask (the math kernel; EagerBackend turns the others off for Gemma 4).
"""
import os
import sys

import torch
import torch.nn.functional as F


def text_model(model):
    """The decoder of a Gemma 4 model: Gemma4(Unified)ForConditionalGeneration -> model.language_model,
    Gemma4ForCausalLM -> model."""
    return getattr(model.model, "language_model", model.model)


class Gemma4Forward:
    # memory bounds as in EagerBackend: prefix parts of at most CHUNK tokens; at most MAX_BATCH_TOKENS padded tokens and
    # ATTN_PAIRS query-key pairs (rows x query tokens x keys: the attention scores of the math kernel) per forward,
    # the pairs of one CHUNK-token part over a 16k-token context
    CHUNK = 2048
    MAX_BATCH_TOKENS = 16384
    ATTN_PAIRS = 2048 * 16384

    def __init__(self, model, answer_rows=True):
        self.model = model
        lm = self.lm = text_model(model)
        cfg = self.cfg = lm.config
        self.layers = list(lm.layers[: cfg.num_hidden_layers])
        self.types = list(cfg.layer_types)
        self.window = cfg.sliding_window
        self.own = [not layer.self_attn.is_kv_shared_layer for layer in self.layers]
        # the last layer with its own keys and values; a shared layer reads the last such layer of its type (the
        # layer transformers marks store_full_length_kv)
        self.last_kv = max(i for i, o in enumerate(self.own) if o)
        self.src = {i: max(j for j in range(len(self.layers)) if self.own[j] and self.types[j] == t)
                    for i, t in enumerate(self.types) if not self.own[i]}
        self.store = set(self.src.values())
        self.ple = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
        self.bidir = getattr(cfg, "use_bidirectional_attention", None) == "vision"
        self.rope = sys.modules[type(self.layers[0].self_attn).__module__].apply_rotary_pos_emb
        self.answer_rows = answer_rows
        top = model.config
        self.media_ids = {k: getattr(top, f"{k}_token_id", None) for k in ("image", "video", "audio")}
        self.pad = cfg.pad_token_id

    # -- embeddings ---------------------------------------------------------------------------------------------------
    def _embed(self, ids, media=None, offsets=None):
        """Token ids [B, S] -> (embeddings [B, S, D], per-layer inputs [B, S, L, P] or None). media: {kind: soft
        tokens [n, D]} of the request; a row's media tokens take them from offsets[b][kind] on (the number of that
        kind's tokens before the row in its prompt). As transformers: media positions read the pad token's rows."""
        lm = self.lm
        kinds = [(k, self.media_ids[k]) for k in (media or {}) if self.media_ids.get(k) is not None]
        llm = ids
        if kinds:
            mm = torch.zeros_like(ids, dtype=torch.bool)
            for _, t in kinds:
                mm |= ids == t
            llm = torch.where(mm, self.pad, ids)
        h = lm.embed_tokens(llm)
        for k, t in kinds:
            m = ids == t
            counts = m.sum(1).tolist()
            if not sum(counts):
                continue
            feats = torch.cat([media[k][offsets[b][k]: offsets[b][k] + c] for b, c in enumerate(counts) if c])
            h = h.masked_scatter(m[..., None], feats.to(h.dtype))
        pli = None
        if self.ple:
            pli = lm.project_per_layer_inputs(h, lm.get_per_layer_inputs(llm, None))
        return h, pli

    def _blocks(self, ids):
        """Vision block ids [B, S] (-1: not an image / video soft token): runs of soft tokens, as transformers'
        get_block_sequence_ids_for_mask; None when the model attends causally only."""
        if not self.bidir:
            return None
        vis = torch.zeros_like(ids, dtype=torch.bool)
        for k in ("image", "video"):
            if self.media_ids.get(k) is not None:
                vis |= ids == self.media_ids[k]
        if not vis.any():
            return None
        start = vis & ~F.pad(vis, (1, 0))[:, :-1]
        return torch.where(vis, start.long().cumsum(1) - 1, -1)

    # -- one forward --------------------------------------------------------------------------------------------------
    def _mask(self, t, qpos, kc, pos, valid, qblk):
        """Boolean attention mask [B, 1, Sq, Lc + S] of layer type t: keys of earlier passes (positions kc [B, Lc],
        -1 = none) then this pass's tokens (positions pos [B, S]; valid = real tokens, a padding query sees itself so
        that no row is empty)."""
        sliding = t == "sliding_attention"
        d = qpos[:, :, None] - pos[:, None, :]
        m = d >= 0
        if sliding and qblk is not None:
            m = m | ((qblk[:, :, None] == qblk[:, None, :]) & (qblk[:, :, None] >= 0))
        if sliding:
            m = m & (d < self.window)
        m = m & (valid[:, None, :] | (d == 0))
        if kc is not None:
            dc = qpos[:, :, None] - kc[:, None, :]
            mc = (kc[:, None, :] >= 0) & (dc >= 0)
            if sliding:
                mc = mc & (dc < self.window)
            m = torch.cat([mc, m], -1)
        return m[:, None]

    def _q(self, a, x, cs):
        q = a.q_norm(a.q_proj(x).view(*x.shape[:2], -1, a.head_dim))
        return self.rope(q, *cs, unsqueeze_dim=2).transpose(1, 2)

    def _kv(self, a, x, cs):
        shape = (*x.shape[:2], -1, a.head_dim)
        k = a.k_proj(x).view(shape)
        v = a.v_proj(x).view(shape) if a.v_proj is not None else k
        return self.rope(a.k_norm(k), *cs, unsqueeze_dim=2).transpose(1, 2), a.v_norm(v).transpose(1, 2)

    def _rest(self, layer, h, o, pli):
        """The decoder layer after attention (transformers' Gemma 4 decoder layer: MLP, MoE block, per-layer input)."""
        o = layer.self_attn.o_proj(o.transpose(1, 2).reshape(*h.shape[:2], -1))
        h = h + layer.post_attention_layernorm(o)
        y = layer.mlp(layer.pre_feedforward_layernorm(h))
        if getattr(layer, "enable_moe_block", False):
            flat = h.reshape(-1, h.shape[-1])
            _, w, idx = layer.router(flat)
            y2 = layer.experts(layer.pre_feedforward_layernorm_2(flat), idx, w).reshape(h.shape)
            y = layer.post_feedforward_layernorm_1(y) + layer.post_feedforward_layernorm_2(y2)
        h = h + layer.post_feedforward_layernorm(y)
        if pli is not None:
            g = layer.act_fn(layer.per_layer_input_gate(h)) * pli
            h = h + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
        return h * layer.layer_scalar

    def _pass(self, h, pli, pos, valid, ctx=None, grp=None, read=None, blk=None):
        """One forward over token rows [B, S]: embeddings h [B, S, D], per-layer inputs pli [B, S, L, P] or None,
        positions pos [B, S], valid [B, S] (right padding), blk vision block ids or None. ctx: (kv, kpos) of earlier
        tokens, kv {own layer: (K, V) [G, kv heads, Lc, d]}, kpos {layer type: positions [G, Lc]}; row b continues
        ctx row grp[b]. read [B]: the index of each row's readout token, or None for a prefix part (it stops after the
        keys and values of the last layer that has its own).
        -> ({own layer: (K, V) of this pass's tokens}, final hidden state at `read` [B, D] or None)."""
        B, S = valid.shape
        kv, kpos = ctx if ctx is not None else ({}, {})
        if ctx is not None and grp is None:
            grp = torch.zeros(B, dtype=torch.long, device=pos.device)
        types = set(self.types)
        rope = {t: self.lm.rotary_emb(h, pos, t) for t in types}
        masks = {t: self._mask(t, pos, kpos[t][grp] if t in kpos else None, pos, valid, blk) for t in types}
        new, kept, sliced = {}, {}, False
        rows = None if read is None else (torch.arange(B, device=pos.device), read)
        # G1: the layers from the last one with its own keys and values on run at the answer rows only
        cut = self.last_kv if self.answer_rows and read is not None and S > 1 else None
        for i, layer in enumerate(self.layers):
            a, t = layer.self_attn, self.types[i]
            x = layer.input_layernorm(h)
            if self.own[i]:  # every own layer comes before the cut: its keys and values at every position
                k, v = self._kv(a, x, rope[t])
                new[i] = (k, v)
                if i in kv:
                    K, V = kv[i]
                    K = K.expand(B, -1, -1, -1) if K.shape[0] == 1 else K.index_select(0, grp)
                    V = V.expand(B, -1, -1, -1) if V.shape[0] == 1 else V.index_select(0, grp)
                    k, v = torch.cat([K, k], 2), torch.cat([V, v], 2)
                if i in self.store:
                    kept[t] = (k, v)
            else:
                k, v = kept[t]
            if read is None and i == self.last_kv:
                break
            if i == cut:
                h, x = h[rows][:, None], x[rows][:, None]
                pli = pli[rows][:, None] if pli is not None else None
                rope = {t2: (c[rows][:, None], s[rows][:, None]) for t2, (c, s) in rope.items()}
                masks = {t2: m[rows[0], :, rows[1]][:, :, None] for t2, m in masks.items()}
                sliced = True
            q = self._q(a, x, rope[t])
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=masks[t], scale=a.scaling, enable_gqa=True)
            h = self._rest(layer, h, o, pli[:, :, i] if pli is not None else None)
        if read is None:
            return new, None
        return new, self.lm.norm(h[:, 0] if sliced else h[rows])

    # -- scheduling ---------------------------------------------------------------------------------------------------
    def _vision(self, tok):
        return self.bidir and tok in (self.media_ids.get("image"), self.media_ids.get("video"))

    def _run_start(self, ids, n):
        """n moved back to the start of the image its cut would split (a prefix never ends inside an image)."""
        while 0 < n < len(ids) and self._vision(ids[n - 1]) and self._vision(ids[n]):
            n -= 1
        return n

    def _parts(self, ids, n):
        """CHUNK-token parts [(start, end)] of ids[:n], none ending inside an image (a longer image is one part)."""
        out, s = [], 0
        while s < n:
            e = min(s + self.CHUNK, n)
            if e < n:
                e = self._run_start(ids, e)
                while e <= s or (e < n and self._vision(ids[e - 1]) and self._vision(ids[e])):
                    e += 1
            out.append((s, e))
            s = e
        return out

    def _rows(self, seqs, starts, media=None, prompts=None):
        """Token rows (right-padded) -> (embeddings, per-layer inputs, positions, valid, block ids, lengths); a row
        starts at position starts[b] of its prompt (prompts[b], for the media offsets)."""
        dev = self.lm.embed_tokens.weight.device
        S = max(map(len, seqs))
        ids = torch.tensor([list(s) + [self.pad] * (S - len(s)) for s in seqs], device=dev)
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        ar = torch.arange(S, device=dev)
        valid = ar[None] < lens[:, None]
        pos = torch.tensor(starts, device=dev)[:, None] + ar[None]
        offsets = None
        if media:
            offsets = [{k: sum(1 for x in p[:s0] if x == self.media_ids[k]) for k in media}
                       for p, s0 in zip(prompts, starts)]
        h, pli = self._embed(ids, media, offsets)
        return h, pli, pos, valid, self._blocks(ids), lens

    def _trim(self, kv, kpos):
        """Keys of sliding-window layers: the last window - 1 of each row (a later token sees no earlier ones); a row's
        keys are its first ones (right padding)."""
        t, keep = "sliding_attention", self.window - 1
        if t not in kpos or kpos[t].shape[1] <= keep:
            return kv, kpos
        n = (kpos[t] >= 0).sum(1)
        idx = (n - keep).clamp_min(0)[:, None] + torch.arange(keep, device=n.device)
        ok = idx < n[:, None]
        idx = idx.clamp_max(kpos[t].shape[1] - 1)
        kpos = dict(kpos, **{t: torch.where(ok, kpos[t].gather(1, idx), -1)})
        out = {}
        for i, (K, V) in kv.items():
            if self.types[i] == t:
                g = idx[:, None, :, None].expand(-1, K.shape[1], -1, K.shape[3])
                K, V = K.gather(2, g), V.gather(2, g)
            out[i] = (K, V)
        return out, kpos

    def _prefix(self, seqs, media=None):
        """Keys and values of prefixes: short ones in one forward, a long one (alone) in parts through the keys of the
        parts before it."""
        if len(seqs) == 1 and len(seqs[0]) > self.CHUNK:
            ctx = None
            for s0, s1 in self._parts(seqs[0], len(seqs[0])):
                h, pli, pos, valid, blk, _ = self._rows([seqs[0][s0:s1]], [s0], media, seqs)
                new, _ = self._pass(h, pli, pos, valid, ctx=ctx, blk=blk)
                if ctx is None:
                    ctx = (new, {t: pos for t in set(self.types)})
                else:
                    ctx = ({i: (torch.cat([ctx[0][i][0], k], 2), torch.cat([ctx[0][i][1], v], 2))
                            for i, (k, v) in new.items()}, {t: torch.cat([p, pos], 1) for t, p in ctx[1].items()})
                ctx = self._trim(*ctx)
            return ctx
        h, pli, pos, valid, blk, _ = self._rows(seqs, [0] * len(seqs), media, seqs)
        new, _ = self._pass(h, pli, pos, valid, blk=blk)
        return self._trim(new, {t: torch.where(valid, pos, -1) for t in set(self.types)})

    def _shared(self, jobs, groups, out, media=None):
        """A batch of jobs (group, prompt indices, prefix length): their prefixes once, then every suffix over them,
        in forwards within MAX_BATCH_TOKENS tokens and ATTN_PAIRS query-key pairs."""
        ctx = self._prefix([groups[g][js[0]][:P] for g, js, P in jobs], media)
        Lc = max(p.shape[1] for p in ctx[1].values())
        rows = sorted(((b, g, j, P) for b, (g, js, P) in enumerate(jobs) for j in js),
                      key=lambda r: len(groups[r[1]][r[2]]) - r[3])
        cur = []
        for r in rows + [None]:
            if cur and r is not None:
                n, S = len(cur) + 1, len(groups[r[1]][r[2]]) - r[3]
                if n * S <= self.MAX_BATCH_TOKENS and n * S * (Lc + S) <= self.ATTN_PAIRS:
                    cur.append(r)
                    continue
            if cur:
                prompts = [groups[g][j] for _, g, j, _ in cur]
                h, pli, pos, valid, blk, lens = self._rows([p[P:] for p, (_, _, _, P) in zip(prompts, cur)],
                                                           [P for *_, P in cur], media, prompts)
                grp = torch.tensor([b for b, *_ in cur], device=pos.device)
                _, hid = self._pass(h, pli, pos, valid, ctx=ctx, grp=grp, read=lens - 1, blk=blk)
                for (_, g, j, _), x in zip(cur, hid):
                    out[g][j] = x
            cur = [r] if r is not None else []

    def _singles(self, items, groups, out, media=None):
        """Prompts without a shared prefix (each at most CHUNK tokens): batched by length, one forward each batch."""
        items = sorted(items, key=lambda x: len(groups[x[0]][x[1]]))
        cur = []
        for it in items + [None]:
            if it is not None and (not cur or (len(cur) + 1) * len(groups[it[0]][it[1]]) <= self.MAX_BATCH_TOKENS):
                cur.append(it)
                continue
            if cur:
                prompts = [groups[g][j] for g, j in cur]
                h, pli, pos, valid, blk, lens = self._rows(prompts, [0] * len(cur), media, prompts)
                _, hid = self._pass(h, pli, pos, valid, read=lens - 1, blk=blk)
                for (g, j), x in zip(cur, hid):
                    out[g][j] = x
            cur = [it] if it is not None else []

    @torch.no_grad()
    def hidden(self, groups, media=None):
        """groups: [[token ids of each prompt]] -> [[final hidden state of each prompt]] ([D] each). The prompts of a
        group share a prefix (the option orders of a question, or every question of a request); media ({kind: soft
        tokens}, media_features) are those of every prompt (one request)."""
        out = [[None] * len(g) for g in groups]
        jobs, singles = [], []
        for gi, toks in enumerate(groups):
            if len(toks) > 1:
                P = self._run_start(toks[0], min(len(os.path.commonprefix(toks)), min(map(len, toks)) - 1))
                if P > 0 and max(map(len, toks)) - P <= self.CHUNK:
                    jobs.append((gi, list(range(len(toks))), P))
                    continue
            for j, t in enumerate(toks):
                if len(t) <= self.CHUNK:
                    singles.append((gi, j))
                    continue
                # a long prompt alone: its CHUNK-token parts through the keys of the earlier ones, the last part reads
                P = self._run_start(t, (len(t) - 1) // self.CHUNK * self.CHUNK)
                if P > 0:
                    jobs.append((gi, [j], P))
                else:
                    singles.append((gi, j))
        if singles:
            self._singles(singles, groups, out, media)
        jobs.sort(key=lambda jb: jb[2])
        cur = []
        for jb in jobs + [None]:
            if jb is not None and jb[2] <= self.CHUNK and (not cur or (len(cur) + 1) * jb[2] <= self.MAX_BATCH_TOKENS):
                cur.append(jb)
                continue
            if cur:
                self._shared(cur, groups, out, media)
            cur = []
            if jb is not None and jb[2] > self.CHUNK:
                self._shared([jb], groups, out, media)
            elif jb is not None:
                cur = [jb]
        return out

    @torch.no_grad()
    def media_features(self, enc):
        """The soft tokens of a request's media from one prompt's processor output ({kind: [n, D]}): the image and audio
        towers run once for all prompts of the request."""
        mm, w = self.model.model, self.lm.embed_tokens.weight
        out = {}
        if enc.get("pixel_values") is not None:
            f = mm.get_image_features(enc["pixel_values"].to(w.device, w.dtype), enc["image_position_ids"].to(w.device),
                                      return_dict=True).pooler_output
            out["image"] = torch.cat(f, 0).to(w.dtype)
        if enc.get("input_features") is not None and enc.get("input_features_mask") is not None:
            a = mm.get_audio_features(enc["input_features"].to(w.device, w.dtype),
                                      enc["input_features_mask"].to(w.device), return_dict=True)
            out["audio"] = a.pooler_output[a.attention_mask.to(a.pooler_output.device)].to(w.dtype)
        return out
