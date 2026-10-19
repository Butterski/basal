"""The Gemma 4 (basal-omni) forward of the CUDA graph engine (GraphBackend: --mode fast, fast-nocompile; basal 2.3).
GraphBackend keeps its machinery (static shapes and one CUDA graph per shape, the prefix trie of a request in one
packed row, pass 1 over the root into key / value buffers, pass 2 over every branch with the trie mask, the state
cache); for a Gemma 4 model (Gemma4 / Gemma4Unified, text: E2B, E4B, 12B, 31B) it replays this forward instead of the
Llama one:

  forward(ids, kv=None, ro=None)                    plain causal rows from position 0 (single prompts, pass-1 roots)
  forward_suffix(ids, mask, pos, prefix, ro=None)   pass 2: rows over root keys / values with the trie mask

What differs from Llama, layer by layer as transformers' Gemma4TextDecoderLayer (the same modules are called):

- **Layer types.** Sliding-window layers (window W: E2B / E4B 512, 12B / 31B 1024; head size 256) between global layers
  (head size 512, partial rotary embeddings, on 12B / 31B one projection for keys and values: attention_k_eq_v, the
  values normalised but not rotated). A token of a packed row sees its ancestors in the trie at the positions they have
  in its own prompt, so in sliding layers the window is a condition on positions added to the trie mask: a branch
  token sees a root key at position j only when its own position is < j + W, exactly as in its prompt alone.
- **Readout rows.** From the last layer with its own keys and values on, a layer's output matters only where it is
  read: E2B / E4B reuse the keys and values of earlier layers in their last layers (E4B: the last 18 of 42), and the
  attention and MLP of the last own-key layer only feed those (on 12B / 31B the last layer is that layer). With `ro`
  (the readout positions of every row) these layers run at the readout rows only, their queries over the keys and
  values of every token; same maths, ~40% less compute on E4B.
- **Per-layer inputs** (E2B / E4B), the embedding scale, q / k / v norms, the four norms around attention and MLP, the
  layer scalar: transformers' own modules and methods.
- **Attention kernels** per layer type (`kernels`, {"sliding_attention": k, "full_attention": k}): "math" = torch
  SDPA's math kernel with a boolean mask, the kernel EagerBackend runs for Gemma 4 (the reference; queries in parts,
  so that the scores of a long row fit); "flex" = FlexAttention with block masks of the same conditions; "fa3" =
  FlashAttention 3 (sliding layers' plain rows; flex where a mask is needed). flex and fa3 run in compiled forwards
  only (uncompiled, flex_attention is a slow reference); uncompiled forwards run math.
- The output head reads the code rows in fp32 with the final logit softcap (GraphBackend.softcap).

Attention is causal for text; the bidirectional attention of image tokens (use_bidirectional_attention "vision") is
not used here, since this path takes text only (media run in the eager backend).
"""
import torch
import torch.nn.functional as F


def text_model(model):
    """The decoder of a Gemma 4 model: Gemma4(Unified)ForConditionalGeneration -> model.language_model,
    Gemma4ForCausalLM -> model. (The same helper as basal/gemma4.py of the eager backend.)"""
    return getattr(model.model, "language_model", model.model)


def is_gemma4(model):
    return str(getattr(model.config, "model_type", "")).startswith("gemma4")


def supports(model):
    """None if this forward runs the model, else why not (the MoE layers of 26B-A4B are not ported)."""
    if not is_gemma4(model):
        return "not a Gemma 4 model"
    if getattr(text_model(model).config, "enable_moe_block", False):
        return "Gemma 4 MoE layers (26B-A4B) are not in the CUDA graph engine; use --mode eager or vllm"
    return None


KERNELS = ("math", "flex", "fa3")
_FLEX = []


def _flex_attention():
    """flex_attention as it is inside the compiled forward (Inductor generates its kernel there); from uncompiled code
    (fast-nocompile, shapes without a graph, the long lane) one compiled instance with dynamic shapes, since
    uncompiled flex_attention is a slow reference implementation."""
    from torch.nn.attention.flex_attention import flex_attention
    if torch.compiler.is_compiling():
        return flex_attention
    if not _FLEX:
        _FLEX.append(torch.compile(flex_attention, dynamic=True))
    return _FLEX[0]


class Gemma4Graph:
    # "math": queries per SDPA call, so that the scores (rows x heads x queries x keys, fp32) stay within SCORES
    # elements: in CUDA graphs every captured shape keeps its own peak in the graph pool (1.1-max with 2,048-query
    # parts: 33 GB in the pool, out of memory before warm-up ended, 2026-10-20)
    SCORES = 2 ** 26
    BLOCK = 128   # "flex": block size of the block masks

    def __init__(self, model, kernels="math", compile=False, merge=False):
        lm = self.lm = text_model(model)
        cfg = self.cfg = lm.config
        n = cfg.num_hidden_layers
        self.layers = list(lm.layers[:n])
        self.types = list(cfg.layer_types[:n])
        self.window = cfg.sliding_window
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self.own = [not layer.self_attn.is_kv_shared_layer for layer in self.layers]
        assert all(self.own[: sum(self.own)]), "layers with their own keys and values come first"
        self.n_kv = sum(self.own)
        # the last layer with its own keys and values; a shared layer reads the last such layer of its type (the
        # layer transformers marks store_full_length_kv)
        self.last_kv = self.n_kv - 1
        self.src = {i: max(j for j in range(self.n_kv) if self.types[j] == t)
                    for i, t in enumerate(self.types) if not self.own[i]}
        self.store = set(self.src.values())
        # per own layer: (key / value heads, head size), the shape of its key / value buffers
        self.kv_shapes = [(a.k_proj.out_features // a.head_dim, a.head_dim)
                          for a in (layer.self_attn for layer in self.layers[: self.n_kv])]
        self.ple = bool(getattr(cfg, "hidden_size_per_layer_input", 0))
        if kernels == "fa3":  # FlashAttention 3 in the sliding-window layers, flex in the 512-wide global ones
            kernels = {"sliding_attention": "fa3", "full_attention": "flex"}
        if isinstance(kernels, str):
            kernels = {t: kernels for t in set(self.types)}
        for k in kernels.values():
            if k not in KERNELS:
                raise ValueError(f"unknown Gemma 4 attention kernel {k!r} (one of {', '.join(KERNELS)})")
        self.kernels = kernels
        if kernels.get("full_attention") == "fa3":
            raise ValueError("fa3 runs the sliding-window layers only (FlashAttention 3 takes heads up to 256 wide)")
        self.flex = any(k != "math" for k in kernels.values())  # fa3 is flex where a mask is needed
        self._bm = {}  # block masks of the plain rows per (length, layer type), for graph_lens
        self.graph_lens = set()  # the lengths of the plain graphs (GraphBackend sets them)
        self.qkv, self.gu = {}, {}  # merged projections per layer: (linear, output sizes)
        self.merged = self._merge() if merge else 0
        if compile:
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = 64
        self.run = torch.compile(self._run, dynamic=True) if compile else self._run

    @staticmethod
    def _fuse(mods):
        """One linear for several that read the same input: their weights concatenated into a new tensor, and the
        modules' own weights become views of it (transformers' forward of the same model still runs; no second
        copy). -> (linear, output sizes)."""
        w = torch.cat([x.weight.detach() for x in mods])
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=w.dtype)
        lin.weight = torch.nn.Parameter(w, requires_grad=False)
        o = 0
        for x in mods:
            x.weight = torch.nn.Parameter(w[o: o + x.out_features], requires_grad=False)
            o += x.out_features
        return lin, [x.out_features for x in mods]

    @torch.no_grad()
    def _merge(self):
        """One q / k / v GEMM per layer with its own keys and values (q / k with attention_k_eq_v) and one gate / up
        GEMM per layer, for plain bf16 / fp32 linears without bias (FP8 and torchao linears keep theirs). -> the
        number of layers merged."""
        def plain(mods):
            return all(type(x) is torch.nn.Linear and x.bias is None for x in mods) \
                and len({x.weight.dtype for x in mods}) == 1
        n = 0
        for i, layer in enumerate(self.layers):
            a, m = layer.self_attn, layer.mlp
            if self.own[i]:
                qkv = [a.q_proj, a.k_proj] + ([a.v_proj] if a.v_proj is not None else [])
                if plain(qkv):
                    self.qkv[i] = self._fuse(qkv)
            if plain([m.gate_proj, m.up_proj]):
                self.gu[i] = self._fuse([m.gate_proj, m.up_proj])
            n += i in self.qkv or i in self.gu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return n

    # -- the layer --------------------------------------------------------------------------------------------------
    def _rope(self, pos, t, dtype):
        """cos / sin [b, L, head size] of layer type t at positions pos [b, L], as Gemma4TextRotaryEmbedding (fp32,
        then the model's dtype)."""
        r = self.lm.rotary_emb
        inv = getattr(r, f"{t}_inv_freq")
        f = pos[..., None].float() * inv.float()
        e = torch.cat([f, f], -1)
        s = getattr(r, f"{t}_attention_scaling")
        return (e.cos() * s).to(dtype), (e.sin() * s).to(dtype)

    @staticmethod
    def _rot(x, cos, sin):
        """apply_rotary_pos_emb of Gemma 4 on [b, L, heads, d] with cos / sin [b, L, d]."""
        cos, sin = cos[:, :, None], sin[:, :, None]
        h = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin

    def _qkv(self, i, a, x):
        """The q, k and v projections of an own-key layer, [b, L, heads x d] (v None with attention_k_eq_v)."""
        if i in self.qkv:
            lin, sizes = self.qkv[i]
            q, k, *v = lin(x).split(sizes, -1)
            return q, k, v[0] if v else None
        return a.q_proj(x), a.k_proj(x), a.v_proj(x) if a.v_proj is not None else None

    def _kv(self, a, k, v, cs):
        """Keys and values of one own-key layer from its projections: [b, kv heads, L, d] each."""
        shape = (*k.shape[:2], -1, a.head_dim)
        k = k.view(shape)
        v = v.view(shape) if v is not None else k
        return self._rot(a.k_norm(k), *cs).transpose(1, 2), a.v_norm(v).transpose(1, 2)

    def _q(self, a, q, cs):
        q = a.q_norm(q.view(*q.shape[:2], -1, a.head_dim))
        return self._rot(q, *cs).transpose(1, 2)

    def _mlp(self, i, m, x):
        if i in self.gu:
            g, u = self.gu[i][0](x).chunk(2, -1)
            return m.down_proj(m.act_fn(g) * u)
        return m(x)

    def _rest(self, i, layer, h, o, pli):
        """The decoder layer after attention: o_proj, MLP, per-layer input, layer scalar."""
        h = h + layer.post_attention_layernorm(layer.self_attn.o_proj(o))
        h = h + layer.post_feedforward_layernorm(self._mlp(i, layer.mlp, layer.pre_feedforward_layernorm(h)))
        if pli is not None:
            g = layer.act_fn(layer.per_layer_input_gate(h)) * pli
            h = h + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
        return (h * layer.layer_scalar).to(h.dtype)  # in place in transformers (h *= scalar): the same rounding

    # -- attention ----------------------------------------------------------------------------------------------------
    @torch.compiler.disable
    def _math(self, q, k, v, m):
        """SDPA's math kernel: q [b, H, Lq, d], k / v [b, Hkv, Lk, d], m bool [b, 1, Lq, Lk] (True = attend) ->
        [b, Lq, H * d]. Long rows in parts of queries (SCORES). Outside the compiled graph (the reference kernel)."""
        from torch.nn.attention import SDPBackend, sdpa_kernel
        b, H, Lq, _ = q.shape
        n = max(16, self.SCORES // (b * H * k.shape[2]))
        outs = []
        with sdpa_kernel([SDPBackend.MATH]):
            for s in range(0, Lq, n):
                outs.append(F.scaled_dot_product_attention(q[:, :, s: s + n], k, v, attn_mask=m[:, :, s: s + n],
                                                           scale=1.0, enable_gqa=True))
        o = outs[0] if len(outs) == 1 else torch.cat(outs, 2)
        return o.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    @staticmethod
    def _flex(q, k, v, bm):
        """FlexAttention with a block mask. The 512-wide global heads take 32 x 32 tiles: Inductor's default tiles
        need more shared memory than an H100 has (Triton "out of resource", 2026-10-20)."""
        opts = {"FORCE_USE_FLEX_ATTENTION": True}
        if q.shape[-1] > 256:
            opts.update(BLOCK_M=32, BLOCK_N=32)
        o = _flex_attention()(q, k, v, block_mask=bm, scale=1.0, enable_gqa=True, kernel_options=opts)
        return o.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    @staticmethod
    def _fa3(q, k, v, W):
        """FlashAttention 3, causal with a window of W tokens (None: none), plain rows from position 0: q [b, H, L, d],
        k / v [b, Hkv, L, d] -> [b, L, H * d]."""
        import flash_attn_interface as fa3
        o = fa3.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=1.0,
                                causal=True, window_size=(W - 1 if W else -1, 0))
        o = o[0] if isinstance(o, (tuple, list)) else o
        return o.reshape(q.shape[0], q.shape[2], -1)

    def _attend(self, q, k, v, m):
        """Attention with the kernel the mask is for: a boolean mask (math), a block mask (flex) or ("fa3", window)."""
        if isinstance(m, tuple):
            return self._fa3(q, k, v, m[1])
        if isinstance(m, torch.Tensor):
            return self._math(q, k, v, m)
        return self._flex(q, k, v, m)

    # -- masks --------------------------------------------------------------------------------------------------------
    def _window(self, t):
        return self.window if t == "sliding_attention" else None

    def _plain_masks(self, L, qpos, dev):
        """Boolean masks [b, 1, Lq, L] per layer type of plain causal rows (keys at positions 0..L-1) for queries at
        qpos [b, Lq] (all positions, or the readout rows)."""
        k = torch.arange(L, device=dev)[None, None, :]
        causal = k <= qpos[:, :, None]
        out = {}
        for t in set(self.types):
            W = self._window(t)
            out[t] = (causal & (k > qpos[:, :, None] - W) if W else causal)[:, None]
        return out

    def _suffix_masks(self, mask, pos, P):
        """Pass 2: the trie mask (additive, 0 = visible) [b, 1, S, P + S] -> boolean masks per layer type; sliding
        layers also need the key within the window of the query's position (root keys at 0..P-1, branch keys at pos)."""
        allow = mask >= 0
        kpos = torch.cat([torch.arange(P, device=pos.device)[None].expand(pos.shape[0], P), pos], 1)
        out = {}
        for t in set(self.types):
            W = self._window(t)
            out[t] = allow & (kpos[:, None, :] > pos[:, :, None] - W)[:, None] if W else allow
        return out

    def _plain_block(self, L, t, dev):
        """FlexAttention block mask of a plain causal row of length L for layer type t (no batch dimension: the same
        for every row), built from block arithmetic (create_block_mask evaluates all L x L pairs: 2.5 GB at 50k
        tokens): key block j is visible from query block i when 0 <= i - j <= dmax (the window), and full (no mask
        evaluated) when 1 <= i - j <= fmax. Cached for the graph lengths."""
        key = (L, t)
        if key in self._bm:
            return self._bm[key]
        from torch.nn.attention.flex_attention import BlockMask
        W, Bk = self._window(t), self.BLOCK
        n = -(-L // Bk)
        dmax, fmax = ((W + Bk - 2) // Bk, (W - Bk) // Bk) if W else (n, n)
        width = min(dmax, n - 1) + 1
        d = torch.arange(width, device=dev)[None]
        j = torch.arange(n, device=dev)[:, None] - d                  # [n, width]: key blocks i, i - 1, ...
        full = (j >= 0) & (d >= 1) & (d <= fmax)
        part = (j >= 0) & ~full

        def pack(sel):  # selected key blocks first, ascending; one column per key block (BlockMask's layout)
            idx = torch.where(sel, j, n).sort(1).values.clamp(max=n - 1)
            idx = torch.cat([idx, idx[:, -1:].expand(n, n - width)], 1)
            return sel.sum(1).to(torch.int32)[None, None], idx.to(torch.int32)[None, None].contiguous()

        def mod(b, h, q, kv):
            return (q >= kv) & (q - kv < W) if W else q >= kv
        bm = BlockMask.from_kv_blocks(*pack(part), *pack(full), BLOCK_SIZE=Bk, mask_mod=mod, seq_lengths=(L, L))
        if L in self.graph_lens:
            self._bm[key] = bm
        return bm

    def _dense_block(self, B, Lq, Lk, mod, dev):
        """A block mask that visits every (query block, key block) pair, each evaluated with mod: its block lists do
        not depend on the values the mask reads, so a CUDA graph keeps it and only those tensors change."""
        from torch.nn.attention.flex_attention import BlockMask
        nq, nk = -(-Lq // self.BLOCK), -(-Lk // self.BLOCK)
        num = torch.full((B, 1, nq), nk, dtype=torch.int32, device=dev)
        idx = torch.arange(nk, dtype=torch.int32, device=dev).expand(B, 1, nq, nk).contiguous()
        return BlockMask.from_kv_blocks(num, idx, BLOCK_SIZE=self.BLOCK, mask_mod=mod, seq_lengths=(Lq, Lk))

    def _rows_block(self, ro, L, t):
        """Block mask of the readout rows ro [b, R] of plain causal rows of length L."""
        W, R = self._window(t), ro.shape[1]

        def mod(b, h, q, kv):  # indices past the end (the last block's padding) are clamped, their result unused
            p = ro[b, q.clamp(max=R - 1)]
            return (p >= kv) & (p - kv < W) if W else p >= kv
        return self._dense_block(ro.shape[0], R, L, mod, ro.device)

    def _tree_block(self, tm, pos, P, t, ro=None):
        """Block mask of pass 2 (the TreeMask of basal/lean.py: root tokens per row, block of every branch token,
        ancestors per block) with the window on positions in sliding layers; ro: the readout rows only."""
        from .lean import tree_mask_mod
        W = self._window(t)
        base = tree_mask_mod(tm, P)
        S = pos.shape[1]

        def mod(b, h, q, kv):
            qq = ro[b, q.clamp(max=ro.shape[1] - 1)] if ro is not None else q.clamp(max=S - 1)
            ok = base(b, h, qq, kv)
            if W:
                kp = torch.where(kv < P, kv, pos[b, (kv - P).clamp(min=0, max=S - 1)])
                ok = ok & (pos[b, qq] - kp < W)
            return ok
        Lq = S if ro is None else ro.shape[1]
        return self._dense_block(pos.shape[0], Lq, P + S, mod, pos.device)

    # -- the forward --------------------------------------------------------------------------------------------------
    def _masks(self, kernels, L, pos, prefix, mask, ro):
        """The attention masks of one forward per layer type, for every query and for the readout rows ro: boolean
        masks for "math", block masks for "flex", ("fa3", window) for FlashAttention 3 over plain rows. Built
        outside the compiled forward (a CUDA graph keeps them)."""
        from .lean import TreeMask
        dev, types = pos.device, set(self.types)
        math = [t for t in types if kernels[t] == "math"]
        flex = [t for t in types if kernels[t] != "math"]  # fa3: FlashAttention 3 for every query of plain rows
        m_all, m_ro = {}, {}
        if prefix is None:
            if math:
                m_all = self._plain_masks(L, pos, dev)
                m_ro = self._plain_masks(L, ro, dev) if ro is not None else {}
            for t in flex:
                m_all[t] = ("fa3", self._window(t)) if kernels[t] == "fa3" else self._plain_block(L, t, dev)
                if ro is not None:
                    m_ro[t] = self._rows_block(ro, L, t)
            return m_all, m_ro
        P = prefix[0][0].shape[2]
        if isinstance(mask, TreeMask):
            if math:
                raise ValueError("the trie as a TreeMask needs the flex kernel in every layer type")
        elif math:
            m_all = self._suffix_masks(mask, pos, P)
            if ro is not None:
                b, R = ro.shape
                m_ro = {t: m.gather(2, ro[:, None, :, None].expand(b, 1, R, m.shape[-1])) for t, m in m_all.items()}
        if flex and not isinstance(mask, TreeMask):
            raise ValueError("the flex kernel takes the trie of pass 2 as a TreeMask")
        for t in flex:
            m_all[t] = self._tree_block(mask, pos, P, t)
            if ro is not None:
                m_ro[t] = self._tree_block(mask, pos, P, t, ro)
        return m_all, m_ro

    def _run(self, ids, pos, m_all, m_ro, kv=None, prefix=None, ro=None, between=None):
        """ids / pos [b, L]; m_all / m_ro: the masks per layer type of every query and of the readout rows (_masks;
        the kernel follows the mask); kv: per own layer (K, V) buffers [b, kv heads, L, d] that receive this pass's
        keys and values (pass 1); prefix: per own layer (K, V) [b, kv heads, P, d] the rows continue (pass 2); ro
        [b, R]: readout positions -> final hidden states [b, R, D] (None: every position, [b, L, D]); between(i) is
        called before layer i (the long lane)."""
        h = self.lm.embed_tokens(ids)
        pli = self.lm.project_per_layer_inputs(h, self.lm.get_per_layer_inputs(ids, None)) if self.ple else None
        types = set(self.types)
        cs = {t: self._rope(pos, t, h.dtype) for t in types}
        masks, kept = m_all, {}
        for i, layer in enumerate(self.layers):
            if between is not None:
                between(i)
            a, t = layer.self_attn, self.types[i]
            x = layer.input_layernorm(h)
            q = None
            if self.own[i]:
                q, k, v = self._qkv(i, a, x)
                k, v = self._kv(a, k, v, cs[t])
                if kv is not None:
                    kv[i][0].copy_(k)
                    kv[i][1].copy_(v)
                if prefix is not None:
                    k, v = torch.cat([prefix[i][0], k], 2), torch.cat([prefix[i][1], v], 2)
                if i in self.store:
                    kept[t] = (k, v)
            else:
                k, v = kept[t]
            if ro is not None and i == self.last_kv:  # from here on, the readout rows only
                g = ro[:, :, None]
                h, x = h.gather(1, g.expand(-1, -1, h.shape[-1])), x.gather(1, g.expand(-1, -1, x.shape[-1]))
                if pli is not None:
                    pli = pli.gather(1, ro[:, :, None, None].expand(-1, -1, *pli.shape[2:]))
                cs = {u: tuple(c.gather(1, g.expand(-1, -1, c.shape[-1])) for c in cs[u]) for u in types}
                if q is not None:
                    q = q.gather(1, g.expand(-1, -1, q.shape[-1]))
                masks = m_ro
            q = self._q(a, q if q is not None else a.q_proj(x), cs[t])
            o = self._attend(q, k, v, masks[t])
            h = self._rest(i, layer, h, o, pli[:, :, i] if pli is not None else None)
        return self.lm.norm(h)

    def _kernels(self, dense=False):
        """The kernels of a call: the configured ones, or the math kernel for pass 2 with a dense mask (flex takes the
        trie as a TreeMask)."""
        if not (dense and self.flex):
            return self.kernels
        return {t: "math" for t in self.kernels}

    @torch.no_grad()
    def forward(self, ids, kv=None, ro=None, between=None, compiled=True):
        """Plain causal rows [b, L] from position 0 (right padding cannot reach a real token); kv: per own layer
        (K, V) buffers [b, kv heads, L, d] that receive the keys and values (pass 1); ro [b, R]: readout positions ->
        [b, R, D] (None: [b, L, D]); between(i) before layer i (the long lane: uncompiled, one layer at a time).
        compiled: through the compiled forward (when the engine compiles; shapes without a graph run uncompiled)."""
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device)[None].expand(ids.shape[0], L)
        compiled = compiled and between is None
        m_all, m_ro = self._masks(self.kernels, L, pos, None, None, ro)
        return (self.run if compiled else self._run)(ids, pos, m_all, m_ro, kv=kv, ro=ro, between=between)

    @torch.no_grad()
    def forward_suffix(self, ids, mask, pos, prefix, ro=None, compiled=True):
        """Pass 2: rows [b, S] at positions pos over per own layer prefix (K, V) [b, kv heads, P, d], with the trie
        mask (additive [b, 1, S, P + S], 0 = visible; a TreeMask for flex) -> [b, R, D] at ro [b, R] (None: all)."""
        from .lean import TreeMask
        kernels = self._kernels(dense=not isinstance(mask, TreeMask))
        m_all, m_ro = self._masks(kernels, ids.shape[1], pos, prefix, mask, ro)
        return (self.run if compiled else self._run)(ids, pos, m_all, m_ro, prefix=prefix, ro=ro)
