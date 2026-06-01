import copy
import logging
from dataclasses import replace as replace_cfg
from tqdm import tqdm
import midigpt._core as _core
from midigpt._types import Score
from midigpt._converters import to_cpp, from_cpp
from midigpt.inference.config import GenerationRequest, InferenceConfig
import midigpt._core as _core # Import _core here
from midigpt._core import GenerationStep # Import GenerationStep directly from _core
import time # Moved from _sample_step


log = logging.getLogger(__name__)


class _ContextOverflow(Exception):
    """Raised when an encoded step exceeds the model's context length."""
    def __init__(self, ctx_len: int, model_max: int):
        super().__init__(f"context_len={ctx_len} >= model max {model_max}")
        self.ctx_len   = ctx_len
        self.model_max = model_max


class _KVRunner:
    """Owns the KV cache across a single sampling step.

    Hides past_kv bookkeeping from the main sampling loop. Tracks whether the
    next forward() is the prefill (no cache yet) or a decode step (cache
    populated). Delegates cache surgery to the model's ModelBase methods so the
    session code stays architecture-agnostic.
    """

    def __init__(self, model, initial_kv):
        self._model      = model
        self._initial_kv = initial_kv   # cached empty-kv from engine.warmup()
        self._past_kv    = None

    @property
    def is_prefill(self) -> bool:
        return self._past_kv is None

    def forward(self, ctx_t, key_mask=None, position_ids=None):
        """Run one model call. Returns logits tensor (full output[0]).

        On the first call past_kv is sourced from the cached empty-kv (avoids
        a redundant make_empty_kv() per step). On subsequent calls the cache
        from the previous step is reused.
        """
        kwargs = {}
        if key_mask    is not None: kwargs["key_mask"]     = key_mask
        if position_ids is not None: kwargs["position_ids"] = position_ids
        pkv = self._past_kv if self._past_kv is not None else self._initial_kv
        try:
            outputs = (self._model(ctx_t, pkv, **kwargs) if pkv is not None
                       else self._model(ctx_t, **kwargs))
        except Exception:
            # TorchScript callables with strict signatures may reject kwargs
            # or kv on certain code paths — fall back to positional-only.
            outputs = self._model(ctx_t)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self._past_kv = outputs[1] if len(outputs) > 1 else None
        return outputs[0]

    def null_positions(self, spans):
        """Neutralize KV at the given (s, e) spans (attention_approx)."""
        if self._past_kv is None or not spans:
            return
        self._model.kv_null_positions(self._past_kv, spans)


class SamplingSession:
    def __init__(self, engine, score: Score, request: GenerationRequest):
        self._engine  = engine
        self._score   = score
        self._request = request
        self.model_forward_time: float = 0.0
        self.encode_time: float = 0.0
        self.decode_time: float = 0.0
        self.enable_profiling: bool = False
        self.gen_count: int = 0
        # Optional callback fired once per run() with a per-bar snapshot of
        # the first step's prompt state (CONTEXT/MASKED/TO_GENERATE per
        # (track, bar)). Used for client-side comparison/debug; pure read-only.
        self.prompt_state_sink = None
        self._snapshot_sent = False

    def __enter__(self): return self
    def __exit__(self, *_): pass

    def run(self) -> Score:
        self.gen_count = 0
        self.model_forward_time = 0.0
        self.encode_time = 0.0
        self.decode_time = 0.0
        self._snapshot_sent = False

        mask    = self._build_selection_mask()
        cfg     = self._request.config
        enc_cfg = self._engine._tokenizer._vocab.config()
        import json as _json
        try:
            dims = sorted(
                set(_json.loads(enc_cfg.to_json()).get("num_bars_map") or []),
                reverse=True,
            )
        except Exception:
            dims = []
        # Candidate model_dims to try, largest-first, capped at the requested
        # value. If num_bars_map is unavailable, just try the requested dim.
        candidates = [d for d in dims if d <= cfg.model_dim] or [cfg.model_dim]

        last_exc = None
        for d in candidates:
            cfg_try = replace_cfg(cfg, model_dim=d) if d != cfg.model_dim else cfg
            enc_cfg.model_dim = d
            planner = _core.StepPlanner(
                mask, enc_cfg,
                cfg_try.bars_per_step, cfg_try.tracks_per_step
            )
            py_score = self._score if isinstance(self._score, Score) else from_cpp(self._score)
            score = to_cpp(copy.deepcopy(py_score))
            try:
                self._snapshot_sent = False
                for step in tqdm(planner.plan()):
                    score = self._run_step(score, step, cfg_try)
                return score
            except _ContextOverflow as exc:
                last_exc = exc
                log.warning(
                    "context overflow at model_dim=%d (ctx=%d, max=%d); "
                    "stepping down", d, exc.ctx_len, exc.model_max,
                )
                continue
        raise RuntimeError(
            f"no model_dim in {candidates} fits the model "
            f"(last: {last_exc})"
        )

    def _run_step(self, score: Score, step: GenerationStep,
                  cfg: InferenceConfig | None = None) -> Score:
        if cfg is None:
            cfg = self._request.config
        # Normalise to _types.Score so helpers can always use bar.notes.
        # On the first step the caller passes a _core.Score; from step 2 on
        # it's already _types.Score (the previous step's candidate).
        if not isinstance(score, Score):
            score = from_cpp(score)
        original_score = score  # baseline for novelty comparison

        errors = []
        base_seed = getattr(cfg, "seed", -1)
        last_candidate = None
        last_diagnostics = None
        for i in range(cfg.max_attempts):
            temperature = cfg.temperature * (cfg.temperature_escalation ** i)
            # Bump the seed per retry when the user pinned one — otherwise all
            # attempts would resample the same tokens deterministically and
            # max_attempts would be a no-op. seed<0 keeps the RNG free-running.
            attempt_seed = (base_seed + i) if base_seed is not None and base_seed >= 0 else -1
            candidate = self._sample_step(score, step, temperature, attempt_seed)
            last_candidate = candidate

            # Always evaluate BOTH checks. Whether a failure is an error or just
            # a warning depends on whether the corresponding check is enabled.
            silence_failed = self._note_count(candidate, step) <= 0
            novelty_failed = self._is_identical(original_score, candidate)
            last_diagnostics = (silence_failed, novelty_failed)

            attempt_errors = []
            attempt_warnings = []
            if silence_failed:
                msg = "silence_check (no notes generated in target bars)"
                (attempt_errors if cfg.silence_check else attempt_warnings).append(msg)
            if novelty_failed:
                msg = "novelty_check (candidate identical to original)"
                (attempt_errors if cfg.novelty_check else attempt_warnings).append(msg)

            for w in attempt_warnings:
                log.warning("attempt %d/%d accepted with warning: %s "
                            "(check disabled)", i + 1, cfg.max_attempts, w)
            if not attempt_errors:
                return candidate

            errors.extend(attempt_errors)
            log.info("attempt %d/%d rejected (%s); retrying with temp=%.2f, seed=%s",
                     i + 1, cfg.max_attempts, "; ".join(attempt_errors),
                     temperature * cfg.temperature_escalation, attempt_seed)

        from collections import Counter
        reason_summary = ", ".join(
            f"{r}×{c}" for r, c in Counter(errors).most_common()
        )
        raise RuntimeError(
            f"Max attempts ({cfg.max_attempts}) reached, no acceptable candidate "
            f"found. Rejection reasons: {reason_summary}. "
            f"Last candidate had {self._note_count(last_candidate, step)} notes in "
            f"generated bars. "
            f"(Hints: vary retries with temperature_escalation > 1, raise "
            f"max_attempts, disable the failing check, or relax top_p/mask_p "
            f"for broader sampling.)"
        )

    def _sample_step(self, score: Score, step, temperature: float, seed: int = -1) -> Score:
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]")

        # Seed the global torch RNG so torch.multinomial below is reproducible
        # when the user pins a seed. seed<0 means "leave the RNG state alone"
        # — i.e. continue from wherever the global generator currently is.
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        # Three attribute regimes:
        #  - Full AR  (tp.autoregressive, no prefix bars): skip analyzer; user
        #    attributes are forced via AttributeValueConstraint so the model
        #    emits the requested tokens. The encoder emits no attribute tokens
        #    for this track.
        #  - Partial AR (tp.autoregressive, prefix bars exist): treat like
        #    infill — run the analyzer over the prefix, let user attributes
        #    override in the prompt, NO constraint (the prompt already pins
        #    attribute tokens for the prefix bars).
        #  - Infill: run the analyzer; user attributes override in the prompt.
        full_ar_ids: set[int] = set()
        for tp in self._request.tracks:
            if tp.autoregressive and (not tp.bars or min(tp.bars) == 0):
                full_ar_ids.add(tp.id)
        self._full_ar_ids = full_ar_ids  # consumed by _build_constraints
        analyzer = self._engine._analyzer
        if analyzer is not None:
            for t_idx, track in enumerate(score.tracks):
                if t_idx in full_ar_ids:
                    continue
                new_attrs = dict(track.attributes)
                new_attrs.update(analyzer.compute_track_tokens(score, t_idx))
                for b_idx in range(len(track.bars)):
                    for k, v in analyzer.compute_bar_tokens(score, t_idx, b_idx).items():
                        new_attrs[f"bar_{k}_{b_idx}"] = v
                track.attributes = new_attrs

        # User attribute overrides flow into the prompt for infill + partial AR.
        # Full AR uses constraints instead (see _build_constraints).
        for tp in self._request.tracks:
            if tp.id < len(score.tracks) and tp.id not in full_ar_ids:
                score.tracks[tp.id].attributes.update(tp.attributes)

        # Per-bar overrides: write bar_{TokenType}_{bar_idx} keys directly into
        # the prompt, overwriting analyzer-derived values. For infill and
        # partial-AR (suffix bars), the encoder reads these straight from
        # `track.attributes`. For full-AR, per-bar overrides require a C++
        # BarAttributeValueConstraint (Phase 2) — emit a one-time warning and
        # ignore them here so the request still runs.
        for tp in self._request.tracks:
            if tp.id >= len(score.tracks):
                continue
            track_attrs = score.tracks[tp.id].attributes
            if tp.id in full_ar_ids:
                # Full-AR per-bar overrides flow through
                # BarAttributeValueConstraint in _build_constraints, not the
                # prompt. Skip the prompt-write path here.
                continue
            # Resolve attribute name -> token_type via analyzer (the same
            # key shape used by compute_bar_tokens above).
            for bar_idx, bar_dict in (tp.bar_attributes or {}).items():
                for attr_name, val in (bar_dict or {}).items():
                    attr_obj = analyzer.get(attr_name) if (analyzer and hasattr(analyzer, "get")) else None
                    if attr_obj is None:
                        continue
                    track_attrs[f"bar_{attr_obj.token_type}_{int(bar_idx)}"] = int(val)
            # Per-bar controls — map control name to its token-type string.
            for bar_idx, bar_dict in (tp.bar_controls or {}).items():
                for ctrl_name, val in (bar_dict or {}).items():
                    if ctrl_name == "time_signature":
                        track_attrs[f"bar_TimeSig_{int(bar_idx)}"] = int(val)

        # Classify every bar in the step window per the request: a bar is
        # CONTEXT unless it is a generation target (step.bars_to_generate) or
        # listed in tp.mask_bars. The C++ planner's AR branch sets ctx=False
        # for every non-yet-generated bar on AR tracks (so suffix-AR prefix
        # bars would otherwise be encoded as MASK_BAR); this enforces the
        # correct semantic: not-marked = not-masked.
        gen_set  = set(step.bars_to_generate)  # {(track_id, bar_abs), …}
        new_ctx  = [list(row) for row in step.context]
        ctx_dirty = False
        req_ids  = {tp.id: set(tp.mask_bars) for tp in self._request.tracks}
        for t_idx, row in enumerate(new_ctx):
            if t_idx not in req_ids:
                continue
            masked = req_ids[t_idx]
            for b_abs in range(step.start_bar, step.end_bar):
                if b_abs >= len(row):
                    continue
                want_ctx = ((t_idx, b_abs) not in gen_set
                            and b_abs not in masked)
                if row[b_abs] != want_ctx:
                    row[b_abs] = want_ctx
                    ctx_dirty = True
        if ctx_dirty:
            step.context = new_ctx

        # Emit the prompt-state snapshot ONCE per run() — first step only.
        # During AR later steps would flip T→C as bars get committed; the
        # caller (e.g. studio) has a single static prediction to compare
        # against, so later snapshots would never line up.
        if self.prompt_state_sink is not None and not self._snapshot_sent:
            self._snapshot_sent = True
            try:
                self.prompt_state_sink(self._snapshot_prompt_state(step))
            except Exception as exc:  # noqa: BLE001
                log.warning("prompt_state_sink failed: %s", exc)

        # Window size flows via EncodeOptions.window_bars, set inside
        # SessionState from the step. Nothing to plumb through attributes.
        t_enc = time.perf_counter()
        mask_mode = getattr(self._request.config, "mask_mode", "token")
        use_span_masks = mask_mode in ("attention", "attention_approx", "attention_skip")
        state = _core.SessionState(
            to_cpp(score), step,
            self._engine._tokenizer._vocab,
            self._build_constraints(step),
            self._engine._tokenizer._encoder,
            self._engine._tokenizer._decoder,
            use_span_masks,
            mask_mode == "remove",   # remove_future_bars
        )

        context_len = len(state.context_tokens())
        spans = state.hidden_spans() if use_span_masks else []

        # ── attention: pre-allocate bool buffer on model device (no per-token cat) ──
        if spans and mask_mode == "attention":
            _kv_dev = next(self._engine._model.parameters()).device
            _kv_cap = self._model_max_context()
            _kv_buf = torch.ones(_kv_cap, dtype=torch.bool, device=_kv_dev)
            for s, e in spans:
                _kv_buf[s:e] = False
            _kv_len = context_len
        else:
            _kv_buf = None
            _kv_len = 0

        # ── attention_approx: one-shot prefill mask (CPU, single use) ──
        if spans and mask_mode == "attention_approx":
            _prefill_mask = torch.ones(context_len, dtype=torch.bool)
            for s, e in spans:
                _prefill_mask[s:e] = False
        else:
            _prefill_mask = None
        _approx_surgery_done = False

        # ── attention_skip: build filtered token sequence and skip position_ids ──
        if spans and mask_mode == "attention_skip":
            _all_ctx = state.context_tokens()
            _keep = [True] * len(_all_ctx)
            for s, e in spans:
                for i in range(s, e):
                    _keep[i] = False
            _skip_ids   = [t for t, k in zip(_all_ctx, _keep) if k]
            _skip_pos   = [i for i, k in enumerate(_keep) if k]
            _next_pos   = _skip_pos[-1] + 1 if _skip_pos else context_len
            _skip_ctx   = torch.tensor([_skip_ids], dtype=torch.long)
            _skip_pos_t = torch.tensor([_skip_pos], dtype=torch.long)
        else:
            _skip_ctx = _skip_pos_t = None
            _next_pos = 0

        if self.enable_profiling:
            self.encode_time += time.perf_counter() - t_enc
        model_max_ctx = self._model_max_context()
        if context_len >= model_max_ctx:
            raise _ContextOverflow(context_len, model_max_ctx)
        max_gen_tokens = model_max_ctx - context_len - 1

        # Pre-allocate mask buffer once for this step (reused every token)
        vocab_size = self._engine._tokenizer.vocab_size()
        mask_buf = torch.empty(vocab_size, dtype=torch.bool)

        kv = _KVRunner(self._engine._model, self._engine._initial_kv)
        with torch.no_grad():
            while not state.complete() and self.gen_count < max_gen_tokens:
                is_prefill = kv.is_prefill

                # Build ctx and determine key_mask / position_ids for this step
                if is_prefill:
                    if mask_mode == "attention_skip" and _skip_ctx is not None:
                        ctx      = _skip_ctx
                        km       = None
                        pos_ids  = _skip_pos_t
                    else:
                        ctx      = torch.tensor([state.context_tokens()], dtype=torch.long)
                        km       = (_kv_buf[:_kv_len] if mask_mode == "attention" and _kv_buf is not None
                                    else _prefill_mask if mask_mode == "attention_approx" and _prefill_mask is not None
                                    else None)
                        pos_ids  = None
                else:
                    ctx      = torch.tensor([[state.context_tokens()[-1]]], dtype=torch.long)
                    km       = _kv_buf[:_kv_len] if mask_mode == "attention" and _kv_buf is not None else None
                    pos_ids  = (torch.tensor([[_next_pos]], dtype=torch.long)
                                if mask_mode == "attention_skip" and _skip_ctx is not None else None)

                t_fwd = time.perf_counter()
                logits_seq = kv.forward(ctx, key_mask=km, position_ids=pos_ids)
                if self.enable_profiling:
                    self.model_forward_time += time.perf_counter() - t_fwd

                # attention_approx: KV surgery after prefill — null masked positions
                if mask_mode == "attention_approx" and is_prefill and not _approx_surgery_done and spans:
                    kv.null_positions(spans)
                    _approx_surgery_done = True

                logits  = logits_seq[0, -1]

                # Reuse pre-allocated bool buffer for grammar mask
                mask_buf.copy_(torch.as_tensor(state.logit_mask(), dtype=torch.bool))
                n_legal = int(mask_buf.sum().item())
                if n_legal == 0:
                    raise RuntimeError(
                        "sampling crashed: constraint graph error "
                        "(zero legal tokens at this step — over-constrained "
                        "attribute values or incompatible grammar state)"
                    )
                masked_logits = logits.masked_fill(~mask_buf, float("-inf"))
                probs = (masked_logits / temperature).softmax(-1)

                if torch.isnan(probs.sum()) or probs.sum() < 1e-6:
                    # The model's distribution collapsed onto masked-out tokens.
                    # Do NOT disable the grammar — pick uniformly over the legal
                    # set so the sequence stays valid (decoder won't drop notes).
                    logging.debug(
                        f"  grammar-collapse: model probs vanish under "
                        f"mask (n_legal={n_legal}); sampling uniformly "
                        f"from legal tokens")
                    probs = mask_buf.to(torch.float32)
                    probs = probs / probs.sum()

                probs = self._apply_sampling_filters(probs)

                token = torch.multinomial(probs, 1).item()
                # Diagnostic: when we just sampled a melodic NoteOnset, check
                # whether the very next mask permits non-NoteDuration tokens
                # (a real-grammar bug if so).
                try:
                    tname = self._engine._tokenizer._vocab.get_type(token).name
                except Exception:
                    tname = "?"
                if tname == "NoteOnset" and logging.getLogger().isEnabledFor(logging.DEBUG):
                    state.advance(token)
                    nxt_mask = list(state.logit_mask())
                    legal_types = set()
                    for i, ok in enumerate(nxt_mask):
                        if ok:
                            try:
                                legal_types.add(
                                    self._engine._tokenizer._vocab.get_type(i).name)
                            except Exception:
                                pass
                    logging.debug(
                        f"  after NoteOnset({token}) n_legal={sum(nxt_mask)} "
                        f"legal_types={sorted(legal_types)}")
                    self.gen_count += 1
                    # attention: extend buffer pointer for generated token
                    if mask_mode == "attention" and _kv_buf is not None:
                        _kv_len += 1
                    # attention_skip: advance next position_id
                    elif mask_mode == "attention_skip" and _skip_ctx is not None:
                        _next_pos += 1
                    continue
                state.advance(token)
                self.gen_count += 1
                # attention: extend buffer pointer for generated token
                if mask_mode == "attention" and _kv_buf is not None:
                    _kv_len += 1
                # attention_skip: advance next position_id
                elif mask_mode == "attention_skip" and _skip_ctx is not None:
                    _next_pos += 1

        t_dec = time.perf_counter()
        result = from_cpp(state.result())
        if self.enable_profiling:
            self.decode_time += time.perf_counter() - t_dec
        return result

    def _snapshot_prompt_state(self, step) -> dict:
        """Per-bar prompt state for the current step, post-mask_bars patch.

        Each entry is "C" (CONTEXT), "M" (MASKED via MaskBar token),
        "A" (MASKED via attention span-mask), or "T" (TO_GENERATE).
        "A" only appears when mask_mode is one of "attention*" — purely a
        visualization label; the bar classification is identical to "M".
        """
        ar_ids = {tp.id for tp in self._request.tracks if tp.autoregressive}
        btg    = set(step.bars_to_generate)
        mask_mode = getattr(self._request.config, "mask_mode", "token")
        masked_label = "A" if mask_mode in ("attention", "attention_approx", "attention_skip") else "M"
        tracks = []
        for t_idx, row in enumerate(step.context):
            states = []
            for b_abs in range(step.start_bar, step.end_bar):
                if (t_idx, b_abs) in btg:
                    states.append("T")
                elif b_abs < len(row) and row[b_abs]:
                    states.append("C")
                else:
                    states.append(masked_label)
            tracks.append({
                "id":       t_idx,
                "is_agent": t_idx in ar_ids,
                "states":   states,
            })
        return {
            "start_bar": int(step.start_bar),
            "end_bar":   int(step.end_bar),
            "tracks":    tracks,
        }

    def _apply_sampling_filters(self, probs):
        """Apply top_k → top_p → mask_k → mask_p in that fixed order.

        Pipeline reasoning: k-filters are rank-based; p-filters are mass-based.
        Applying top filters first narrows the pool to the model's preferred
        region; mask filters then carve out the most-obvious tokens *within*
        that pool — useful for novelty (force the model off its top picks
        while staying inside its high-confidence region).

        All filters mutate a local `keep` boolean mask and return a
        renormalized probability vector. If the pool would become empty, the
        offending filter is skipped (validation catches impossible combos at
        config time; this is a runtime safety net for edge distributions).
        """
        import torch
        cfg = self._request.config
        top_p = float(getattr(cfg, "top_p", 1.0) or 1.0)
        top_k = int(getattr(cfg, "top_k", 0) or 0)
        mask_p = float(getattr(cfg, "mask_p", 0.0) or 0.0)
        mask_k = int(getattr(cfg, "mask_k", 0) or 0)
        if top_p >= 1.0 and top_k <= 0 and mask_p <= 0.0 and mask_k <= 0:
            return probs

        keep = probs > 0
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        # number of currently-positive (i.e. legal & non-filtered) tokens, used
        # to size rank-based filters relative to the remaining pool
        legal_n = int(keep.sum().item())

        # top_k: keep only the top_k highest-prob tokens
        if 0 < top_k < legal_n:
            cutoff = sorted_idx[:top_k]
            new_keep = torch.zeros_like(keep)
            new_keep[cutoff] = True
            keep = keep & new_keep

        # top_p: nucleus — keep smallest descending-prob set with cumsum ≥ top_p
        if 0.0 < top_p < 1.0:
            # rank at which cumsum first ≥ top_p (inclusive)
            nucleus_rank = int(torch.searchsorted(cumsum, torch.tensor(top_p)).item()) + 1
            nucleus_rank = max(1, min(nucleus_rank, sorted_idx.numel()))
            nucleus = sorted_idx[:nucleus_rank]
            new_keep = torch.zeros_like(keep)
            new_keep[nucleus] = True
            keep = keep & new_keep

        # Re-sort over the surviving pool for mask_* (mass/rank measured on the
        # POST-top distribution, otherwise mask_p semantics shift with top_p).
        survivor_probs = probs * keep.to(probs.dtype)
        s_total = float(survivor_probs.sum().item())
        if s_total <= 0:
            return probs / probs.sum()
        survivor_norm = survivor_probs / s_total
        sorted_probs2, sorted_idx2 = torch.sort(survivor_norm, descending=True)
        cumsum2 = torch.cumsum(sorted_probs2, dim=0)
        survivors_n = int(keep.sum().item())

        # mask_k: remove the top mask_k highest-prob (within survivors) tokens
        if 0 < mask_k < survivors_n:
            drop = sorted_idx2[:mask_k]
            keep[drop] = False

        # mask_p: remove most-likely tokens summing (cumulative) to ≥ mask_p
        if 0.0 < mask_p < 1.0 and survivors_n > 0:
            drop_rank = int(torch.searchsorted(cumsum2, torch.tensor(mask_p)).item()) + 1
            drop_rank = min(drop_rank, survivors_n - 1)  # never empty the pool
            if drop_rank > 0:
                drop = sorted_idx2[:drop_rank]
                keep[drop] = False

        filtered = probs * keep.to(probs.dtype)
        total = float(filtered.sum().item())
        if total <= 0:
            return probs / probs.sum()
        return filtered / total

    def _is_acceptable(self, original: Score, candidate: Score, cfg: InferenceConfig, step: GenerationStep) -> bool:
        logging.debug(f"Checking acceptability: silence_check={cfg.silence_check}, novelty_check={cfg.novelty_check}")

        # Dump request vs. step alignment + per-track/bar note counts.
        try:
            btg = list(step.bars_to_generate)
        except Exception:
            btg = "?"
        logging.debug(f"  step: start_bar={step.start_bar} end_bar={step.end_bar} "
                      f"track_indices={list(step.track_indices)} "
                      f"bars_to_generate={btg}")
        for i, t in enumerate(candidate.tracks):
            counts = [len(b.notes) for b in t.bars]
            logging.debug(f"  candidate.tracks[{i}] type={t.track_type} "
                          f"instrument={t.instrument} bars={len(t.bars)} "
                          f"notes_per_bar={counts}")
        for tp in self._request.tracks:
            logging.debug(f"  request.tracks tp.id={tp.id} bars={tp.bars} "
                          f"autoreg={tp.autoregressive} ignore={tp.ignore}")

        if cfg.silence_check:
            notes_added = self._note_count(candidate, step)
            logging.debug(f"  Notes added in this step (in generated bars): {notes_added}")
            if notes_added <= 0:
                logging.debug("  Rejected by silence_check: No new notes in generated bars or notes removed.")
                return False
        if cfg.novelty_check:
            if self._is_identical(original, candidate):
                logging.debug("  Rejected by novelty_check: Candidate is identical to original.")
                return False

        logging.debug("  Accepted: Candidate is acceptable.")
        return True

    def _note_count(self, score: Score, step: GenerationStep) -> int:
        count = 0
        for tp in self._request.tracks:
            # Check if this track is the one being generated in this step
            
            for bar_idx in tp.bars:
                if bar_idx >= len(score.tracks[tp.id].bars):
                    continue
                bar = score.tracks[tp.id].bars[bar_idx]
                
                # Only count notes in generated bars for the current step
                if (tp.id, bar_idx) in step.bars_to_generate:
                    for note in bar.notes:
                        if note.pitch >= 0: # Just count notes with a valid pitch
                            count += 1
        return count

    def _is_identical(self, a: Score, b: Score) -> bool:
        for tp in self._request.tracks:
            for bar_idx in tp.bars:
                ta = a.tracks[tp.id].bars[bar_idx] if tp.id < len(a.tracks) and bar_idx < len(a.tracks[tp.id].bars) else None
                tb = b.tracks[tp.id].bars[bar_idx] if tp.id < len(b.tracks) and bar_idx < len(b.tracks[tp.id].bars) else None
                if ta is None or tb is None:
                    continue
                if sorted((n.pitch, n.onset_ticks) for n in ta.notes) != \
                   sorted((n.pitch, n.onset_ticks) for n in tb.notes):
                    return False
        return True

    def _model_max_context(self) -> int:
        """Read the model's positional context length via the ModelBase protocol.

        TorchScript checkpoints are wrapped in TorchScriptAdapter at load time,
        so every model surfaces max_context().
        """
        return self._engine._model.max_context()

    def _build_selection_mask(self) -> _core.SelectionMask:
        n_tracks = len(self._score.tracks)
        n_bars   = max((len(t.bars) for t in self._score.tracks), default=0)

        mask = _core.SelectionMask()
        selected       = [[False] * n_bars for _ in range(n_tracks)]
        autoregressive = [False] * n_tracks
        ignore         = [False] * n_tracks

        for tp in self._request.tracks:
            if tp.id >= n_tracks:
                continue
            for b in tp.bars:
                if b < n_bars:
                    selected[tp.id][b] = True
            autoregressive[tp.id] = tp.autoregressive
            ignore[tp.id]         = tp.ignore

        mask.selected = selected
        mask.autoregressive = autoregressive
        mask.ignore = ignore
        return mask

    def _build_constraints(self, step) -> _core.ConstraintGraph:
        graph = _core.ConstraintGraph()
        grammar = _core.GrammarConstraint()

        # `Track` (TrackStart) and `TrackEnd` tokens are NEVER masked here. The
        # grammar FSM and `set_max_tracks(N)` (below) already decide when they
        # are syntactically legal:
        #   - AR, tracks_per_step=1:  model must sample TrackEnd to terminate.
        #   - AR, tracks_per_step>1:  model samples Track between consecutive
        #                             AR tracks and TrackEnd to close each.
        #   - Infill (any track count): model is in FillInStart→…→FillInEnd
        #                             states; Track/TrackEnd are never visited
        #                             during sampling — they live in the prompt
        #                             only.
        # Hard-masking these tokens for `len(request.tracks) <= 1` removed the
        # exit transition that AR needs and was the cause of the "zero legal
        # tokens" crash on single-track generation.

        # Exact bar count enforcement for autoregressive: each track must end
        # with exactly step.end_bar Bar tokens. (Infill is bounded by FillIn
        # block count instead, so leave the grammar unconstrained.)
        if step.is_autoregressive:
            # Window length — SessionState trims the prompt to this many bars
            # and bar_count_ resets per Track, so the AR grammar should expect
            # exactly window_len Bar tokens. Passing the absolute end_bar would
            # force the model to emit (end_bar - window_len) extra Bar tokens
            # of fake notes per generation, scaling cost with the absolute
            # playhead (O(N²) over a session).
            grammar.set_exact_bars(step.end_bar - step.start_bar)
            grammar.set_autoregressive_mode(True)
        # Ignored tracks are omitted from the token sequence by the encoder,
        # so the grammar's track_count_ only ever reaches (n_tracks - n_ignored).
        # Using len(score.tracks) here would force the model to emit extra
        # phantom tracks to satisfy max_tracks.
        ignored_ids = {tp.id for tp in self._request.tracks if tp.ignore}
        grammar.set_max_tracks(len(self._score.tracks) - len(ignored_ids))
        grammar.set_require_notes(True)

        graph.add_constraint(grammar)

        # Global hard cap on simultaneous note onsets — applies to every step
        # regardless of attribute controls. 0 = off.
        hard_limit = int(getattr(self._request.config, "polyphony_hard_limit", 0) or 0)
        if hard_limit > 0:
            graph.add_constraint(_core.PolyphonyConstraint(hard_limit))
        density_limit = int(getattr(self._request.config, "density_hard_limit", 0) or 0)
        if density_limit > 0:
            graph.add_constraint(_core.DensityConstraint(density_limit))

        attr_to_token = {
            "note_density":       _core.TokenType.NoteDensity,
            "onset_polyphony":    _core.TokenType.OnsetPolyphony,
            "pitch_range":        _core.TokenType.PitchRange,
            "key_signature":      _core.TokenType.KeySignature,
            "note_duration_dist": _core.TokenType.NoteDurationDist,
            "silence_proportion": _core.TokenType.SilenceProportion,
            "pitch_class_set":    _core.TokenType.PitchClassSet,
            "min_note_duration":  _core.TokenType.MinNoteDuration,
            "max_note_duration":  _core.TokenType.MaxNoteDuration,
            "min_polyphony":      _core.TokenType.MinPolyphony,
            "max_polyphony":      _core.TokenType.MaxPolyphony,
        }
        # First-class non-attribute controls. These don't flow through the
        # AttributeAnalyzer (no per-bar/per-track computation); they just pin a
        # token to a specific value when generated. Live on tp.controls.
        control_to_token = {
            "time_signature": _core.TokenType.TimeSig,
        }

        full_ar_ids = getattr(self, "_full_ar_ids", set())
        # Ordinal of each generating track in this step (position among the
        # step's track_indices in emit order). Used by per-bar constraints
        # which need to know "are we currently inside the target track?".
        track_ordinal = {tid: i for i, tid in enumerate(step.track_indices)}
        # Analyzer (for bar-level attr_name -> token_type lookup).
        analyzer = self._engine._analyzer
        # Bar attribute name -> TokenType (resolved via analyzer so we don't
        # hardcode the bar-level attribute schema here).
        def _bar_attr_token(name):
            if analyzer is None:
                return None
            obj = analyzer.get(name) if hasattr(analyzer, "get") else None
            if obj is None or getattr(obj, "level", "track") != "bar":
                return None
            tt_name = getattr(obj, "token_type", None)
            return getattr(_core.TokenType, tt_name, None) if tt_name else None
        # Bar control name -> TokenType.
        bar_control_to_token = {
            "time_signature": _core.TokenType.TimeSig,
        }
        for tp in self._request.tracks:
            # Attribute constraints only apply to full-AR tracks; infill and
            # partial-AR pin attributes through the encoded prompt.
            if tp.id not in full_ar_ids:
                continue
            if tp.id in step.track_indices:
                for attr_name, token_type in attr_to_token.items():
                    if attr_name in tp.attributes:
                        val = tp.attributes[attr_name]
                        graph.add_constraint(_core.AttributeValueConstraint(token_type, val))

                # Non-attribute controls: same constraint mechanism, separate
                # source dict to keep the attribute pipeline analyzer-pure.
                controls = getattr(tp, "controls", {}) or {}
                for ctrl_name, token_type in control_to_token.items():
                    if ctrl_name in controls:
                        graph.add_constraint(_core.AttributeValueConstraint(
                            token_type, int(controls[ctrl_name])))

                # Per-bar overrides — only meaningful for full-AR since
                # infill/partial-AR pin per-bar values via the prompt.
                # Bar-index is RELATIVE to step.start_bar (the grammar
                # resets bar_count_ at each Track token). For full-AR the
                # whole track is generated, so start_bar==0 and absolute
                # == relative.
                t_ord = track_ordinal.get(tp.id, 0)
                start_bar = int(step.start_bar)
                for bar_idx, bar_dict in (tp.bar_attributes or {}).items():
                    rel = int(bar_idx) - start_bar
                    if rel < 0:
                        continue
                    for attr_name, val in (bar_dict or {}).items():
                        tok = _bar_attr_token(attr_name)
                        if tok is None:
                            continue
                        graph.add_constraint(_core.BarAttributeValueConstraint(
                            tok, t_ord, rel, int(val)))
                for bar_idx, bar_dict in (tp.bar_controls or {}).items():
                    rel = int(bar_idx) - start_bar
                    if rel < 0:
                        continue
                    for ctrl_name, val in (bar_dict or {}).items():
                        tok = bar_control_to_token.get(ctrl_name)
                        if tok is None:
                            continue
                        graph.add_constraint(_core.BarAttributeValueConstraint(
                            tok, t_ord, rel, int(val)))

        return graph
