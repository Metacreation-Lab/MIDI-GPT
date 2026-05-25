import copy
import logging
from dataclasses import replace as replace_cfg
from tqdm import tqdm
import midigpt._core as _core
from midigpt._types import Score
from midigpt._converters import to_cpp, from_cpp
from midigpt.inference.config import GenerationRequest, SamplingConfig
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
            score = copy.deepcopy(self._score)
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
                  cfg: SamplingConfig | None = None) -> Score:
        if cfg is None:
            cfg = self._request.config
        original_score = score # Keep original score for _is_acceptable comparison

        for i in range(cfg.max_attempts):
            temperature = cfg.temperature * (cfg.temperature_escalation ** i)
            candidate = self._sample_step(score, step, temperature)
            
            if self._is_acceptable(original_score, candidate, cfg, step):
                return candidate
            
            logging.debug(f"Attempt {i+1}/{cfg.max_attempts} failed, retrying with temp={temperature*cfg.temperature_escalation:.2f}")

        # If max attempts reached and no acceptable candidate found, raise an error
        raise RuntimeError(f"Max attempts ({cfg.max_attempts}) reached, no acceptable candidate found. Last candidate had {self._note_count(candidate, step)} notes in generated bars.")

    def _sample_step(self, score: Score, step, temperature: float) -> Score:
        try:
            import torch
        except ImportError:
            raise ImportError("pip install midigpt[inference]")

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
        state = _core.SessionState(
            to_cpp(score), step,
            self._engine._tokenizer._vocab,
            self._build_constraints(step),
            self._engine._tokenizer._encoder,
            self._engine._tokenizer._decoder,
            getattr(self._request.config, "use_span_masks", False),
        )

        context_len = len(state.context_tokens())
        # Encoder-driven span mask: bool tensor of shape (T,) where True = key
        # position is visible to attention. Built once per step; SDPA combines
        # this with the causal mask. Only non-None when use_span_masks is set
        # AND the encoder produced spans (no spans => fast unmasked path).
        spans = state.hidden_spans() if getattr(self._request.config, "use_span_masks", False) else []
        if spans:
            key_visible = torch.ones(context_len, dtype=torch.bool)
            for s, e in spans:
                key_visible[s:e] = False
        else:
            key_visible = None
        if self.enable_profiling:
            self.encode_time += time.perf_counter() - t_enc
        model_max_ctx = self._model_max_context()
        if context_len >= model_max_ctx:
            raise _ContextOverflow(context_len, model_max_ctx)
        max_gen_tokens = model_max_ctx - context_len - 1

        # Pre-allocate mask buffer once for this step (reused every token)
        vocab_size = self._engine._tokenizer.vocab_size()
        mask_buf = torch.empty(vocab_size, dtype=torch.bool)

        initial_kv = self._engine._initial_kv  # cached — no model call

        with torch.no_grad():
            past_kv = None
            while not state.complete() and self.gen_count < max_gen_tokens:
                if past_kv is None:
                    ctx = torch.tensor([state.context_tokens()], dtype=torch.long)
                else:
                    ctx = torch.tensor([[state.context_tokens()[-1]]], dtype=torch.long)

                t_fwd = time.perf_counter()
                # Pass key_mask only on the production path. Stubs / TorchScript
                # callables may not accept kwargs, so fall back without it.
                model_call = self._engine._model
                try:
                    if past_kv is None and initial_kv is not None:
                        if key_visible is not None:
                            outputs = model_call(ctx, initial_kv, key_mask=key_visible)
                        else:
                            outputs = model_call(ctx, initial_kv)
                    elif past_kv is None:
                        if key_visible is not None:
                            outputs = model_call(ctx, key_mask=key_visible)
                        else:
                            outputs = model_call(ctx)
                    else:
                        if key_visible is not None:
                            outputs = model_call(ctx, past_kv, key_mask=key_visible)
                        else:
                            outputs = model_call(ctx, past_kv)
                except Exception:
                    outputs = model_call(ctx)
                    past_kv = None
                if self.enable_profiling:
                    self.model_forward_time += time.perf_counter() - t_fwd

                if not isinstance(outputs, tuple):
                    outputs = (outputs,)

                logits  = outputs[0][0, -1]
                past_kv = outputs[1] if len(outputs) > 1 else None

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
                    if key_visible is not None:
                        key_visible = torch.cat(
                            [key_visible, torch.ones(1, dtype=torch.bool)]
                        )
                    continue
                state.advance(token)
                self.gen_count += 1
                if key_visible is not None:
                    # Generated tokens are always visible to subsequent queries.
                    key_visible = torch.cat(
                        [key_visible, torch.ones(1, dtype=torch.bool)]
                    )

        t_dec = time.perf_counter()
        result = from_cpp(state.result())
        if self.enable_profiling:
            self.decode_time += time.perf_counter() - t_dec
        return result

    def _snapshot_prompt_state(self, step) -> dict:
        """Per-bar prompt state for the current step, post-mask_bars patch.

        Each entry is "C" (CONTEXT), "M" (MASKED via MaskBar token),
        "A" (MASKED via attention span-mask), or "T" (TO_GENERATE).
        "A" only appears when SamplingConfig.use_span_masks is True — purely
        a visualization label; the bar classification is identical to "M".
        """
        ar_ids = {tp.id for tp in self._request.tracks if tp.autoregressive}
        btg    = set(step.bars_to_generate)
        masked_label = "A" if getattr(self._request.config, "use_span_masks", False) else "M"
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

    def _is_acceptable(self, original: Score, candidate: Score, cfg: SamplingConfig, step: GenerationStep) -> bool:
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
        """Read the model's positional context length.

        Prefers model.max_context() (ModelBase protocol). Falls back to
        attribute probing for TorchScript / legacy callables, then 2048.
        """
        m = self._engine._model
        if hasattr(m, "max_context"):
            return m.max_context()
        for path in (
            ("config", "n_positions"),
            ("config", "n_ctx"),
            ("config", "max_position_embeddings"),
        ):
            obj = m
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                v = int(obj)
                if v > 0:
                    return v
            except Exception:
                continue
        try:
            wpe = m.transformer.wpe.weight
            v = int(wpe.shape[0])
            if v > 0:
                return v
        except Exception:
            pass
        return 2048

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
        if len(self._request.tracks) <= 1:
            grammar.set_mask_track_start(True)
            grammar.set_mask_track_end(True)
            
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
        grammar.set_max_tracks(len(self._score.tracks))
        grammar.set_require_notes(True)
        
        graph.add_constraint(grammar)

        attr_to_token = {
            "pitch_range":        _core.TokenType.PitchRange,
            "key_signature":      _core.TokenType.KeySignature,
            "note_duration_dist": _core.TokenType.NoteDurationDist,
            "tension":            _core.TokenType.Tension,
            "silence_proportion": _core.TokenType.SilenceProportion,
            "pitch_class_set":    _core.TokenType.PitchClassSet,
            "min_note_duration":  _core.TokenType.MinNoteDuration,
            "max_note_duration":  _core.TokenType.MaxNoteDuration,
            "min_polyphony":      _core.TokenType.MinPolyphony,
            "max_polyphony":      _core.TokenType.MaxPolyphony,
        }

        full_ar_ids = getattr(self, "_full_ar_ids", set())
        for tp in self._request.tracks:
            # Attribute constraints only apply to full-AR tracks; infill and
            # partial-AR pin attributes through the encoded prompt.
            if tp.id not in full_ar_ids:
                continue
            if tp.id in step.track_indices:
                if "note_density" in tp.attributes:
                    graph.add_constraint(_core.DensityConstraint(tp.attributes["note_density"]))
                if "onset_polyphony" in tp.attributes:
                    graph.add_constraint(_core.PolyphonyConstraint(tp.attributes["onset_polyphony"]))

                for attr_name, token_type in attr_to_token.items():
                    if attr_name in tp.attributes:
                        val = tp.attributes[attr_name]
                        graph.add_constraint(_core.AttributeValueConstraint(token_type, val))

        return graph
