"""Run orig and ref generation on Etude with identical prompt; dump both token sequences.

Prompt setup is identical to test_full_generation_speed:
  file=6338816_Etude No. 4.mid, track=0, window bars 2-5, gen=[3, 4]
  barsPerStep=2, modelDim=4, temp=1.0
"""
import json, sys, copy, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "midigpt_refactor" / "src" / "python"))

import torch
import midigpt
import midigpt_refactor._core as _core
from midigpt_refactor._converters import from_cpp, to_cpp
from midigpt_refactor.attributes import AttributeAnalyzer
from midigpt_refactor.tokenizer.tokenizer import Tokenizer
from midigpt_refactor.inference.engine import InferenceEngine
from midigpt_refactor.inference.config import GenerationRequest, TrackPrompt, SamplingConfig

MIDI = ROOT / "midigpt_refactor" / "tests" / "comparison" / "midi" / "6338816_Etude No. 4.mid"
MODEL = ROOT / "models" / "yellow.pt"
CFG_TXT = (ROOT / "models" / "yellow_config.json").read_text()

WIN_START, WIN_END = 2, 5
WIN_SIZE = 4
N_GEN    = 2
GEN_BARS_GLOBAL = [3, 4]
GEN_BARS_LOCAL  = [b - WIN_START for b in GEN_BARS_GLOBAL]
TRACK_ID = 0
TEMP = 1.0
SEED = 42

print("=" * 80)
print("PROMPT SETUP")
print("=" * 80)
print(f"  file:        {MIDI.name}")
print(f"  window:      bars [{WIN_START}, {WIN_END}]  (size={WIN_SIZE})")
print(f"  gen bars:    global={GEN_BARS_GLOBAL}  local={GEN_BARS_LOCAL}")
print(f"  track_id:    {TRACK_ID}")
print(f"  barsPerStep: {N_GEN}")
print(f"  modelDim:    {WIN_SIZE}")
print(f"  temperature: {TEMP}")
print(f"  seed:        {SEED}")

cfg     = _core.EncoderConfig.from_json(CFG_TXT)
vocab   = _core.Vocabulary(cfg)
ref_enc = _core.Encoder(vocab)
ref_dec = _core.Decoder(vocab)
orig_enc = midigpt.ElVelocityDurationPolyphonyYellowEncoder()
analyzer = AttributeAnalyzer.yellow_default()

def pretty_ref(t):
    try:
        tt, val = vocab.decode(t)
        return f"{str(tt).split('.')[-1]}:{val}"
    except Exception:
        return f"?:{t}"

# ---- ORIG ----
print("\n" + "=" * 80)
print("ORIG GENERATION")
print("=" * 80)

orig_piece = json.loads(orig_enc.midi_to_json(str(MIDI)))
for t in orig_piece["tracks"]:
    t["bars"] = t["bars"][WIN_START:WIN_END + 1]
orig_piece_json = json.dumps(orig_piece)

status = json.loads(midigpt.status_from_piece(orig_piece_json))
for ti, t in enumerate(status["tracks"]):
    for bi in range(len(t.get("selectedBars", []))):
        t["selectedBars"][bi] = (ti == TRACK_ID and bi in GEN_BARS_LOCAL)

param = {
    "ckpt": str(MODEL),
    "barsPerStep": N_GEN,
    "modelDim": WIN_SIZE,
    "tracksPerStep": 1,
    "temperature": TEMP,
    "batchSize": 1,
    "percentage": 100,
    "polyphonyHardLimit": 10,
}

torch.manual_seed(SEED)
rec = midigpt.RecordTokenSequenceCallback()
cm  = midigpt.CallbackManager()
cm.add_callback(rec)

t0 = time.perf_counter()
result_str, attempts, tmg = midigpt.sample_multi_step_timed(
    orig_piece_json, json.dumps(status), json.dumps(param), 1, cm
)
orig_time_ms = (time.perf_counter() - t0) * 1000
orig_tokens = list(rec.tokens)

print(f"  wall:       {orig_time_ms:.1f}ms  attempts={attempts}")
print(f"  steps:      {json.loads(tmg).get('steps')}")
print(f"  tokens emitted (RecordTokenSequenceCallback): {len(orig_tokens)}")
print(f"  first 30:   {orig_tokens[:30]}")

# Count notes in generated bars of result
result_piece = json.loads(result_str)
gen_note_count = 0
for bi in GEN_BARS_LOCAL:
    gen_note_count += len(result_piece["tracks"][TRACK_ID]["bars"][bi].get("events", []))
print(f"  notes in gen bars (re-counted from result): {gen_note_count}")

# ---- REF ----
print("\n" + "=" * 80)
print("REF GENERATION")
print("=" * 80)

_model = torch.jit.load(str(MODEL), map_location="cpu")
_model.eval()
tokenizer = Tokenizer(cfg, analyzer)
engine = InferenceEngine(model=_model, tokenizer=tokenizer, analyzer=analyzer)
engine.warmup()

full_score_cpp = _core.MidiReader(cfg.resolution).read(str(MIDI))
score_prompt = from_cpp(full_score_cpp)
for t in score_prompt.tracks:
    t.bars = t.bars[WIN_START:WIN_END + 1]

request = GenerationRequest(
    tracks=[TrackPrompt(id=TRACK_ID, bars=GEN_BARS_LOCAL)],
    config=SamplingConfig(
        max_attempts=1,
        silence_check=False,
        novelty_check=False,
        bars_per_step=N_GEN,
        model_dim=WIN_SIZE,
        temperature=TEMP,
    ),
)

# Monkey-patch _sample_step to capture generated tokens
from midigpt_refactor.inference import session as _session_mod
original_sample_step = _session_mod.SamplingSession._sample_step
captured_tokens = []

def patched_sample_step(self, score, step, temperature):
    # Replicate the loop but record sampled tokens
    import torch as _torch
    analyzer_ = self._engine._analyzer
    if analyzer_ is not None:
        cpp_score = to_cpp(score)
        for t_idx, track in enumerate(score.tracks):
            new_attrs = dict(track.attributes)
            new_attrs.update(analyzer_.compute_track_tokens(cpp_score, t_idx))
            for b_idx in range(len(track.bars)):
                for k, v in analyzer_.compute_bar_tokens(cpp_score, t_idx, b_idx).items():
                    new_attrs[f"bar_{k}_{b_idx}"] = v
            track.attributes = new_attrs
    for tp in self._request.tracks:
        if tp.id < len(score.tracks):
            score.tracks[tp.id].attributes.update(tp.attributes)
    for t in score.tracks:
        t.attributes["num_bars"] = step.end_bar + 1

    state = _core.SessionState(
        to_cpp(score), step,
        self._engine._tokenizer._vocab,
        self._build_constraints(step),
        self._engine._tokenizer._encoder,
        self._engine._tokenizer._decoder
    )
    ctx_len = len(state.context_tokens())
    captured_tokens.append({'prompt_len': ctx_len, 'tokens': []})
    max_gen = 2048 - ctx_len - 1
    vocab_size = self._engine._tokenizer.vocab_size()
    mask_buf = _torch.empty(vocab_size, dtype=_torch.bool)
    initial_kv = self._engine._initial_kv
    with _torch.no_grad():
        past_kv = None
        while not state.complete() and self.gen_count < max_gen:
            if past_kv is None:
                ctx = _torch.tensor([state.context_tokens()], dtype=_torch.long)
            else:
                ctx = _torch.tensor([[state.context_tokens()[-1]]], dtype=_torch.long)
            try:
                if past_kv is None and initial_kv is not None:
                    outputs = self._engine._model(ctx, initial_kv)
                elif past_kv is None:
                    outputs = self._engine._model(ctx)
                else:
                    outputs = self._engine._model(ctx, past_kv)
            except Exception:
                outputs = self._engine._model(ctx)
                past_kv = None
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            logits = outputs[0][0, -1]
            past_kv = outputs[1] if len(outputs) > 1 else None
            mask_buf.copy_(_torch.as_tensor(state.logit_mask(), dtype=_torch.bool))
            masked = logits.masked_fill(~mask_buf, float("-inf"))
            probs = (masked / temperature).softmax(-1)
            if _torch.isnan(probs.sum()) or probs.sum() < 1e-6:
                probs = (logits / temperature).softmax(-1)
            tok = _torch.multinomial(probs, 1).item()
            captured_tokens[-1]['tokens'].append(tok)
            state.advance(tok)
            self.gen_count += 1
    return from_cpp(state.result())

_session_mod.SamplingSession._sample_step = patched_sample_step

torch.manual_seed(SEED)
session = engine.session(score_prompt, request)
t0 = time.perf_counter()
result = session.run()
ref_time_ms = (time.perf_counter() - t0) * 1000

ref_total_toks = sum(len(s['tokens']) for s in captured_tokens)
print(f"  wall:        {ref_time_ms:.1f}ms")
print(f"  steps:       {len(captured_tokens)}")
for i, s in enumerate(captured_tokens):
    print(f"    step {i}: prompt_len={s['prompt_len']}  generated={len(s['tokens'])}")
print(f"  total tokens generated: {ref_total_toks}")
ref_note_count = 0
for bi in GEN_BARS_LOCAL:
    ref_note_count += len(result.tracks[TRACK_ID].bars[bi].notes)
print(f"  notes in gen bars: {ref_note_count}")

# ---- DUMP ----
print("\n" + "=" * 80)
print("ORIG TOKEN STREAM (decoded with ref vocab — they share the encoder)")
print("=" * 80)
for i, t in enumerate(orig_tokens):
    print(f"  [{i:4d}] {t:4d}  {pretty_ref(t)}")

print("\n" + "=" * 80)
print("REF TOKEN STREAM (concatenated across steps)")
print("=" * 80)
running = 0
for si, s in enumerate(captured_tokens):
    print(f"  --- step {si} (prompt_len={s['prompt_len']}) ---")
    for j, t in enumerate(s['tokens']):
        print(f"  [{running:4d}] {t:4d}  {pretty_ref(t)}")
        running += 1

# ---- diff ----
print("\n" + "=" * 80)
print("FIRST DIVERGENCE")
print("=" * 80)
ref_flat = []
for s in captured_tokens:
    ref_flat.extend(s['tokens'])
minl = min(len(orig_tokens), len(ref_flat))
first_diff = None
for i in range(minl):
    if orig_tokens[i] != ref_flat[i]:
        first_diff = i
        break
if first_diff is None:
    print(f"  Streams identical for first {minl} tokens.")
else:
    print(f"  First divergence at token {first_diff}:")
    for j in range(max(0, first_diff - 3), min(minl, first_diff + 5)):
        marker = "  * " if j == first_diff else "    "
        print(f"{marker}[{j:4d}] orig={orig_tokens[j]:4d}({pretty_ref(orig_tokens[j])})  ref={ref_flat[j]:4d}({pretty_ref(ref_flat[j])})")
print(f"\n  orig total: {len(orig_tokens)}   ref total: {len(ref_flat)}")
print(f"  orig notes: {gen_note_count}      ref notes: {ref_note_count}")
