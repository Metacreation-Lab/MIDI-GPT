import os
import copy
import torch
from midigpt import Score, Track, Bar, Note
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt

# Create output dir if not exists
os.makedirs("docs/assets/midi", exist_ok=True)

# Set default seed
SEED = 42

print("Loading InferenceEngine...")
engine = InferenceEngine.from_pretrained("yellow")

def slice_score(score: Score, start_bar: int, num_bars: int) -> Score:
    """Slice score to include start_bar to start_bar + num_bars."""
    new_tracks = []
    for track in score.tracks:
        sliced_bars = [copy.deepcopy(b) for b in track.bars[start_bar : start_bar + num_bars]]
        new_track = Track(
            bars=sliced_bars,
            instrument=track.instrument,
            track_type=track.track_type,
            attributes=dict(track.attributes)
        )
        new_tracks.append(new_track)
    return Score(new_tracks, resolution=score.resolution, tempo=score.tempo)

# --------------------------------------------------------------------------- #
# Scenario 1: Track Sampling (Drums) — Hotel California (Rock/Pop)
# --------------------------------------------------------------------------- #
print("\n--- Scenario 1: Drum Track Sampling (Hotel California) ---")
hotel_score = Score.from_midi("tmp/midi_examples/HotelCalifornia.mid")
hotel_score = engine._tokenizer.normalize_input(hotel_score)
hotel_slice = slice_score(hotel_score, 16, 8)

# Context tracks: Guitar (0) and Bass (2)
guitar_context = copy.deepcopy(hotel_slice.tracks[0])
bass_context = copy.deepcopy(hotel_slice.tracks[2])

# Target track: Drums (5)
drums_target = copy.deepcopy(hotel_slice.tracks[5])
drums_target.track_type = "drum"
drums_target.instrument = 0
for bar in drums_target.bars:
    bar.notes = []

score_for_drums = Score(
    tracks=[guitar_context, bass_context, drums_target],
    resolution=hotel_slice.resolution,
    tempo=hotel_slice.tempo
)

# Save prompt
prompt_drums_score = copy.deepcopy(score_for_drums)
prompt_drums_score.to_midi("docs/assets/midi/prompt_sampling.mid")
print("Saved prompt_sampling.mid")

# 1. Sparse Drums
print("Generating sparse drums...")
req_sparse_drums = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=list(range(8)), ignore=False, attributes={"note_density": 1})
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=SEED, max_attempts=5, tracks_per_step=2)
)
res_sparse_drums = engine.session(score_for_drums, req_sparse_drums).run()
res_sparse_drums.to_midi("docs/assets/midi/sampling_drums_sparse.mid")
print("Saved sampling_drums_sparse.mid")

# 2. Dense Drums
print("Generating dense drums...")
req_dense_drums = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=list(range(8)), ignore=False, attributes={"note_density": 8})
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=SEED, max_attempts=5, tracks_per_step=2)
)
res_dense_drums = engine.session(score_for_drums, req_dense_drums).run()
res_dense_drums.to_midi("docs/assets/midi/sampling_drums_dense.mid")
print("Saved sampling_drums_dense.mid")


# --------------------------------------------------------------------------- #
# Scenario 2: Track Sampling (Bassline) — Bach Concerto (Classical)
# --------------------------------------------------------------------------- #
print("\n--- Scenario 2: Cello Bassline Track Sampling (Bach) ---")
bach_score = Score.from_midi("tmp/midi_examples/bwv1052a.mid")
bach_score = engine._tokenizer.normalize_input(bach_score)
bach_slice = slice_score(bach_score, 12, 8)

# Context tracks: Violin 1 (0), Violin 2 (1), Viola (2)
violin1_context = copy.deepcopy(bach_slice.tracks[0])
violin2_context = copy.deepcopy(bach_slice.tracks[1])
viola_context = copy.deepcopy(bach_slice.tracks[2])

# Target track: Cello (3)
cello_target = copy.deepcopy(bach_slice.tracks[3])
cello_target.track_type = "melodic"
cello_target.instrument = 43
for bar in cello_target.bars:
    bar.notes = []

score_for_cello = Score(
    tracks=[violin1_context, violin2_context, viola_context, cello_target],
    resolution=bach_slice.resolution,
    tempo=bach_slice.tempo
)

# Save prompt
prompt_cello_score = copy.deepcopy(score_for_cello)
prompt_cello_score.to_midi("docs/assets/midi/prompt_bach.mid")
print("Saved prompt_bach.mid")

# 1. Staccato Cello Bassline (Seed 43)
print("Generating staccato cello bassline...")
req_staccato_cello = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False),
        TrackPrompt(id=3, bars=list(range(8)), ignore=False, attributes={"max_polyphony": 0, "max_note_duration": 1})
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=43, max_attempts=5, tracks_per_step=2)
)
res_staccato_cello = engine.session(score_for_cello, req_staccato_cello).run()
res_staccato_cello.to_midi("docs/assets/midi/sampling_cello_staccato.mid")
print("Saved sampling_cello_staccato.mid")

# 2. Legato/Sustained Cello Bassline (Seed 42 + Autoregressive mode)
print("Generating sustained cello bassline...")
req_sustained_cello = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False),
        # Use autoregressive=True so min_note_duration=4 warning is bypassed and constraint is applied
        TrackPrompt(id=3, bars=list(range(8)), autoregressive=True, ignore=False, attributes={"max_polyphony": 0, "min_note_duration": 4})
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=42, max_attempts=5, tracks_per_step=2)
)
res_sustained_cello = engine.session(score_for_cello, req_sustained_cello).run()
res_sustained_cello.to_midi("docs/assets/midi/sampling_cello_sustained.mid")
print("Saved sampling_cello_sustained.mid")


# --------------------------------------------------------------------------- #
# Scenario 3: Bar Infilling — Everything In Its Right Place (Electronic)
# --------------------------------------------------------------------------- #
print("\n--- Scenario 3: Keyboard Bar Infilling (Radiohead) ---")
radiohead_score = Score.from_midi("tmp/midi_examples/EverythingIsInItsRightPlace.mid")
radiohead_score = engine._tokenizer.normalize_input(radiohead_score)
radiohead_slice = slice_score(radiohead_score, 16, 8) # bars 16-24

# Keyboard is track 0. Bass is track 4. Drums is track 5.
keyboard_track = copy.deepcopy(radiohead_slice.tracks[0])
bass_context = copy.deepcopy(radiohead_slice.tracks[4])
drums_context = copy.deepcopy(radiohead_slice.tracks[5])

score_for_infill = Score(
    tracks=[keyboard_track, bass_context, drums_context],
    resolution=radiohead_slice.resolution,
    tempo=radiohead_slice.tempo
)

# Clear middle bars 3 and 4 of the 8-bar slice to create the infill prompt
infill_prompt = copy.deepcopy(score_for_infill)
infill_prompt.tracks[0].bars[3].notes = []
infill_prompt.tracks[0].bars[4].notes = []
infill_prompt.to_midi("docs/assets/midi/prompt_infill.mid")
print("Saved prompt_infill.mid")

# 1. Sparse Infill (Seed 44)
print("Generating sparse infill...")
req_sparse_infill = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[3, 4], autoregressive=False, attributes={"min_polyphony": 0, "max_polyphony": 2}),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False)
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=44, max_attempts=5)
)
res_sparse_infill = engine.session(score_for_infill, req_sparse_infill).run()
res_sparse_infill.to_midi("docs/assets/midi/infill_sparse.mid")
print("Saved infill_sparse.mid")

# 2. Chordal Infill (Seed 42)
print("Generating chordal infill...")
req_chordal_infill = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[3, 4], autoregressive=False, attributes={"min_polyphony": 1, "max_polyphony": 8}),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False)
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=42, max_attempts=5)
)
res_chordal_infill = engine.session(score_for_infill, req_chordal_infill).run()
res_chordal_infill.to_midi("docs/assets/midi/infill_chordal.mid")
print("Saved infill_chordal.mid")


# --------------------------------------------------------------------------- #
# Scenario 4: Autoregressive Continuation — Summer (Film Soundtrack)
# --------------------------------------------------------------------------- #
print("\n--- Scenario 4: Piano Continuation (Summer) ---")
summer_score = Score.from_midi("tmp/midi_examples/Summer.mid")
summer_score = engine._tokenizer.normalize_input(summer_score)
summer_slice = slice_score(summer_score, 13, 8)

# Piano is track 0. Bass is track 1. Strings Synth is track 5.
piano_track = copy.deepcopy(summer_slice.tracks[0])
bass_context = copy.deepcopy(summer_slice.tracks[1])
strings_context = copy.deepcopy(summer_slice.tracks[5])

score_for_ar = Score(
    tracks=[piano_track, bass_context, strings_context],
    resolution=summer_slice.resolution,
    tempo=summer_slice.tempo
)

# 4-bar seed prompt
ar_prompt_score = copy.deepcopy(score_for_ar)
for bar_idx in range(4, 8):
    ar_prompt_score.tracks[0].bars[bar_idx].notes = []
ar_prompt_score.to_midi("docs/assets/midi/prompt_ar.mid")
print("Saved prompt_ar.mid")

# 1. Melodic Lead Continuation (Seed 42)
print("Generating melodic piano continuation...")
req_monophonic_ar = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[4, 5, 6, 7], autoregressive=True, attributes={"min_polyphony": 0, "max_polyphony": 4}),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False)
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=42, max_attempts=5)
)
res_monophonic_ar = engine.session(score_for_ar, req_monophonic_ar).run()
res_monophonic_ar.to_midi("docs/assets/midi/ar_monophonic.mid")
print("Saved ar_monophonic.mid")

# 2. Chordal Continuation (Seed 44)
print("Generating chordal piano continuation...")
req_chordal_ar = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[4, 5, 6, 7], autoregressive=True, attributes={"min_polyphony": 2, "max_polyphony": 8}),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False)
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=44, max_attempts=5)
)
res_chordal_ar = engine.session(score_for_ar, req_chordal_ar).run()
res_chordal_ar.to_midi("docs/assets/midi/ar_chordal.mid")
print("Saved ar_chordal.mid")


# --------------------------------------------------------------------------- #
# Scenario 5: Autoregressive Continuation — SSBM All Star (Video Game)
# --------------------------------------------------------------------------- #
print("\n--- Scenario 5: Harp Continuation (SSBM All Star Intro) ---")
ssbm_score = Score.from_midi("tmp/midi_examples/SSBM_All_Star_Intro.mid")
ssbm_score = engine._tokenizer.normalize_input(ssbm_score)
ssbm_slice = slice_score(ssbm_score, 0, 8)

# Bells is track 1. Strings is track 2. Harp is track 4.
bells_context = copy.deepcopy(ssbm_slice.tracks[1])
strings_context = copy.deepcopy(ssbm_slice.tracks[2])
harp_track = copy.deepcopy(ssbm_slice.tracks[4])
harp_track.track_type = "melodic"
harp_track.instrument = 46

score_for_harp = Score(
    tracks=[bells_context, strings_context, harp_track],
    resolution=ssbm_slice.resolution,
    tempo=ssbm_slice.tempo
)

# 4-bar seed prompt
harp_prompt_score = copy.deepcopy(score_for_harp)
for bar_idx in range(4, 8):
    harp_prompt_score.tracks[2].bars[bar_idx].notes = []
harp_prompt_score.to_midi("docs/assets/midi/prompt_harp.mid")
print("Saved prompt_harp.mid")

# 1. Rapid Arpeggios (Max Note Duration = 1 (16th notes), Seed 42)
print("Generating rapid arpeggios harp continuation...")
req_sparse_harp = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[4, 5, 6, 7], autoregressive=True, attributes={"max_note_duration": 1})
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=42, max_attempts=5)
)
res_sparse_harp = engine.session(score_for_harp, req_sparse_harp).run()
res_sparse_harp.to_midi("docs/assets/midi/harp_monophonic.mid")
print("Saved harp_monophonic.mid")

# 2. Sustained Chords (Max Polyphony = 4, Seed 42)
print("Generating sustained chords harp continuation...")
req_dense_harp = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[4, 5, 6, 7], autoregressive=True, attributes={"max_polyphony": 4})
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=42, max_attempts=5)
)
res_dense_harp = engine.session(score_for_harp, req_dense_harp).run()
res_dense_harp.to_midi("docs/assets/midi/harp_chordal.mid")
print("Saved harp_chordal.mid")

print("\nAll 5 multi-track doc examples generated successfully!")
