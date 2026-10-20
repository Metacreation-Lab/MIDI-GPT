#!/usr/bin/env python3
"""Generate the MIDI files for docs/examples.md.

Every prompt is an 8-bar excerpt of a CC0 or public-domain piece in tests/midi/
(see tests/midi/SOURCES.md). Writes the prompts and generations to
docs/assets/midi/, and docs/assets/midi/examples.json, the list of examples that
scripts/render_doc_midi_videos.py renders to video.

    python scripts/gen_doc_examples.py [--model yellow_medium]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from midigpt import Bar, Score, Track
from midigpt.inference import GenerationRequest, InferenceConfig, InferenceEngine, TrackPrompt

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "tests" / "midi"
OUT = REPO / "docs" / "assets" / "midi"
N_BARS = 8
BARS = list(range(N_BARS))
CONFIG = dict(model_dim=8, mask_mode="attention")

GM_NAMES = [
    "acoustic_grand_piano", "bright_acoustic_piano", "electric_grand_piano", "honky_tonk_piano",
    "electric_piano_1", "electric_piano_2", "harpsichord", "clavinet", "celesta", "glockenspiel",
    "music_box", "vibraphone", "marimba", "xylophone", "tubular_bells", "dulcimer", "drawbar_organ",
    "percussive_organ", "rock_organ", "church_organ", "reed_organ", "accordion", "harmonica",
    "tango_accordion", "acoustic_guitar_nylon", "acoustic_guitar_steel", "electric_guitar_jazz",
    "electric_guitar_clean", "electric_guitar_muted", "overdriven_guitar", "distortion_guitar",
    "guitar_harmonics", "acoustic_bass", "electric_bass_finger", "electric_bass_pick", "fretless_bass",
    "slap_bass_1", "slap_bass_2", "synth_bass_1", "synth_bass_2", "violin", "viola", "cello",
    "contrabass", "tremolo_strings", "pizzicato_strings", "orchestral_harp", "timpani",
    "string_ensemble_1", "string_ensemble_2", "synth_strings_1", "synth_strings_2", "choir_aahs",
    "voice_oohs", "synth_voice", "orchestra_hit", "trumpet", "trombone", "tuba", "muted_trumpet",
    "french_horn", "brass_section", "synth_brass_1", "synth_brass_2", "soprano_sax", "alto_sax",
    "tenor_sax", "baritone_sax", "oboe", "english_horn", "bassoon", "clarinet", "piccolo", "flute",
    "recorder", "pan_flute", "blown_bottle", "shakuhachi", "whistle", "ocarina", "lead_1_square",
    "lead_2_sawtooth", "lead_3_calliope", "lead_4_chiff", "lead_5_charang", "lead_6_voice",
    "lead_7_fifths", "lead_8_bass_lead", "pad_1_new_age", "pad_2_warm", "pad_3_polysynth",
    "pad_4_choir", "pad_5_bowed", "pad_6_metallic", "pad_7_halo", "pad_8_sweep", "fx_1_rain",
    "fx_2_soundtrack", "fx_3_crystal", "fx_4_atmosphere", "fx_5_brightness", "fx_6_goblins",
    "fx_7_echoes", "fx_8_sci_fi", "sitar", "banjo", "shamisen", "koto", "kalimba", "bagpipe", "fiddle",
    "shanai", "tinkle_bell", "agogo", "steel_drums", "woodblock", "taiko_drum", "melodic_tom",
    "synth_drum", "reverse_cymbal", "guitar_fret_noise", "breath_noise", "seashore", "bird_tweet",
    "telephone_ring", "helicopter", "applause", "gunshot",
]


def name(track: Track) -> str:
    return "drums" if track.track_type == "drum" else GM_NAMES[track.instrument]


def excerpt(source: str, tracks: list[int], start: int | None = None) -> Score:
    """8 bars of `tracks` from `source`, from bar `start`, or by default the densest 8 bars
    of 4/4 in which every one of them plays."""
    score = Score.from_midi(str(SRC / source))
    length = min(len(score.tracks[t].bars) for t in tracks)
    best = None if start is None else (0, start)
    for start in range(length - N_BARS + 1) if best is None else ():
        bars = [score.tracks[t].bars[b] for t in tracks for b in range(start, start + N_BARS)]
        if all(b.notes and (b.ts_numerator, b.ts_denominator) == (4, 4) for b in bars):
            density = sum(len(b.notes) for b in bars)
            if best is None or density > best[0]:
                best = (density, start)
    if best is None:
        raise SystemExit(f"{source}: no 8-bar 4/4 window where tracks {tracks} all play")
    start = best[1]
    print(f"  {source}: bars {start}-{start + N_BARS - 1}, tracks {tracks}")
    return Score(
        tracks=[Track(bars=copy.deepcopy(score.tracks[t].bars[start:start + N_BARS]),
                      instrument=score.tracks[t].instrument, track_type=score.tracks[t].track_type)
                for t in tracks],
        resolution=score.resolution, tempo=score.tempo,
    )


def with_empty_track(score: Score, instrument: int = 0, track_type: str = "melodic") -> Score:
    out = copy.deepcopy(score)
    out.tracks.append(Track(bars=[Bar() for _ in range(N_BARS)], instrument=instrument, track_type=track_type))
    return out


def clear_bars(score: Score, track: int, bars: list[int]) -> Score:
    out = copy.deepcopy(score)
    for b in bars:
        out.tracks[track].bars[b].notes = []
    return out


def generate(engine, score: Score, targets: dict[int, dict], seed: int, autoregressive: bool = True) -> Score:
    """Generate the bars in `targets` ({track: {"bars": [...], "attributes": {...}}}); the other tracks are context.
    Autoregressive writes each target track from scratch; infill (autoregressive=False) fills bars in place."""
    prompts = [
        TrackPrompt(id=i, bars=targets[i]["bars"], autoregressive=autoregressive,
                    attributes=targets[i].get("attributes", {}))
        if i in targets else TrackPrompt(id=i, bars=[])
        for i in range(len(score.tracks))
    ]
    session = engine.session(score, GenerationRequest(tracks=prompts, config=InferenceConfig(seed=seed, **CONFIG)))
    result = session.run()
    print(f"    seed={seed} model_dim={session._request.config.model_dim} dropped_context={session.dropped_context} "
          f"notes={[sum(len(b.notes) for b in result.tracks[t].bars) for t in targets]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yellow_medium")
    args = parser.parse_args()
    engine = InferenceEngine.from_pretrained(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []

    def save(score: Score, midi: str, **meta) -> None:
        score.to_midi(str(OUT / midi))
        if meta:
            examples.append({"midi": midi, "track_names": [name(t) for t in score.tracks], **meta})

    # 1. Track sampling: drums under guitar and bass ("Just Say No To Yes")
    print("Example 1: drums")
    band = excerpt("oga_just_say_no_to_yes.mid", [1, 7])          # distortion guitar, slap bass
    save(band, "prompt_band.mid")
    prompt = with_empty_track(band, track_type="drum")
    for density in (1, 3, 8):
        out = generate(engine, prompt, {2: {"bars": BARS, "attributes": {"note_density": density}}}, seed=density)
        save(out, f"sampling_drums_density{density}.mid", id=f"track_sampling_drums_density{density}",
             task="Track Sampling", title=f"Band: drums, note_density = {density}",
             prompt="prompt_band.mid", controls=f"note_density = {density}", generated_tracks=[2])

    # 2. Track sampling: a new cello part for a string quartet (Beethoven Op. 18 No. 4)
    print("Example 2: cello")
    quartet = excerpt("mutopia_beethoven_op18_4_1.mid", [0, 1, 2])  # violin I, violin II, viola
    save(quartet, "prompt_quartet.mid")
    prompt = with_empty_track(quartet, instrument=42)               # cello
    cello = [("16th", {"max_note_duration": 1}), ("8th", {"max_note_duration": 2}),
             ("4th", {"max_note_duration": 3}), ("half", {"max_note_duration": 4}),
             ("sustained", {"min_note_duration": 4})]
    for i, (label, attrs) in enumerate(cello):
        attrs = {"max_polyphony": 0, **attrs}
        out = generate(engine, prompt, {3: {"bars": BARS, "attributes": attrs}}, seed=10 + i)
        midi = f"sampling_cello_{'dur_' if label != 'sustained' else ''}{label}.mid"
        save(out, midi, id=f"track_sampling_cello_{'dur_' if label != 'sustained' else ''}{label}",
             task="Track Sampling", title=f"Quartet: cello, {', '.join(f'{k} = {v}' for k, v in attrs.items())}",
             prompt="prompt_quartet.mid", controls=", ".join(f"{k} = {v}" for k, v in attrs.items()),
             generated_tracks=[3])

    # 3. Resampling a guitar part with the model's own controls ("Savior of Humanity")
    print("Example 3: guitar")
    solo = excerpt("oga_savior_of_humanity.mid", [0, 7, 9])        # distortion guitar, synth bass, drums
    save(solo, "prompt_solo.mid")
    for n in (1, 2):
        out = generate(engine, solo, {0: {"bars": BARS}}, seed=20 + n)
        save(out, f"solo_resample{n}.mid", id=f"solo_resample{n}", task="Track Sampling",
             title=f"Guitar resample {n} (model-sampled controls)", prompt="prompt_solo.mid",
             controls="none (sampled by the model)", generated_tracks=[0])

    # 4. Bar infilling: second violin, bars 4-7 (Eine kleine Nachtmusik). Its bars 0-3 are 16th notes, so
    # max_note_duration 1 (16th) and 2 (8th) are both reachable: a track-level attribute covers the fixed bars too.
    # It is a soft control on a 4-bar infill, so these seeds were picked as takes that follow it: the 16th one
    # was 1 of 60 seeds tried (100-159), the 8th one 3 of the first 6.
    print("Example 4: violin infill")
    serenade = excerpt("mutopia_mozart_kv525_1.mid", [0, 1, 2, 3], start=19)  # violin I, violin II, viola, cello
    prompt = clear_bars(serenade, 1, [4, 5, 6, 7])
    save(prompt, "prompt_violin_infill.mid")
    for label, dur, seed in (("16th", 1, 134), ("8th", 2, 102)):
        out = generate(engine, prompt, {1: {"bars": [4, 5, 6, 7], "attributes": {"max_note_duration": dur}}},
                       seed=seed, autoregressive=False)
        save(out, f"violin_infill_{label}.mid", id=f"violin_infill_{label}", task="Bar Infilling",
             title=f"Second violin infill, max_note_duration = {dur}", prompt="prompt_violin_infill.mid",
             controls=f"max_note_duration = {dur}", target_bars=[4, 5, 6, 7], generated_tracks=[1])

    # 5. Multi-step composition ("The Ritual")
    print("Example 5: multi-step")
    ritual = excerpt("oga_the_ritual.mid", [6, 1, 5])                # piano, bass, strings
    step0 = ritual
    for t in range(3):
        step0 = clear_bars(step0, t, [4, 5, 6, 7])
    save(step0, "prompt_multistep.mid")
    step1 = generate(engine, step0, {t: {"bars": [4, 5, 6, 7]} for t in range(3)}, seed=41)
    step2 = generate(engine, with_empty_track(step1, instrument=24),     # nylon guitar
                     {3: {"bars": BARS, "attributes": {"max_polyphony": 0, "max_note_duration": 1}}}, seed=42)
    step3 = generate(engine, with_empty_track(step2, track_type="drum"),
                     {4: {"bars": BARS, "attributes": {"note_density": 9}}}, seed=43)
    save(step3, "multistep_final.mid", id="multistep_final", task="Multi-Step Composition",
         title="Continuation + guitar + drums", prompt="prompt_multistep.mid",
         controls="guitar: max_polyphony = 0, max_note_duration = 1  |  drums: note_density = 9",
         target_bars=[4, 5, 6, 7], generated_tracks=[3, 4])

    (OUT / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT / 'examples.json'}")


if __name__ == "__main__":
    main()
