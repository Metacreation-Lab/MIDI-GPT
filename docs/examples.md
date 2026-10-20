# Examples & Controllable Generation

This page shows **MIDI-GPT**'s core capabilities — **Track Sampling**, **Bar Infilling (Inpainting)** and **Autoregressive Continuation** — on short excerpts of freely licensed music: original pieces by Ivan Stanton released under CC0 on [OpenGameArt](https://opengameart.org/users/northivanastan), and public-domain scores by Beethoven and Mozart from the [Mutopia Project](https://www.mutopiaproject.org). All examples are generated with `yellow_medium`.

By setting **Attribute Controls** (such as polyphony, note density, and note duration), you can guide the model to generate musical parts that fit your composition.

---

## Video Renders & MIDI Downloads
Each example is rendered as an MP4 video with synthesized audio and a per-track piano-roll view. Context tracks are shown in **blue/green/purple**, generated tracks in **gold**. Download links are provided for importing the `.mid` files into a DAW.

The examples are reproducible: `python scripts/gen_doc_examples.py` regenerates every MIDI file on this page (fixed seeds), and `python scripts/render_doc_midi_videos.py` renders the videos.

---

## 1. Controllable Track Sampling (Arrangement)
Track sampling generates a completely new track conditioned on the existing ones. The new track is generated autoregressively (`autoregressive=True`), every other track is context (`bars=[]`). You can pin attribute controls to steer the new track, or leave them out and let the model **sample its own controls** from the prior.

### Example 1: Drums for a Rock Groove
We take a distortion guitar and slap bass part from Ivan Stanton's "Just Say No To Yes" (CC0) as context and generate a drum track. On `yellow`, `note_density` applies to drum tracks; adjusting it (scale `0–9`) steers the drums from a sparse beat to a dense, active pattern.

<div class="midi-example-card">
  <strong>Guitar &amp; Bass Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_band.mid" class="midi-download-btn" download>
    Download Prompt MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `distortion_guitar` | Context |
| 1 | `slap_bass_1` | Context |
| 2 | `drums` | **Generated** |

=== "Sparse (`note_density = 1`)"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_density1.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_density1.mid" class="midi-download-btn" download>
        Download Sparse Drums MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # distortion_guitar (context)
            TrackPrompt(id=1, bars=[]),  # slap_bass_1 (context)
            TrackPrompt(id=2, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"note_density": 1}),  # drums
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=1),
    )
    ```

=== "Medium (`note_density = 3`)"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_density3.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_density3.mid" class="midi-download-btn" download>
        Download Medium Drums MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # distortion_guitar (context)
            TrackPrompt(id=1, bars=[]),  # slap_bass_1 (context)
            TrackPrompt(id=2, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"note_density": 3}),  # drums
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=3),
    )
    ```

=== "Dense (`note_density = 8`)"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_density8.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_density8.mid" class="midi-download-btn" download>
        Download Dense Drums MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # distortion_guitar (context)
            TrackPrompt(id=1, bars=[]),  # slap_bass_1 (context)
            TrackPrompt(id=2, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"note_density": 8}),  # drums
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=8),
    )
    ```

---

### Example 2: A New Cello Part for a String Quartet
We take the two violins and the viola from the first movement of Beethoven's String Quartet Op. 18 No. 4 (public domain) and generate a new `cello` part, monophonic (`max_polyphony = 0`). Sweeping `max_note_duration` from index `1` (16th notes) to `4` (half notes), plus a `min_note_duration = 4` (sustained) variant, gives five cello lines across the duration range. The controls shift the distribution of note lengths rather than set a hard limit, so a take can still hold a few longer notes.

<div class="midi-example-card">
  <strong>String Quartet Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_quartet.mid" class="midi-download-btn" download>
    Download Prompt MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `violin` | Context |
| 1 | `violin` | Context |
| 2 | `viola` | Context |
| 3 | `cello` | **Generated** |

=== "16th notes (`max_note_duration = 1`)"
    Continuous 16th notes.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_16th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_16th.mid" class="midi-download-btn" download>
        Download 16th notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # violin (context)
            TrackPrompt(id=1, bars=[]),  # violin (context)
            TrackPrompt(id=2, bars=[]),  # viola (context)
            TrackPrompt(id=3, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"max_polyphony": 0, "max_note_duration": 1}),  # cello
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=10),
    )
    ```

=== "8th notes (`max_note_duration = 2`)"
    A sparser line, mostly 8th notes.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_8th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_8th.mid" class="midi-download-btn" download>
        Download 8th notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # violin (context)
            TrackPrompt(id=1, bars=[]),  # violin (context)
            TrackPrompt(id=2, bars=[]),  # viola (context)
            TrackPrompt(id=3, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"max_polyphony": 0, "max_note_duration": 2}),  # cello
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=11),
    )
    ```

=== "Quarter notes (`max_note_duration = 3`)"
    8th and 16th notes with a few longer ones.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_4th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_4th.mid" class="midi-download-btn" download>
        Download Quarter notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # violin (context)
            TrackPrompt(id=1, bars=[]),  # violin (context)
            TrackPrompt(id=2, bars=[]),  # viola (context)
            TrackPrompt(id=3, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"max_polyphony": 0, "max_note_duration": 3}),  # cello
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=12),
    )
    ```

=== "Half notes (`max_note_duration = 4`)"
    8ths, quarters and half notes.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_half.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_half.mid" class="midi-download-btn" download>
        Download Half notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # violin (context)
            TrackPrompt(id=1, bars=[]),  # violin (context)
            TrackPrompt(id=2, bars=[]),  # viola (context)
            TrackPrompt(id=3, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"max_polyphony": 0, "max_note_duration": 4}),  # cello
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=13),
    )
    ```

=== "Sustained (`min_note_duration = 4`)"
    One held half note per bar.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_sustained.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_sustained.mid" class="midi-download-btn" download>
        Download Sustained Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # violin (context)
            TrackPrompt(id=1, bars=[]),  # violin (context)
            TrackPrompt(id=2, bars=[]),  # viola (context)
            TrackPrompt(id=3, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                        attributes={"max_polyphony": 0, "min_note_duration": 4}),  # cello
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=14),
    )
    ```

---

### Example 3: Guitar Part (model-sampled controls)
We take the slap bass and drums from Ivan Stanton's "Savior of Humanity" (CC0) as context and generate a new `distortion_guitar` part in place of the original. No attribute controls are set — the model samples its own controls from the prior, giving two different guitar parts over the same groove.

> **Tip:** When you omit attribute controls, the model draws from its learned style distribution. This is useful for exploring diverse outputs without manually specifying constraints.

<div class="midi-example-card">
  <strong>Original (Context + original guitar)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_solo.mid" class="midi-download-btn" download>
    Download Original MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `distortion_guitar` | **Generated** |
| 1 | `slap_bass_1` | Context |
| 2 | `drums` | Context |

=== "Take 1"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/solo_resample1.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/solo_resample1.mid" class="midi-download-btn" download>
        Download Take 1 MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True),
            # no attributes — the model samples its own controls
            TrackPrompt(id=1, bars=[]),  # bass (context)
            TrackPrompt(id=2, bars=[]),  # drums (context)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=21),
    )
    ```

=== "Take 2"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/solo_resample2.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/solo_resample2.mid" class="midi-download-btn" download>
        Download Take 2 MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True),
            # no attributes — the model samples its own controls
            TrackPrompt(id=1, bars=[]),  # bass (context)
            TrackPrompt(id=2, bars=[]),  # drums (context)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=22),
    )
    ```

---

## 2. Bar Infilling (Inpainting)
Bar infilling regenerates selected bars of a track while the model blends them with the surrounding context: the bars before and after, and the other tracks.

### Example 4: Second Violin, Bars 4–7
We take eight bars of the first movement of Mozart's *Eine kleine Nachtmusik* (public domain), **clear bars 4–7 of the second violin**, and ask the model to infill them with a note-duration control. The first violin, viola and cello play through all 8 bars as context, and the second violin's bars 0–3 anchor the generation. The shaded region in the video marks the infilled bars.

> **Note:** `max_note_duration` is a track-level attribute: it describes the whole track, including the bars that stay as they are, so in a 4-bar infill it is a soft control and not every seed follows it. The two takes below were picked from several seeds (the 16th-note one took the most tries). Track sampling (Example 2) follows the same control much more reliably.

<div class="midi-example-card">
  <strong>Infill Prompt (second violin bars 4–7 empty)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_violin_infill.mid" class="midi-download-btn" download>
    Download Seed MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `violin` | Context |
| 1 | `violin` | Context (bars 0–3) + **Generated** (bars 4–7) |
| 2 | `viola` | Context |
| 3 | `cello` | Context |

=== "16th notes (`max_note_duration = 1`)"
    Running 16th notes, continuing the figure of bars 0–3.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/violin_infill_16th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/violin_infill_16th.mid" class="midi-download-btn" download>
        Download 16th notes Infill MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # violin I (context)
            TrackPrompt(id=1, bars=[4, 5, 6, 7], attributes={"max_note_duration": 1}),  # violin II
            TrackPrompt(id=2, bars=[]),  # viola (context)
            TrackPrompt(id=3, bars=[]),  # cello (context)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=134),
    )
    ```

=== "8th notes (`max_note_duration = 2`)"
    A calmer 8th-note line.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/violin_infill_8th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/violin_infill_8th.mid" class="midi-download-btn" download>
        Download 8th notes Infill MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[]),  # violin I (context)
            TrackPrompt(id=1, bars=[4, 5, 6, 7], attributes={"max_note_duration": 2}),  # violin II
            TrackPrompt(id=2, bars=[]),  # viola (context)
            TrackPrompt(id=3, bars=[]),  # cello (context)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", seed=102),
    )
    ```

---

## 3. Multi-Step Composition
By chaining multiple generation tasks you can build up a full arrangement incrementally.

### Example 5: Multi-Step Composition
Starting from the first four bars of a piano, bass and strings passage from Ivan Stanton's "The Ritual" (CC0), we perform three successive generations to build a 5-track arrangement:

1. **Continuation** — extend all three tracks from bar 4 to bar 7 autoregressively
2. **Track sampling** — add `acoustic_guitar_nylon` (monophonic 16th notes)
3. **Track sampling** — add `drums` at high density

The final video shows all five tracks. The gold tracks (guitar and drums) come from steps 2 and 3; the shaded region (bars 4–7) marks the continuation of the first three tracks.

<div class="midi-example-card">
  <strong>Prompt (bars 0–3)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_multistep.mid" class="midi-download-btn" download>
    Download Prompt MIDI
  </a>
</div>

<div class="midi-example-card">
  <strong>All 5 Tracks (Final State)</strong>
  <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/multistep_final.mp4"></video>
  <a href="/MIDI-GPT/assets/midi/multistep_final.mid" class="midi-download-btn" download>
    Download Final MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `acoustic_grand_piano` | Context (bars 0–3) + Continued (bars 4–7) |
| 1 | `electric_bass_finger` | Context (bars 0–3) + Continued (bars 4–7) |
| 2 | `string_ensemble_1` | Context (bars 0–3) + Continued (bars 4–7) |
| 3 | `acoustic_guitar_nylon` | **Generated** — `max_polyphony = 0, max_note_duration = 1` |
| 4 | `drums` | **Generated** — `note_density = 9` |

```python
# Step 1: continue all three tracks for bars 4-7 (one track per step, each sees the others)
step1 = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[4, 5, 6, 7], autoregressive=True),  # piano
        TrackPrompt(id=1, bars=[4, 5, 6, 7], autoregressive=True),  # bass
        TrackPrompt(id=2, bars=[4, 5, 6, 7], autoregressive=True),  # strings
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=41),
)

# Step 2: add a guitar track (monophonic 16th notes) to the result of step 1
step2 = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[]),  # piano (context)
        TrackPrompt(id=1, bars=[]),  # bass (context)
        TrackPrompt(id=2, bars=[]),  # strings (context)
        TrackPrompt(id=3, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                    attributes={"max_polyphony": 0, "max_note_duration": 1}),
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=42),
)

# Step 3: add drums to the result of step 2
step3 = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[]),
        TrackPrompt(id=1, bars=[]),
        TrackPrompt(id=2, bars=[]),
        TrackPrompt(id=3, bars=[]),  # guitar (context)
        TrackPrompt(id=4, bars=[0, 1, 2, 3, 4, 5, 6, 7], autoregressive=True,
                    attributes={"note_density": 9}),
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", seed=43),
)
```

Before steps 2 and 3, append an empty track to the score for the new part: `score.tracks.append(Track(bars=[Bar() for _ in range(8)], instrument=24))` for the nylon guitar, `Track(..., track_type="drum")` for the drums.

---

## Sources

- "Just Say No To Yes", "Savior of Humanity" and "The Ritual" by Ivan Stanton (northivanastan), [CC0](https://creativecommons.org/publicdomain/zero/1.0/), via OpenGameArt.org.
- Beethoven, String Quartet Op. 18 No. 4, mvt. 1, and Mozart, *Eine kleine Nachtmusik* KV 525, mvt. 1: public-domain editions from the [Mutopia Project](https://www.mutopiaproject.org).
- Audio in the videos is rendered with the FluidR3_GM SoundFont (MIT).

The excerpts are taken from the files in `tests/midi/`; see `tests/midi/SOURCES.md`.
