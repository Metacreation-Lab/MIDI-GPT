# Examples & Controllable Generation

This page showcases **MIDI-GPT**'s core capabilities using recognizable, real-world musical examples: **Track Sampling**, **Bar Infilling (Inpainting)**, and **Autoregressive Continuation**. All examples below are generated using the public `yellow` model checkpoint.

By setting **Attribute Controls** (such as polyphony, note density, and note duration), you can guide the model to generate musical parts that perfectly fit your composition.

---

## Video Renders & MIDI Downloads
Generated examples below are rendered as MP4 videos with synthesized audio and a per-track piano-roll view. Context tracks are shown in **blue/green**, generated tracks in **gold**. Download links are provided for importing the `.mid` files into a DAW.

---

## 1. Controllable Track Sampling (Arrangement)
Track sampling generates a completely new track conditioned on one or more existing tracks. The model can also **sample its own attribute controls** from the prior — letting it choose style freely — or you can pin specific values to steer the output.

### Example 1: Rock/Pop Drums — *Californication*
We take the guitar (`acoustic_guitar_steel`) and bass (`electric_bass_finger`) from RHCP's ***Californication*** as context and generate a drum track. By adjusting `note_density` (scale `0–9`), we steer the drums from a near-silent beat to a dense, active pattern.

<div class="midi-example-card">
  <strong>RHCP — Californication Guitar &amp; Bass Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_californication.mid" class="midi-download-btn" download>
    Download Prompt MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `acoustic_guitar_steel` | Context |
| 1 | `electric_bass_finger` | Context |
| 2 | `drums` | **Generated** |

=== "Sparse (`note_density = 1`)"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_calif_density1.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_calif_density1.mid" class="midi-download-btn" download>
        Download Sparse Drums MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # acoustic_guitar_steel (context)
            TrackPrompt(id=1, bars=[], ignore=False),  # electric_bass_finger (context)
            TrackPrompt(
                id=2,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"note_density": 1}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Medium (`note_density = 3`)"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_calif_density3.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_calif_density3.mid" class="midi-download-btn" download>
        Download Medium Drums MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(
                id=2,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"note_density": 3}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Dense (`note_density = 8`)"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_calif_density8.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_calif_density8.mid" class="midi-download-btn" download>
        Download Dense Drums MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(
                id=2,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"note_density": 8}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

---

### Example 2: Classical Cello — Bach's *Harpsichord Concerto*
We take the `violin`, `violin`, and `viola` tracks from J.S. Bach's ***Harpsichord Concerto in D Minor (BWV 1052)*** (solo section, bars 12–20) as context and generate a `cello` bassline restricted to monophonic (`max_polyphony = 0`). By sweeping `max_note_duration` from index `1` (16th notes) through index `4` (half notes) and adding a `min_note_duration = 4` (sustained legato) variant, we get 5 generations spanning the full duration spectrum.

<div class="midi-example-card">
  <strong>J.S. Bach — BWV 1052 Strings Ensemble Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_bach.mid" class="midi-download-btn" download>
    Download Prompt MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `violin` | Context |
| 1 | `viola` | Context |
| 2 | `cello` | Context |
| 3 | `cello` | **Generated** |

=== "16th notes (`max_note_duration = 1`)"
    Rapid, bouncy staccato lines.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_16th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_16th.mid" class="midi-download-btn" download>
        Download 16th Notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # violin (context)
            TrackPrompt(id=1, bars=[], ignore=False),  # viola (context)
            TrackPrompt(id=2, bars=[], ignore=False),  # cello (context)
            TrackPrompt(
                id=3,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"max_polyphony": 0, "max_note_duration": 1}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "8th notes (`max_note_duration = 2`)"
    Moderate 8th-note runs.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_8th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_8th.mid" class="midi-download-btn" download>
        Download 8th Notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False),
            TrackPrompt(
                id=3,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"max_polyphony": 0, "max_note_duration": 2}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Quarter notes (`max_note_duration = 3`)"
    Quarter-note walking-bass feel.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_4th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_4th.mid" class="midi-download-btn" download>
        Download Quarter Notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False),
            TrackPrompt(
                id=3,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"max_polyphony": 0, "max_note_duration": 3}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Half notes (`max_note_duration = 4`)"
    Slow half-note pedal points.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_dur_half.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_dur_half.mid" class="midi-download-btn" download>
        Download Half Notes Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False),
            TrackPrompt(
                id=3,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"max_polyphony": 0, "max_note_duration": 4}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Sustained legato (`min_note_duration = 4`)"
    Long held bass notes — pure legato phrasing.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_sustained.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_sustained.mid" class="midi-download-btn" download>
        Download Sustained Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False),
            TrackPrompt(
                id=3,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False,
                attributes={"max_polyphony": 0, "min_note_duration": 4}
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

---

### Example 3: Guitar Solo — *Black Magic Woman* (model-sampled controls)
We take the `electric_bass_finger` and `drums` tracks from Santana's ***Black Magic Woman*** as context and resample the `overdriven_guitar` solo. No explicit attribute controls are set — the model samples its own controls from the prior, producing two stylistically distinct solos in the same harmonic context.

> **Tip:** When you omit attribute controls, the model draws from its learned style distribution. This is useful for exploring diverse outputs without manually specifying constraints.

<div class="midi-example-card">
  <strong>Black Magic Woman — Original (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_bmw.mid" class="midi-download-btn" download>
    Download Original MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `electric_bass_finger` | Context |
| 1 | `overdriven_guitar` | **Generated** |
| 2 | `drums` | Context |

=== "Resample 1"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/bmw_solo_resample1.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/bmw_solo_resample1.mid" class="midi-download-btn" download>
        Download Resample 1 MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # electric_bass_finger (context)
            TrackPrompt(
                id=1,
                bars=[0, 1, 2, 3, 4, 5, 6, 7],
                ignore=False
                # no attributes — model samples its own controls
            ),
            TrackPrompt(id=2, bars=[], ignore=False),  # drums (context)
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Resample 2"

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/bmw_solo_resample2.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/bmw_solo_resample2.mid" class="midi-download-btn" download>
        Download Resample 2 MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[0, 1, 2, 3, 4, 5, 6, 7], ignore=False),
            TrackPrompt(id=2, bars=[], ignore=False),
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

---

## 2. Controllable Bar Infilling (Inpainting)
Bar infilling lets you select specific bars in any track and regenerate them while the model blends with the surrounding context.

### Example 4: Orchestral Harp — *SSBM All Star Intro*
We take the full harp, bells, and strings tracks from the ***Super Smash Bros. Melee All Star Intro*** and **delete bars 4–7 of the harp**, asking the model to infill them with a note-duration constraint. The `tubular_bells` and `string_ensemble_1` tracks play through all 8 bars as fixed context; the original harp bars 0–3 anchor the generation. The shaded region in the video marks the infilled bars.

<div class="midi-example-card">
  <strong>SSBM — Harp Infill Prompt (Bars 4–7 empty)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_harp.mid" class="midi-download-btn" download>
    Download Seed MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `tubular_bells` | Context |
| 1 | `string_ensemble_1` | Context |
| 2 | `orchestral_harp` | Context (bars 0–3) + **Generated** (bars 4–7) |

=== "16th notes (`max_note_duration = 1`)"
    Rapid 16th-note arpeggiated runs.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/infill_allstar_dur_16th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/infill_allstar_dur_16th.mid" class="midi-download-btn" download>
        Download 16th Notes Harp MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # tubular_bells (context)
            TrackPrompt(id=1, bars=[], ignore=False),  # string_ensemble_1 (context)
            TrackPrompt(
                id=2,
                bars=[4, 5, 6, 7],
                autoregressive=False,  # infill mode
                attributes={"max_note_duration": 1}
            ),
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```

=== "8th notes (`max_note_duration = 2`)"
    Slower 8th-note arpeggios — more lyrical.

    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/infill_allstar_dur_8th.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/infill_allstar_dur_8th.mid" class="midi-download-btn" download>
        Download 8th Notes Harp MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),
            TrackPrompt(id=1, bars=[], ignore=False),
            TrackPrompt(
                id=2,
                bars=[4, 5, 6, 7],
                autoregressive=False,
                attributes={"max_note_duration": 2}
            ),
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```

---

## 3. Multi-Step Composition
By chaining multiple generation tasks you can build up a full arrangement incrementally.

### Example 5: Multi-Step — Joe Hisaishi's *Summer*
Starting from a 3-track `acoustic_grand_piano` + `electric_bass_finger` + `synth_strings_1` context, we perform three successive generations to build a 5-track arrangement:

1. **Continuation** — extend all three tracks from bar 4 to bar 7 autoregressively
2. **Track sampling** — add `acoustic_guitar_nylon` (monophonic 16th notes)
3. **Track sampling** — add `drums` at high density

The final video shows all five tracks simultaneously. Gold tracks (`acoustic_guitar_nylon`, `drums`) are the last two generation steps; the shaded region (bars 4–7) marks the continuation portion of the first three tracks.

<div class="midi-example-card">
  <strong>Joe Hisaishi — Summer (all 5 tracks, final state)</strong>
  <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/summer_final.mp4"></video>
  <a href="/MIDI-GPT/assets/midi/summer_final.mid" class="midi-download-btn" download>
    Download Final MIDI
  </a>
</div>

| Track | Instrument | Role |
|-------|-----------|------|
| 0 | `acoustic_grand_piano` | Context (bars 0–3) + Continued (bars 4–7) |
| 1 | `electric_bass_finger` | Context (bars 0–3) + Continued (bars 4–7) |
| 2 | `synth_strings_1` | Context (bars 0–3) + Continued (bars 4–7) |
| 3 | `acoustic_guitar_nylon` | **Generated** — `max_polyphony=0, max_note_duration=1` |
| 4 | `drums` | **Generated** — `note_density=9` |

```python
# Step 1: Continue all three tracks for bars 4-7
step1 = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[4, 5, 6, 7], autoregressive=True),  # piano
        TrackPrompt(id=1, bars=[4, 5, 6, 7], autoregressive=True),  # bass
        TrackPrompt(id=2, bars=[4, 5, 6, 7], autoregressive=True),  # strings
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention")
)

# Step 2: Add guitar (monophonic 16th notes)
step2 = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),  # piano (context)
        TrackPrompt(id=1, bars=[], ignore=False),  # bass (context)
        TrackPrompt(id=2, bars=[], ignore=False),  # strings (context)
        TrackPrompt(
            id=3,
            bars=[0, 1, 2, 3, 4, 5, 6, 7],
            ignore=False,
            attributes={"max_polyphony": 0, "max_note_duration": 1}
        ),
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
)

# Step 3: Add drums
step3 = GenerationRequest(
    tracks=[
        TrackPrompt(id=0, bars=[], ignore=False),
        TrackPrompt(id=1, bars=[], ignore=False),
        TrackPrompt(id=2, bars=[], ignore=False),
        TrackPrompt(id=3, bars=[], ignore=False),  # guitar (context)
        TrackPrompt(
            id=4,
            bars=[0, 1, 2, 3, 4, 5, 6, 7],
            ignore=False,
            attributes={"note_density": 9}
        ),
    ],
    config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
)
```
