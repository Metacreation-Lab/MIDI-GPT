# Examples & Controllable Generation

This page showcases **MIDI-GPT**'s core capabilities using recognizable, real-world musical examples from 5 different styles: **Track Sampling**, **Bar Infilling (Inpainting)**, and **Autoregressive Continuation**. All examples below are generated using the public `yellow` model checkpoint.

By setting **Attribute Controls** (such as polyphony, note density, and note duration), you can guide the model to generate musical parts that perfectly fit your composition.

---

## Video Renders & MIDI Downloads
Generated examples below are rendered as MP4 videos with synthesized audio and a piano-roll view. Download links are also provided for importing the `.mid` files into a DAW.

---

## 1. Controllable Track Sampling (Arrangement)
Track sampling is the process of generating a completely new track (e.g., accompaniment) conditioned on one or more existing tracks (e.g., a guide melody or chord progression).

### Example 1: Rock/Pop Drums — *Hotel California*
We take the iconic 12-string guitar track and bassline from the Eagles' ***Hotel California*** (bars 16–24) as the multi-track context, and generate an accompanying drum track. By adjusting `note_density` (scale `0-9`), we steer the drums from a sparse beat to an active, dense pattern.

<div class="midi-example-card">
  <strong>Eagles — Hotel California Guitar & Bass Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_sampling.mid" class="midi-download-btn" download>
    Download Prompt MIDI
  </a>
</div>

=== "Sparse Drums (`note_density = 1`)"
    A minimal, laid-back beat matching the tempo and feel of the guitar and bass chords.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_sparse.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_sparse.mid" class="midi-download-btn" download>
        Download Sparse Drums MIDI
      </a>
    </div>

    ```python
    # Request config
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # Guitar (context only)
            TrackPrompt(id=1, bars=[], ignore=False),  # Bass (context only)
            TrackPrompt(
                id=2, 
                bars=[0, 1, 2, 3, 4, 5, 6, 7], 
                ignore=False, 
                attributes={"note_density": 1}  # Quantized level 1 (out of 10)
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Dense Drums (`note_density = 8`)"
    An active, busy drum pattern with high syncopation and rapid subdivisions.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_drums_dense.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_drums_dense.mid" class="midi-download-btn" download>
        Download Dense Drums MIDI
      </a>
    </div>

    ```python
    # Request config
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # Guitar
            TrackPrompt(id=1, bars=[], ignore=False),  # Bass
            TrackPrompt(
                id=2, 
                bars=[0, 1, 2, 3, 4, 5, 6, 7], 
                ignore=False, 
                attributes={"note_density": 8}  # Quantized level 8 (out of 10)
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

---

### Example 2: Classical Bassline — Bach's *Harpsichord Concerto*
We take the Violin 1, Violin 2, and Viola tracks from J.S. Bach's ***Harpsichord Concerto in D Minor (BWV 1052)*** (solo section, bars 12–20) as the multi-track context, and generate a cello/continuo bassline. We restrict the cello to be monophonic (`max_polyphony = 0`) and steer its rhythm using note duration boundaries.

<div class="midi-example-card">
  <strong>J.S. Bach — BWV 1052 Strings Ensemble Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_bach.mid" class="midi-download-btn" download>
    Download Prompt MIDI
  </a>
</div>

=== "Staccato Cello (`max_note_duration = 1`)"
    By capping the maximum note duration to index `1` (16th notes), the cello is forced to play rapid, bouncy staccato lines.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_staccato.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_staccato.mid" class="midi-download-btn" download>
        Download Staccato Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # Violin 1
            TrackPrompt(id=1, bars=[], ignore=False),  # Violin 2
            TrackPrompt(id=2, bars=[], ignore=False),  # Viola
            TrackPrompt(
                id=3, 
                bars=[0, 1, 2, 3, 4, 5, 6, 7], 
                ignore=False, 
                attributes={
                    "max_polyphony": 0,      # strictly monophonic (1 voice)
                    "max_note_duration": 1   # cap note lengths to 16th notes
                }
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

=== "Legato/Sustained Cello (`min_note_duration = 4`)"
    By locking the minimum note duration to index `4` (half notes), the model is steered to generate long, sustained bass notes (pedal points/legato style).
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/track_sampling_cello_sustained.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/sampling_cello_sustained.mid" class="midi-download-btn" download>
        Download Sustained Cello MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # Violin 1
            TrackPrompt(id=1, bars=[], ignore=False),  # Violin 2
            TrackPrompt(id=2, bars=[], ignore=False),  # Viola
            TrackPrompt(
                id=3, 
                bars=[0, 1, 2, 3, 4, 5, 6, 7], 
                ignore=False, 
                attributes={
                    "max_polyphony": 0,      # strictly monophonic
                    "min_note_duration": 4   # minimum note duration: half notes
                }
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention", tracks_per_step=2)
    )
    ```

---

## 2. Controllable Bar Infilling (Inpainting)
Bar infilling allows you to select specific bars in the middle of a track and regenerate them while the model automatically blends the generation with the surrounding context (preceding and succeeding bars).

### Example 3: Electronic Keyboard — Radiohead's *Everything In Its Right Place*
We take the iconic repeating synth keyboard progression, bassline, and drum track from Radiohead's ***Everything In Its Right Place*** (bars 16–24) and **delete the keyboard notes in bars 3 and 4** (indices 3 and 4 of the 8-bar slice), leaving them empty. The model sees the surrounding keyboard bars and the full bass and drums tracks as fixed context, and infills the keyboard gap.

<div class="midi-example-card">
  <strong>Radiohead — Infill Prompt (Bars 3 & 4 empty)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_infill.mid" class="midi-download-btn" download>
    Download Infill Prompt
  </a>
</div>

---

### Contrasting Infills
We ask the model to regenerate only bars `3` and `4` of the keyboard line.

=== "Sparse, Sustained Infill"
    Generates a simple, monophonic transition using longer notes.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/bar_infill_sparse.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/infill_sparse.mid" class="midi-download-btn" download>
        Download Sparse Infill MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(
                id=0,
                bars=[3, 4],             # target bars to generate
                autoregressive=False,    # infill mode
                attributes={
                    "max_polyphony": 0,      # monophonic (1 voice)
                    "min_note_duration": 3   # minimum note duration: quarter notes
                }
            ),
            TrackPrompt(id=1, bars=[], ignore=False),  # Bass context
            TrackPrompt(id=2, bars=[], ignore=False)   # Drums context
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```

=== "Chordal/Polyphonic Infill"
    Generates complex multi-note chords and dense transitions, preserving the keyboard texture.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/bar_infill_chordal.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/infill_chordal.mid" class="midi-download-btn" download>
        Download Chordal Infill MIDI
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(
                id=0,
                bars=[3, 4],             # target bars to generate
                autoregressive=False,    # infill mode
                attributes={
                    "min_polyphony": 2,      # at least 3 voices
                    "max_polyphony": 4       # up to 5 voices
                }
            ),
            TrackPrompt(id=1, bars=[], ignore=False),  # Bass context
            TrackPrompt(id=2, bars=[], ignore=False)   # Drums context
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```

---

## 3. Autoregressive Continuation (Composition)
Autoregressive continuation takes a short musical motif (seed) and extends it forward in time.

### Example 4: Piano Melody — Joe Hisaishi's *Summer*
The first 4 bars of the piano, bass, and strings tracks from Joe Hisaishi's ***Summer*** (bars 13–21) are used as the seed. The bass and strings tracks play through the entire 8 bars as fixed context. We ask the model to generate the next 4 bars of the piano (bars 4–7) autoregressively.

<div class="midi-example-card">
  <strong>Joe Hisaishi — Summer Piano, Bass & Strings Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_ar.mid" class="midi-download-btn" download>
    Download Seed MIDI
  </a>
</div>

=== "Monophonic Lead Continuation"
    The continuation is kept monophonic, producing a single-note piano melody winding out from the seed.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/ar_monophonic.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/ar_monophonic.mid" class="midi-download-btn" download>
        Download Monophonic Continuation
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(
                id=0,
                bars=[4, 5, 6, 7],      # generate 4 bars after the 4-bar seed
                autoregressive=True,    # continue forward
                attributes={
                    "max_polyphony": 0  # restrict continuation to 1 voice (monophonic lead)
                }
            ),
            TrackPrompt(id=1, bars=[], ignore=False),  # Bass context
            TrackPrompt(id=2, bars=[], ignore=False)   # Strings context
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```

=== "Chordal Continuation"
    The continuation is allowed to have rich chords, maintaining the piano accompaniment texture.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/ar_chordal.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/ar_chordal.mid" class="midi-download-btn" download>
        Download Chordal Continuation
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(
                id=0,
                bars=[4, 5, 6, 7],
                autoregressive=True,
                attributes={
                    "min_polyphony": 2,  # minimum 3 voices
                    "max_polyphony": 4   # maximum 5 voices
                }
            ),
            TrackPrompt(id=1, bars=[], ignore=False),  # Bass context
            TrackPrompt(id=2, bars=[], ignore=False)   # Strings context
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```

---

### Example 5: Video Game Theme — *Super Smash Bros. Melee All Star Intro*
The first 4 bars of the Orchestral Harp, Bells, and Strings flourish from the ***Super Smash Bros. Melee All Star Intro*** (bars 0–8) are used as the seed. The bells and strings tracks play through the entire 8 bars as fixed context. We ask the model to generate the next 4 bars of the harp (bars 4–7) autoregressively.

<div class="midi-example-card">
  <strong>SSBM — Harp, Bells & Strings Prompt (Context)</strong>
  <a href="/MIDI-GPT/assets/midi/prompt_harp.mid" class="midi-download-btn" download>
    Download Seed MIDI
  </a>
</div>

=== "Rapid Arpeggios (`max_note_duration = 1`)"
    Restricts the generated harp notes to rapid 16th-note arpeggiated runs.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/harp_monophonic.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/harp_monophonic.mid" class="midi-download-btn" download>
        Download Monophonic Continuation
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # Bells context
            TrackPrompt(id=1, bars=[], ignore=False),  # Strings context
            TrackPrompt(
                id=2,
                bars=[4, 5, 6, 7],      # generate 4 bars after the 4-bar seed
                autoregressive=True,    # continue forward
                attributes={
                    "max_note_duration": 1  # cap note lengths to 16th notes
                }
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```

=== "Sustained Chords (`min_note_duration = 4`)"
    Enables rich, sustained chords that fit the epic intro orchestrations.
    
    <div class="midi-example-card">
      <video class="midi-example-video" controls preload="metadata" src="/MIDI-GPT/assets/video/harp_chordal.mp4"></video>
      <a href="/MIDI-GPT/assets/midi/harp_chordal.mid" class="midi-download-btn" download>
        Download Chordal Continuation
      </a>
    </div>

    ```python
    req = GenerationRequest(
        tracks=[
            TrackPrompt(id=0, bars=[], ignore=False),  # Bells context
            TrackPrompt(id=1, bars=[], ignore=False),  # Strings context
            TrackPrompt(
                id=2,
                bars=[4, 5, 6, 7],
                autoregressive=True,
                attributes={
                    "min_note_duration": 4  # minimum note duration: half notes
                }
            )
        ],
        config=InferenceConfig(model_dim=8, mask_mode="attention")
    )
    ```
