# midigpt

**GPT-2 transformer for controllable, multitrack symbolic music generation.**

- **Fill in missing bars** while preserving your existing arrangement
- **Generate new tracks** from scratch, conditioned on musical attributes
- **Steer the output** by controlling note density, polyphony, and note duration — globally or per bar
- **Integrate with your DAW** via a real-time OSC server
- **One-line setup** — load pretrained models from HuggingFace Hub, no compiler needed

## Quick start

```bash
pip install "midigpt[inference]"
```

```python
from midigpt import Score, Track, Bar
from midigpt.inference import InferenceEngine, GenerationRequest, InferenceConfig, TrackPrompt

engine = InferenceEngine.from_pretrained("yellow")

# 4-bar score with one empty melodic track
score = Score(tracks=[Track(bars=[Bar() for _ in range(4)])])

result = engine.session(
    score,
    GenerationRequest(
        tracks=[TrackPrompt(id=0, bars=[0, 1, 2, 3])],
        config=InferenceConfig(model_dim=4, mask_mode="attention"),
    ),
).run()

total = sum(len(b.notes) for t in result.tracks for b in t.bars)
print(f"Generated {total} notes")
result.to_midi("output.mid")
```

The model is downloaded once and cached by `huggingface_hub` in `~/.cache/huggingface/hub/`.

## Citation

```bibtex
@misc{pasquier2025midigptcontrollablegenerativemodel,
      title={MIDI-GPT: A Controllable Generative Model for Computer-Assisted Multitrack Music Composition},
      author={Philippe Pasquier and Jeff Ens and Nathan Fradet and Paul Triana and Davide Rizzotti and Jean-Baptiste Rolland and Maryam Safi},
      year={2025},
      eprint={2501.17011},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2501.17011},
}
```
