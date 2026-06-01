# SoundFonts

SF2 files are not included in the repository due to size and licensing.

Place your SoundFont files here so the studio can find them.

## Recommended: Arachno SoundFont

Arachno SoundFont is a high-quality GM SoundFont used as the default in
MIDI-GPT Studio.

1. Download from the official site: https://www.arachnosoft.com/main/soundfont.php
2. Rename the downloaded file to `arachno.sf2`
3. Place it in this directory (`src/python/midigpt/osc/studio/static/sf2/`)

## Other SoundFonts

Any GM-compatible SF2 file works. Place it here and select it in the studio
interface, or pass `--sf2 path/to/your.sf2` to `midigpt-studio`.

The `default.sf2` (5.7 MB, already in the repo) is a smaller fallback
SoundFont used when no other file is available.
