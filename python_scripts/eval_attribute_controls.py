#!/usr/bin/env python3
"""Evaluate how well MIDI-GPT respects attribute controls during generation.

Dynamically discovers the encoder's attribute controls and evaluates how well
the model follows them. Works with any encoder type (ExpressiveEncoder,
SteinbergWPCSEncoder, etc.).

Randomly samples attribute control configurations, runs generation, and
compares requested vs actual control values using precision, recall, F1,
cross-entropy, mean absolute distance, and directional accuracy.

Usage:
    python eval_attribute_controls.py \
        --ckpt /path/to/model.pt \
        --encoder STEINBERG_WPCS_ENCODER \
        --midi_dir tests/midi_files/ \
        --n_configs 10 \
        --seed 42 \
        --output eval_results.json
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import midigpt


# ---------------------------------------------------------------------------
# Attribute control registry
# ---------------------------------------------------------------------------
# Each entry maps an ATTRIBUTE_CONTROL_TYPE enum name to metadata:
#   status_fields: list of (status_field_name, n_classes, lo, hi) tuples
#                  lo..hi is the 1-indexed sampling range (0 = ANY)
#   feature_fields: list of corresponding TrackFeatures JSON keys (camelCase)
#   level: "track" or "bar"
#   track_type: "instrument", "drum", or "both"
#   paired_min_max: if True, fields come in min/max pairs (sample min <= max)

CONTROL_REGISTRY = {
    # --- ExpressiveEncoder controls ---
    "ATTRIBUTE_CONTROL_POLYPHONY_QUANTILE": {
        "status_fields": [
            ("min_polyphony_q", 6, 1, 6),
            ("max_polyphony_q", 6, 1, 6),
        ],
        "feature_fields": ["minPolyphonyQ", "maxPolyphonyQ"],
        "level": "track",
        "track_type": "instrument",
        "paired_min_max": True,
    },
    "ATTRIBUTE_CONTROL_NOTE_DURATION_QUANTILE": {
        "status_fields": [
            ("min_note_duration_q", 6, 1, 6),
            ("max_note_duration_q", 6, 1, 6),
        ],
        "feature_fields": ["minNoteDurationQ", "maxNoteDurationQ"],
        "level": "track",
        "track_type": "instrument",
        "paired_min_max": True,
    },
    "ATTRIBUTE_CONTROL_NOTE_DENSITY": {
        "status_fields": [("note_density_level", 10, 1, 10)],
        "feature_fields": ["noteDensityLevel"],
        "level": "track",
        "track_type": "drum",
        "paired_min_max": False,
    },
    # --- SteinbergWPCSEncoder / shared controls ---
    "ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_DENSITY": {
        "status_fields": [
            ("onset_density_min", 17, 1, 17),
            ("onset_density_max", 17, 1, 17),
        ],
        "feature_fields": ["onsetDensityMin", "onsetDensityMax"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": True,
    },
    "ATTRIBUTE_CONTROL_TRACK_LEVEL_ONSET_POLYPHONY": {
        "status_fields": [
            ("onset_polyphony_min", 6, 1, 6),
            ("onset_polyphony_max", 6, 1, 6),
        ],
        "feature_fields": ["onsetPolyphonyMin", "onsetPolyphonyMax"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": True,
    },
    "ATTRIBUTE_CONTROL_TRACK_LEVEL_NOTE_DURATION": {
        "status_fields": [
            ("contains_note_duration_thirty_second", 2, 1, 2),
            ("contains_note_duration_sixteenth", 2, 1, 2),
            ("contains_note_duration_eighth", 2, 1, 2),
            ("contains_note_duration_quarter", 2, 1, 2),
            ("contains_note_duration_half", 2, 1, 2),
            ("contains_note_duration_whole", 2, 1, 2),
        ],
        "feature_fields": [
            "containsNoteDurationThirtySecond",
            "containsNoteDurationSixteenth",
            "containsNoteDurationEighth",
            "containsNoteDurationQuarter",
            "containsNoteDurationHalf",
            "containsNoteDurationWhole",
        ],
        "level": "track",
        "track_type": "instrument",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_REPETITION": {
        "status_fields": [("repetition", 10, 1, 10)],
        "feature_fields": ["repetition"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_GENRE": {
        "status_fields": [("genre", 30, 1, 30)],
        "feature_fields": ["genre"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_BAR_LEVEL_PITCH_CLASS_SET": {
        # Bar-level control with 12 boolean pitch classes; handled specially
        "status_fields": [],  # set on StatusBar, not StatusTrack
        "feature_fields": [],
        "level": "bar",
        "track_type": "instrument",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_PITCH_RANGE": {
        "status_fields": [
            ("min_pitch", 128, 0, 127),
            ("max_pitch", 128, 0, 127),
        ],
        "feature_fields": ["minPitch", "maxPitch"],
        "level": "track",
        "track_type": "instrument",
        "paired_min_max": True,
    },
    "ATTRIBUTE_CONTROL_TRACK_LEVEL_SILENCE_PROPORTION": {
        "status_fields": [
            ("silence_proportion_min", 10, 1, 10),
            ("silence_proportion_max", 10, 1, 10),
        ],
        "feature_fields": ["silenceProportionMin", "silenceProportionMax"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": True,
    },
    "ATTRIBUTE_CONTROL_BAR_LEVEL_ONSET_DENSITY": {
        "status_fields": [],  # set on StatusBar
        "feature_fields": [],
        "level": "bar",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_BAR_LEVEL_ONSET_POLYPHONY": {
        "status_fields": [],  # set on StatusBar
        "feature_fields": [],
        "level": "bar",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_PITCH_CLASS_COUNT": {
        "status_fields": [("pitch_class_count", 12, 1, 12)],
        "feature_fields": ["pitchClassCount"],
        "level": "track",
        "track_type": "instrument",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_KEY_SIGNATURE": {
        "status_fields": [("key_signature", 24, 1, 24)],
        "feature_fields": ["keySignature"],
        "level": "track",
        "track_type": "instrument",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_VALENCE_SPOTIFY": {
        "status_fields": [("valence_spotify", 10, 1, 10)],
        "feature_fields": ["valenceSpotify"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_ENERGY_SPOTIFY": {
        "status_fields": [("energy_spotify", 10, 1, 10)],
        "feature_fields": ["energySpotify"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_DANCEABILITY_SPOTIFY": {
        "status_fields": [("danceability_spotify", 10, 1, 10)],
        "feature_fields": ["danceabilitySpotify"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_DANCEABILITY": {
        "status_fields": [("danceability", 10, 1, 10)],
        "feature_fields": ["danceability"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_TENSION": {
        "status_fields": [],  # bar-level: set on StatusBar
        "feature_fields": [],
        "level": "bar",
        "track_type": "both",
        "paired_min_max": False,
    },
    "ATTRIBUTE_CONTROL_WNBD_SYNCOPATION": {
        "status_fields": [("wnbd_syncopation", 10, 1, 10)],
        "feature_fields": ["wnbdSyncopation"],
        "level": "track",
        "track_type": "both",
        "paired_min_max": False,
    },
}


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

# Map encoder type name → Python class in midigpt
ENCODER_CLASSES = {
    "EXPRESSIVE_ENCODER": "ExpressiveEncoder",
    "STEINBERG_WPCS_ENCODER": "SteinbergWPCSEncoder",
    "STEINBERG_W_P_C_S_ENCODER": "SteinbergWPCSEncoder",
}


def get_encoder(encoder_type: str):
    """Instantiate an encoder by type name."""
    cls_name = ENCODER_CLASSES.get(encoder_type)
    if cls_name is None:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Known types: {list(ENCODER_CLASSES.keys())}"
        )
    cls = getattr(midigpt, cls_name)
    return cls()


def discover_controls(encoder_type: str) -> list[dict]:
    """Discover track-level evaluable controls for a given encoder type.

    Returns a list of control descriptors from CONTROL_REGISTRY that are:
    1. Registered for this encoder's attribute control types
    2. Track-level (bar-level controls are excluded from per-track evaluation)
    3. Have status_fields defined (can be set and measured)
    """
    enc = get_encoder(encoder_type)
    # get_attribute_control_types() returns a list of enum name strings
    control_type_names = enc.get_attribute_control_types()

    active_controls = []
    for ct_name in control_type_names:
        meta = CONTROL_REGISTRY.get(ct_name)
        if meta is None:
            print(f"  WARNING: No registry entry for {ct_name}, skipping")
            continue
        if meta["level"] != "track":
            continue
        if not meta["status_fields"]:
            continue
        active_controls.append({"name": ct_name, **meta})

    return active_controls


# ---------------------------------------------------------------------------
# Random config sampling
# ---------------------------------------------------------------------------

def sample_controls(
    controls: list[dict],
    is_drum: bool,
    rng: random.Random,
) -> dict:
    """Sample random values for active controls applicable to this track type."""
    sampled = {}
    for ctrl in controls:
        tt = ctrl["track_type"]
        if tt == "instrument" and is_drum:
            continue
        if tt == "drum" and not is_drum:
            continue

        if ctrl["paired_min_max"]:
            # Sample min <= max for paired fields
            fields = ctrl["status_fields"]
            assert len(fields) == 2, f"paired_min_max expects 2 fields, got {len(fields)}"
            _, n_cls, lo, hi = fields[0]
            v_min = rng.randint(lo, hi)
            v_max = rng.randint(v_min, hi)
            sampled[fields[0][0]] = v_min
            sampled[fields[1][0]] = v_max
        else:
            for field_name, n_cls, lo, hi in ctrl["status_fields"]:
                sampled[field_name] = rng.randint(lo, hi)

    return sampled


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def load_piece(midi_path: str, encoder_type: str) -> dict:
    """Parse a MIDI file into a Piece dict using the specified encoder."""
    enc = get_encoder(encoder_type)
    return json.loads(enc.midi_to_json(midi_path))


def is_drum_track(track: dict) -> bool:
    tt = track.get("trackType", track.get("track_type", 0))
    return tt == 2


def count_bars(track: dict) -> int:
    return len(track.get("bars", []))


def build_status_track(
    track_idx: int,
    track: dict,
    controls: dict,
    num_bars: int,
    temperature: float = 0.9,
) -> dict:
    """Build a StatusTrack dict for one track."""
    is_drum = is_drum_track(track)
    if num_bars <= 2:
        selected = [True] * num_bars
    else:
        selected = [False] * num_bars
        for i in range(1, num_bars - 1):
            selected[i] = True

    st = {
        "track_id": track_idx,
        "track_type": 2 if is_drum else 10,
        "temperature": temperature,
        "ignore": False,
        "selected_bars": selected,
        "autoregressive": False,
        "polyphony_hard_limit": 0,
    }
    st.update(controls)
    return st


def generate(
    piece_dict: dict,
    status_tracks: list[dict],
    ckpt: str,
    num_bars: int,
    seed: int = -1,
    max_steps: int = 0,
) -> tuple[dict, float]:
    """Run sample_multi_step and return (output_piece_dict, elapsed_seconds)."""
    param = {
        "tracks_per_step": 1,
        "bars_per_step": min(2, num_bars),
        "model_dim": min(4, num_bars),
        "percentage": 100,
        "batch_size": 1,
        "temperature": 0.9,
        "max_steps": max_steps,
        "polyphony_hard_limit": 0,
        "shuffle": False,
        "verbose": False,
        "ckpt": ckpt,
        "sampling_seed": seed,
        "mask_top_k": 0,
    }

    status = {"tracks": status_tracks}
    callbacks = midigpt.CallbackManager()

    piece_json = json.dumps(piece_dict)
    status_json = json.dumps(status)
    param_json = json.dumps(param)

    start = time.perf_counter()
    result = midigpt.sample_multi_step(
        piece_json, status_json, param_json, 3, callbacks
    )
    elapsed = time.perf_counter() - start

    return json.loads(result[0]), elapsed


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(piece_dict: dict) -> dict:
    """Compute attribute controls on a piece and return per-track features."""
    piece_json = json.dumps(piece_dict)
    updated = midigpt.compute_all_attribute_controls(piece_json)
    return json.loads(updated)


def get_actual_values(
    piece_dict: dict,
    track_idx: int,
    controls: list[dict],
    is_drum: bool,
) -> dict:
    """Extract computed attribute control values for a track.

    Returns a dict mapping status_field_name → actual int value.
    """
    track = piece_dict["tracks"][track_idx]
    features_list = track.get("internalFeatures", [])
    if not features_list:
        return {}
    feat = features_list[0]

    result = {}
    for ctrl in controls:
        tt = ctrl["track_type"]
        if tt == "instrument" and is_drum:
            continue
        if tt == "drum" and not is_drum:
            continue

        for (field_name, _, _, _), feat_key in zip(
            ctrl["status_fields"], ctrl["feature_fields"]
        ):
            val = feat.get(feat_key)
            if val is not None:
                result[field_name] = int(val)

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(requested: list[int], actual: list[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for r, a in zip(requested, actual):
        if 0 <= r < n_classes and 0 <= a < n_classes:
            cm[r, a] += 1
    return cm


def compute_metrics(requested: list[int], actual: list[int], n_classes: int) -> dict:
    if not requested:
        return {}

    cm = confusion_matrix(requested, actual, n_classes)

    correct = sum(1 for r, a in zip(requested, actual) if r == a)
    accuracy = correct / len(requested)

    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    active_classes = set(requested)
    active_p = [precisions[c] for c in active_classes if c < n_classes]
    active_r = [recalls[c] for c in active_classes if c < n_classes]
    active_f1 = [f1s[c] for c in active_classes if c < n_classes]

    macro_precision = statistics.mean(active_p) if active_p else 0.0
    macro_recall = statistics.mean(active_r) if active_r else 0.0
    macro_f1 = statistics.mean(active_f1) if active_f1 else 0.0

    mad = statistics.mean(abs(r - a) for r, a in zip(requested, actual))

    eps = 1e-10
    ce_values = []
    for r, a in zip(requested, actual):
        row_sum = cm[r, :].sum()
        if row_sum > 0:
            prob = (cm[r, a] + eps) / (row_sum + eps * n_classes)
            ce_values.append(-math.log(prob))
    cross_entropy = statistics.mean(ce_values) if ce_values else float("inf")

    mid = n_classes / 2
    dir_correct = 0
    dir_total = 0
    for r, a in zip(requested, actual):
        if r < mid - 0.5:
            dir_total += 1
            if a <= mid:
                dir_correct += 1
        elif r > mid + 0.5:
            dir_total += 1
            if a >= mid - 1:
                dir_correct += 1
    directional_accuracy = dir_correct / dir_total if dir_total > 0 else float("nan")

    return {
        "n_samples": len(requested),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mean_absolute_distance": mad,
        "cross_entropy": cross_entropy,
        "directional_accuracy": directional_accuracy,
        "confusion_matrix": cm.tolist(),
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    ckpt: str,
    midi_dir: str,
    encoder_type: str,
    n_configs: int,
    seed: int,
    temperature: float,
    max_steps: int,
) -> dict:
    rng = random.Random(seed)

    # Discover active controls for this encoder
    active_controls = discover_controls(encoder_type)
    if not active_controls:
        print("ERROR: No evaluable track-level controls found for "
              f"encoder {encoder_type}", file=sys.stderr)
        sys.exit(1)

    ctrl_names = [c["name"] for c in active_controls]
    print(f"Encoder: {encoder_type}")
    print(f"Active controls ({len(active_controls)}):")
    for c in active_controls:
        fields = [f[0] for f in c["status_fields"]]
        print(f"  {c['name']}: {fields} ({c['track_type']})")
    print()

    midi_files = sorted(
        list(Path(midi_dir).glob("**/*.mid"))
        + list(Path(midi_dir).glob("**/*.midi"))
    )
    if not midi_files:
        print(f"ERROR: No MIDI files found in {midi_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(midi_files)} MIDI files")
    print(f"Checkpoint: {ckpt}")
    print(f"Configs per file: {n_configs}, seed: {seed}")
    print(f"Temperature: {temperature}, max_steps: {max_steps}")
    print()

    # Collectors: status_field_name → {requested: [], actual: [], n_classes: int}
    collectors: dict[str, dict] = {}
    for ctrl in active_controls:
        for field_name, n_cls, lo, hi in ctrl["status_fields"]:
            collectors[field_name] = {
                "requested": [], "actual": [], "n_classes": n_cls,
                "lo": lo, "control_name": ctrl["name"],
            }

    timing_records = []
    total_generated = 0
    total_skipped = 0

    for midi_path in midi_files:
        print(f"--- {midi_path.name} ---")
        try:
            piece_dict = load_piece(str(midi_path), encoder_type)
        except Exception as e:
            print(f"  SKIP (parse error): {e}")
            total_skipped += 1
            continue

        tracks = piece_dict.get("tracks", [])
        if not tracks:
            print("  SKIP (no tracks)")
            total_skipped += 1
            continue

        num_bars = count_bars(tracks[0])
        if num_bars < 2:
            print("  SKIP (fewer than 2 bars)")
            total_skipped += 1
            continue

        for config_i in range(n_configs):
            gen_seed = rng.randint(0, 2**31 - 1)

            track_idx = rng.randint(0, len(tracks) - 1)
            track = tracks[track_idx]
            is_drum = is_drum_track(track)

            controls = sample_controls(active_controls, is_drum, rng)

            status_tracks = []
            for ti, t in enumerate(tracks):
                if ti == track_idx:
                    st = build_status_track(
                        ti, t, controls, num_bars, temperature
                    )
                else:
                    st = {
                        "track_id": ti,
                        "track_type": 2 if is_drum_track(t) else 10,
                        "ignore": False,
                        "selected_bars": [False] * count_bars(t),
                        "autoregressive": False,
                        "temperature": temperature,
                    }
                status_tracks.append(st)

            try:
                output_piece, elapsed = generate(
                    piece_dict, status_tracks, ckpt, num_bars,
                    seed=gen_seed, max_steps=max_steps,
                )
            except Exception as e:
                print(f"  config {config_i}: GENERATION FAILED: {e}")
                continue

            timing_records.append(elapsed)

            try:
                output_with_features = extract_features(output_piece)
            except Exception as e:
                print(f"  config {config_i}: FEATURE EXTRACTION FAILED: {e}")
                continue

            actual_vals = get_actual_values(
                output_with_features, track_idx, active_controls, is_drum
            )

            for field_name, req_val in controls.items():
                act_val = actual_vals.get(field_name)
                if act_val is None:
                    continue
                col = collectors.get(field_name)
                if col is None:
                    continue
                # Convert to 0-indexed
                req_0 = req_val - col["lo"]
                act_0 = int(act_val)
                col["requested"].append(req_0)
                col["actual"].append(act_0)

            total_generated += 1
            ctrl_str = " ".join(f"{k}={v}" for k, v in controls.items())
            print(f"  config {config_i}: {ctrl_str}  ({elapsed:.2f}s)")

    # Compute metrics per field
    results = {}
    for field_name, col in collectors.items():
        req = col["requested"]
        act = col["actual"]
        if req:
            results[field_name] = compute_metrics(req, act, col["n_classes"])
            results[field_name]["control"] = col["control_name"]
        else:
            results[field_name] = {"n_samples": 0, "control": col["control_name"]}

    # Speed stats
    speed_stats = {}
    if timing_records:
        speed_stats = {
            "n_generations": len(timing_records),
            "mean_latency_s": statistics.mean(timing_records),
            "median_latency_s": statistics.median(timing_records),
            "min_latency_s": min(timing_records),
            "max_latency_s": max(timing_records),
            "stdev_latency_s": (
                statistics.stdev(timing_records)
                if len(timing_records) > 1 else 0.0
            ),
        }

    return {
        "config": {
            "ckpt": ckpt,
            "midi_dir": midi_dir,
            "encoder_type": encoder_type,
            "n_configs": n_configs,
            "seed": seed,
            "temperature": temperature,
            "max_steps": max_steps,
        },
        "summary": {
            "total_generated": total_generated,
            "total_skipped": total_skipped,
            "total_midi_files": len(midi_files),
        },
        "attribute_metrics": results,
        "speed": speed_stats,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict):
    print("\n" + "=" * 72)
    print("ATTRIBUTE CONTROL EVALUATION RESULTS")
    print(f"Encoder: {results['config']['encoder_type']}")
    print("=" * 72)

    summary = results["summary"]
    print(f"\nFiles: {summary['total_midi_files']}  "
          f"Generated: {summary['total_generated']}  "
          f"Skipped: {summary['total_skipped']}")

    # Group fields by control name for cleaner output
    by_control: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for field_name, metrics in results["attribute_metrics"].items():
        ctrl = metrics.get("control", field_name)
        by_control[ctrl].append((field_name, metrics))

    for ctrl_name, field_metrics in by_control.items():
        print(f"\n{'─' * 72}")
        print(f"  CONTROL: {ctrl_name}")
        print(f"{'─' * 72}")

        for field_name, metrics in field_metrics:
            n = metrics.get("n_samples", 0)
            if n == 0:
                print(f"\n  {field_name}: no samples")
                continue
            print(f"\n  {field_name} (n={n}):")
            print(f"    Accuracy:              {metrics['accuracy']:.3f}")
            print(f"    Macro Precision:       {metrics['macro_precision']:.3f}")
            print(f"    Macro Recall:          {metrics['macro_recall']:.3f}")
            print(f"    Macro F1:              {metrics['macro_f1']:.3f}")
            print(f"    Mean Abs Distance:     {metrics['mean_absolute_distance']:.3f}")
            print(f"    Cross-Entropy:         {metrics['cross_entropy']:.3f}")
            da = metrics.get("directional_accuracy")
            if da is not None and not math.isnan(da):
                print(f"    Directional Accuracy:  {da:.3f}")

            cm = metrics.get("confusion_matrix")
            if cm:
                n_cls = len(cm)
                print(f"\n    Confusion Matrix ({n_cls} classes):")
                header = "        " + "".join(f"{c:>5}" for c in range(n_cls))
                print(header)
                for r_idx, row in enumerate(cm):
                    row_str = "".join(f"{v:>5}" for v in row)
                    print(f"    {r_idx:>3} {row_str}")

    speed = results.get("speed", {})
    if speed:
        print(f"\n{'─' * 72}")
        print(f"  Generation Speed")
        print(f"{'─' * 72}")
        print(f"  N generations:    {speed['n_generations']}")
        print(f"  Mean latency:     {speed['mean_latency_s']:.3f}s")
        print(f"  Median latency:   {speed['median_latency_s']:.3f}s")
        print(f"  Min latency:      {speed['min_latency_s']:.3f}s")
        print(f"  Max latency:      {speed['max_latency_s']:.3f}s")
        print(f"  Stdev:            {speed['stdev_latency_s']:.3f}s")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MIDI-GPT attribute control precision."
    )
    parser.add_argument(
        "--ckpt", required=True,
        help="Path to TorchScript checkpoint (.pt)",
    )
    parser.add_argument(
        "--encoder", required=True,
        help="Encoder type name (e.g. STEINBERG_WPCS_ENCODER, EXPRESSIVE_ENCODER)",
    )
    parser.add_argument(
        "--midi_dir", required=True,
        help="Directory containing test MIDI files",
    )
    parser.add_argument("--n_configs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--output", type=str, default="")

    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    results = run_evaluation(
        ckpt=args.ckpt,
        midi_dir=args.midi_dir,
        encoder_type=args.encoder,
        n_configs=args.n_configs,
        seed=args.seed,
        temperature=args.temperature,
        max_steps=args.max_steps,
    )

    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
