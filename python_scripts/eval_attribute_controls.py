#!/usr/bin/env python3
"""Evaluate how well MIDI-GPT respects attribute controls during generation.

Randomly samples attribute control configurations, runs generation, and
compares requested vs actual control values using precision, recall, F1,
cross-entropy, mean absolute distance, and directional accuracy.

Usage:
    python scripts/eval_attribute_controls.py \
        --ckpt /path/to/model.pt \
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
# Attribute control metadata
# ---------------------------------------------------------------------------

# Status field → (number of classes, proto-enum 1-indexed range, TrackFeatures JSON key)
INSTRUMENT_CONTROLS = {
    "min_polyphony_q": (6, (1, 6), "minPolyphonyQ"),
    "max_polyphony_q": (6, (1, 6), "maxPolyphonyQ"),
    "min_note_duration_q": (6, (1, 6), "minNoteDurationQ"),
    "max_note_duration_q": (6, (1, 6), "maxNoteDurationQ"),
}

DRUM_CONTROLS = {
    "density": (10, (1, 10), "noteDensityLevel"),
}


# ---------------------------------------------------------------------------
# Random config sampling
# ---------------------------------------------------------------------------

def sample_instrument_controls(rng: random.Random) -> dict:
    """Sample random polyphony and duration controls for an instrument track."""
    min_poly = rng.randint(1, 6)
    max_poly = rng.randint(min_poly, 6)
    min_dur = rng.randint(1, 6)
    max_dur = rng.randint(min_dur, 6)
    return {
        "min_polyphony_q": min_poly,
        "max_polyphony_q": max_poly,
        "min_note_duration_q": min_dur,
        "max_note_duration_q": max_dur,
    }


def sample_drum_controls(rng: random.Random) -> dict:
    """Sample random density control for a drum track."""
    return {"density": rng.randint(1, 10)}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

FROZEN_ENCODER_CONFIG = {
    "both_in_one": True,
    "unquantized": False,
    "do_multi_fill": False,
    "use_velocity_levels": True,
    "use_microtiming": True,
    "transpose": 0,
    "resolution": 12,
    "decode_resolution": 1920,
    "decode_final": False,
    "delta_resolution": 1920,
}


def load_piece(midi_path: str) -> dict:
    """Parse a MIDI file into a Piece dict."""
    enc = midigpt.ExpressiveEncoder()
    for k, v in FROZEN_ENCODER_CONFIG.items():
        setattr(enc.config, k, v)
    return json.loads(enc.midi_to_json(midi_path))


def is_drum_track(track: dict) -> bool:
    """Check if a track is a drum track (track_type == 2 in proto enum)."""
    tt = track.get("trackType", track.get("track_type", 0))
    # STANDARD_DRUM_TRACK = 2 in proto
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
    # Select middle bars for generation (leave first/last as context)
    if num_bars <= 2:
        selected = [True] * num_bars
    else:
        selected = [False] * num_bars
        # Generate bars 1..(num_bars-2) as infill
        for i in range(1, num_bars - 1):
            selected[i] = True

    st = {
        "track_id": track_idx,
        "track_type": 2 if is_drum else 10,  # STANDARD_DRUM_TRACK or STANDARD_TRACK
        "temperature": temperature,
        "ignore": False,
        "selected_bars": selected,
        "autoregressive": False,
        "polyphony_hard_limit": 0,
    }
    st.update(controls)

    # Set defaults for controls not relevant to this track type
    if is_drum:
        st.setdefault("min_polyphony_q", "POLYPHONY_ANY")
        st.setdefault("max_polyphony_q", "POLYPHONY_ANY")
        st.setdefault("min_note_duration_q", "DURATION_ANY")
        st.setdefault("max_note_duration_q", "DURATION_ANY")
    else:
        st.setdefault("density", 0)  # DENSITY_ANY

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

    result_json = result[0]
    return json.loads(result_json), elapsed


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(piece_dict: dict) -> dict:
    """Compute attribute controls on a piece and return per-track features."""
    piece_json = json.dumps(piece_dict)
    updated = midigpt.compute_all_attribute_controls(piece_json)
    updated_dict = json.loads(updated)
    return updated_dict


def get_actual_values(piece_dict: dict, track_idx: int) -> dict:
    """Extract computed attribute control values for a track."""
    track = piece_dict["tracks"][track_idx]
    features_list = track.get("internalFeatures", [])
    if not features_list:
        return {}
    feat = features_list[0]
    return {
        "noteDensityLevel": feat.get("noteDensityLevel"),
        "minPolyphonyQ": feat.get("minPolyphonyQ"),
        "maxPolyphonyQ": feat.get("maxPolyphonyQ"),
        "minNoteDurationQ": feat.get("minNoteDurationQ"),
        "maxNoteDurationQ": feat.get("maxNoteDurationQ"),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(requested: list[int], actual: list[int], n_classes: int) -> np.ndarray:
    """Build confusion matrix: rows=requested, cols=actual."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for r, a in zip(requested, actual):
        if 0 <= r < n_classes and 0 <= a < n_classes:
            cm[r, a] += 1
    return cm


def compute_metrics(requested: list[int], actual: list[int], n_classes: int) -> dict:
    """Compute classification and distance metrics."""
    if not requested:
        return {}

    cm = confusion_matrix(requested, actual, n_classes)

    # Exact match accuracy
    correct = sum(1 for r, a in zip(requested, actual) if r == a)
    accuracy = correct / len(requested)

    # Per-class precision, recall, F1
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

    # Macro averages (only over classes that appear in requested)
    active_classes = set(requested)
    active_p = [precisions[c] for c in active_classes if c < n_classes]
    active_r = [recalls[c] for c in active_classes if c < n_classes]
    active_f1 = [f1s[c] for c in active_classes if c < n_classes]

    macro_precision = statistics.mean(active_p) if active_p else 0.0
    macro_recall = statistics.mean(active_r) if active_r else 0.0
    macro_f1 = statistics.mean(active_f1) if active_f1 else 0.0

    # Mean Absolute Distance
    mad = statistics.mean(abs(r - a) for r, a in zip(requested, actual))

    # Cross-entropy: for each sample, CE = -log(P(actual_class | requested_class))
    # where P is estimated from the confusion matrix rows (normalized)
    eps = 1e-10
    ce_values = []
    for r, a in zip(requested, actual):
        row_sum = cm[r, :].sum()
        if row_sum > 0:
            prob = (cm[r, a] + eps) / (row_sum + eps * n_classes)
            ce_values.append(-math.log(prob))
    cross_entropy = statistics.mean(ce_values) if ce_values else float("inf")

    # Directional accuracy: split into low/high halves, check direction
    mid = n_classes / 2
    dir_correct = 0
    dir_total = 0
    for r, a in zip(requested, actual):
        if r < mid - 0.5:  # requested low
            dir_total += 1
            if a <= mid:
                dir_correct += 1
        elif r > mid + 0.5:  # requested high
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
    n_configs: int,
    seed: int,
    temperature: float,
    max_steps: int,
) -> dict:
    rng = random.Random(seed)

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

    # Collectors: control_name → lists of (requested_0indexed, actual_0indexed)
    collectors: dict[str, dict[str, list[int]]] = {}
    for name in list(INSTRUMENT_CONTROLS) + list(DRUM_CONTROLS):
        collectors[name] = {"requested": [], "actual": []}

    timing_records = []
    total_generated = 0
    total_skipped = 0

    for midi_path in midi_files:
        print(f"--- {midi_path.name} ---")
        try:
            piece_dict = load_piece(str(midi_path))
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

            # Pick a random track to control
            track_idx = rng.randint(0, len(tracks) - 1)
            track = tracks[track_idx]
            is_drum = is_drum_track(track)

            controls = (
                sample_drum_controls(rng) if is_drum
                else sample_instrument_controls(rng)
            )

            status_tracks = []
            for ti, t in enumerate(tracks):
                if ti == track_idx:
                    st = build_status_track(
                        ti, t, controls, num_bars, temperature
                    )
                else:
                    # Keep other tracks as context (all bars unselected)
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

            # Compute actual features on output
            try:
                output_with_features = extract_features(output_piece)
            except Exception as e:
                print(f"  config {config_i}: FEATURE EXTRACTION FAILED: {e}")
                continue

            actual_vals = get_actual_values(output_with_features, track_idx)

            # Collect results
            control_map = DRUM_CONTROLS if is_drum else INSTRUMENT_CONTROLS
            for ctrl_name, (n_classes, (lo, hi), feat_key) in control_map.items():
                req_val = controls.get(ctrl_name)
                act_val = actual_vals.get(feat_key)
                if req_val is None or act_val is None:
                    continue
                # Convert to 0-indexed
                req_0 = req_val - lo
                act_0 = int(act_val)
                collectors[ctrl_name]["requested"].append(req_0)
                collectors[ctrl_name]["actual"].append(act_0)

            total_generated += 1
            ctrl_str = " ".join(f"{k}={v}" for k, v in controls.items())
            print(f"  config {config_i}: {ctrl_str}  ({elapsed:.2f}s)")

    # Compute metrics per control
    results = {}
    for ctrl_name in list(INSTRUMENT_CONTROLS) + list(DRUM_CONTROLS):
        all_meta = {**INSTRUMENT_CONTROLS, **DRUM_CONTROLS}
        n_classes = all_meta[ctrl_name][0]
        req = collectors[ctrl_name]["requested"]
        act = collectors[ctrl_name]["actual"]
        if req:
            results[ctrl_name] = compute_metrics(req, act, n_classes)
        else:
            results[ctrl_name] = {"n_samples": 0}

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
    print("=" * 72)

    summary = results["summary"]
    print(f"\nFiles: {summary['total_midi_files']}  "
          f"Generated: {summary['total_generated']}  "
          f"Skipped: {summary['total_skipped']}")

    for ctrl_name, metrics in results["attribute_metrics"].items():
        n = metrics.get("n_samples", 0)
        if n == 0:
            continue
        print(f"\n--- {ctrl_name} (n={n}) ---")
        print(f"  Accuracy:              {metrics['accuracy']:.3f}")
        print(f"  Macro Precision:       {metrics['macro_precision']:.3f}")
        print(f"  Macro Recall:          {metrics['macro_recall']:.3f}")
        print(f"  Macro F1:              {metrics['macro_f1']:.3f}")
        print(f"  Mean Abs Distance:     {metrics['mean_absolute_distance']:.3f}")
        print(f"  Cross-Entropy:         {metrics['cross_entropy']:.3f}")
        da = metrics.get("directional_accuracy")
        if da is not None and not math.isnan(da):
            print(f"  Directional Accuracy:  {da:.3f}")

        cm = metrics.get("confusion_matrix")
        if cm:
            n_cls = len(cm)
            print(f"\n  Confusion Matrix (rows=requested, cols=actual, {n_cls} classes):")
            header = "      " + "".join(f"{c:>5}" for c in range(n_cls))
            print(header)
            for r_idx, row in enumerate(cm):
                row_str = "".join(f"{v:>5}" for v in row)
                print(f"  {r_idx:>3} {row_str}")

    speed = results.get("speed", {})
    if speed:
        print(f"\n--- Generation Speed ---")
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
    parser.add_argument("--ckpt", required=True, help="Path to TorchScript checkpoint (.pt)")
    parser.add_argument("--midi_dir", required=True, help="Directory containing test MIDI files")
    parser.add_argument("--n_configs", type=int, default=10, help="Random configs per MIDI file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--max_steps", type=int, default=0, help="Max tokens per step (0=unlimited)")
    parser.add_argument("--output", type=str, default="", help="Output JSON path (default: print only)")

    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    results = run_evaluation(
        ckpt=args.ckpt,
        midi_dir=args.midi_dir,
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
