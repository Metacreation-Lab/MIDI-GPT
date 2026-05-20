#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <array>
#include <iomanip>

#include "../../common/midi_parsing/util_protobuf.h"

// ================================================
// SPIRAL ARRAY TONAL TENSION (Chew 2002)
// Pitch positions follow Chew's original formulation:
//   - Circle-of-fifths ordering (PC_TO_SPIRAL) not chromatic
//   - Key position computed from weighted triads (SA_W)
//   - verticalStep = 0.4, radius = 1.0
//   - Minor key position uses alpha=beta=0.75 blending (matches Python)
// ================================================
namespace spiral_array {

using Point3D = std::array<double, 3>;

// Pitch class (0=C..11=B) → spiral array index (circle of fifths)
static const int PC_TO_SPIRAL[12] = {0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5};

// Reverse: (spiral_idx + 6) % 12 → chromatic pitch class
// spiral -6..5 → index 0..11 (offset by 6)
static const int SPIRAL_TO_PC[12] = {6, 1, 8, 3, 10, 5, 0, 7, 2, 9, 4, 11};

// Map spiral index to chromatic pitch class (handles any integer, wraps mod 12)
inline int spiral_to_chromatic(int s) {
    int idx = ((s + 6) % 12 + 12) % 12;
    return SPIRAL_TO_PC[idx];
}

static constexpr double SA_R = 1.0;
static constexpr double SA_V = 0.4;
static const double SA_W[3] = {0.536, 0.274, 0.190};
static constexpr double SA_ALPHA = 0.75;  // minor key fifth blending
static constexpr double SA_BETA  = 0.75;  // minor key fourth blending

inline double sa_dist(const Point3D& a, const Point3D& b) {
    return std::sqrt(std::pow(a[0]-b[0], 2) + std::pow(a[1]-b[1], 2) + std::pow(a[2]-b[2], 2));
}

// Position of a note in the spiral array given its spiral index.
// c = idx mod 4 (corrected for negative), axis assignment follows pitch_index_to_position().
inline Point3D sa_pos(int spiral_idx) {
    int c = spiral_idx % 4;
    if (c < 0) c += 4;
    Point3D p = {0.0, 0.0, spiral_idx * SA_V};
    if      (c == 0) p[1] =  SA_R;
    else if (c == 1) p[0] =  SA_R;
    else if (c == 2) p[1] = -SA_R;
    else             p[0] = -SA_R;
    return p;
}

// Weighted centroid of the major triad rooted at spiral index `root`.
inline Point3D major_triad_pos(int root) {
    auto r = sa_pos(root);
    auto f = sa_pos(root + 1);   // perfect fifth
    auto t = sa_pos(root + 4);   // major third
    return {SA_W[0]*r[0] + SA_W[1]*f[0] + SA_W[2]*t[0],
            SA_W[0]*r[1] + SA_W[1]*f[1] + SA_W[2]*t[1],
            SA_W[0]*r[2] + SA_W[1]*f[2] + SA_W[2]*t[2]};
}

// Weighted centroid of the minor triad rooted at spiral index `root`.
// Minor third = root - 3 in spiral index (matches Python minor_triad_position).
inline Point3D minor_triad_pos(int root) {
    auto r = sa_pos(root);
    auto f = sa_pos(root + 1);   // perfect fifth
    auto t = sa_pos(root - 3);   // minor third
    return {SA_W[0]*r[0] + SA_W[1]*f[0] + SA_W[2]*t[0],
            SA_W[0]*r[1] + SA_W[1]*f[1] + SA_W[2]*t[1],
            SA_W[0]*r[2] + SA_W[1]*f[2] + SA_W[2]*t[2]};
}

// Key position for the major key rooted at spiral index `root`.
inline Point3D major_key_pos(int root) {
    auto tr = major_triad_pos(root);       // tonic triad
    auto tf = major_triad_pos(root + 1);   // dominant triad
    auto ts = major_triad_pos(root - 1);   // subdominant triad
    return {SA_W[0]*tr[0] + SA_W[1]*tf[0] + SA_W[2]*ts[0],
            SA_W[0]*tr[1] + SA_W[1]*tf[1] + SA_W[2]*ts[1],
            SA_W[0]*tr[2] + SA_W[1]*tf[2] + SA_W[2]*ts[2]};
}

// Key position for the minor key rooted at spiral index `root`.
// Matches Python minor_key_position() with alpha=beta=0.75.
inline Point3D minor_key_pos(int root) {
    auto root_t  = minor_triad_pos(root);
    auto maj_f   = major_triad_pos(root + 1);
    auto min_f   = minor_triad_pos(root + 1);
    auto min_s   = minor_triad_pos(root - 1);
    auto maj_s   = major_triad_pos(root - 1);
    // fifth blend: alpha*major + (1-alpha)*minor
    Point3D f_bl = {SA_ALPHA*maj_f[0]+(1-SA_ALPHA)*min_f[0],
                    SA_ALPHA*maj_f[1]+(1-SA_ALPHA)*min_f[1],
                    SA_ALPHA*maj_f[2]+(1-SA_ALPHA)*min_f[2]};
    // fourth blend: beta*minor + (1-beta)*major
    Point3D s_bl = {SA_BETA*min_s[0]+(1-SA_BETA)*maj_s[0],
                    SA_BETA*min_s[1]+(1-SA_BETA)*maj_s[1],
                    SA_BETA*min_s[2]+(1-SA_BETA)*maj_s[2]};
    return {SA_W[0]*root_t[0] + SA_W[1]*f_bl[0] + SA_W[2]*s_bl[0],
            SA_W[0]*root_t[1] + SA_W[1]*f_bl[1] + SA_W[2]*s_bl[1],
            SA_W[0]*root_t[2] + SA_W[1]*f_bl[2] + SA_W[2]*s_bl[2]};
}

// Precomputed key positions (root spiral index 0 = C).
inline const Point3D& c_major_key_pos() {
    static const Point3D pos = major_key_pos(0);
    return pos;
}
// A minor (spiral index 3) is the reference for all minor keys, matching Python minor_key_position(3).
inline const Point3D& a_minor_key_pos() {
    static const Point3D pos = minor_key_pos(3);
    return pos;
}

// Per-track spiral-array key detection: tries all 24 keys (12 major + 12 minor),
// returns chromatic pitch class of the detected key root (0=C..11=B).
// Matches Python cal_key() with all_key_names.
//
// For major key with chromatic root r:
//   shift = r, compare centroid-shifted-by-r to c_major_key_pos().
// For minor key with chromatic root r_minor:
//   spiral index of minor root: s_minor = PC_TO_SPIRAL[r_minor]
//   shift = chromatic_pc of (s_minor - 3) [maps relative major root to C]
//   compare centroid-shifted-by-shift to a_minor_key_pos().
inline int detect_key_shift_spiral(const midi::Piece* x, int track_num,
                                    const std::vector<int>& bar_starts,
                                    bool verbose = false,
                                    bool* is_minor_out = nullptr) {
    // Build a binary piano roll (128 pitches × T steps) exactly matching Python's
    // get_piano_roll: for each pitch, mark which 16th-note grid steps are active.
    // roll[p][s] = true means pitch p is active at step s (no double-counting for
    // same-pitch overlapping notes, matching Python's roll[p, i0:i1] = 1 semantics).
    const auto& track = x->tracks(track_num);
    int res  = x->has_resolution() ? x->resolution() : 12;
    int step = std::max(1, res / 4);  // sixteenth-note step (beat_division=4)

    // First pass: compute max step to size the roll
    int max_step = 0;
    for (int b = 0; b < track.bars_size(); b++) {
        for (int eid : track.bars(b).events()) {
            const auto& ev = x->events(eid);
            if (ev.velocity() > 0) {
                int dur = ev.has_internal_duration() ? ev.internal_duration() : res;
                int abs_end = bar_starts[b] + ev.time() + std::max(1, dur);
                int i1 = (abs_end - 1) / step + 1;
                if (i1 > max_step) max_step = i1;
            }
        }
    }
    if (max_step <= 0) return 0;

    // Build piano roll: roll[pitch * max_step + s] = true if active
    std::vector<bool> piano_roll(128 * max_step, false);
    for (int b = 0; b < track.bars_size(); b++) {
        for (int eid : track.bars(b).events()) {
            const auto& ev = x->events(eid);
            if (ev.velocity() > 0) {
                int dur = ev.has_internal_duration() ? ev.internal_duration() : res;
                int abs_start = bar_starts[b] + ev.time();
                int abs_end   = abs_start + std::max(1, dur);
                int i0 = (abs_start + step - 1) / step;  // ceil — searchsorted "left"
                int i1 = (abs_end   - 1)        / step + 1;
                if (i1 <= i0) continue;  // matches Python: skip zero-step notes
                int pitch = ev.pitch() & 127;
                int base  = pitch * max_step;
                for (int s = i0; s < i1 && s < max_step; s++)
                    piano_roll[base + s] = true;
            }
        }
    }

    // Count active steps per pitch class: matches Python's piano_roll_to_ce summing
    double pc_counts[12] = {};
    for (int pitch = 0; pitch < 128; pitch++) {
        int base = pitch * max_step;
        for (int s = 0; s < max_step; s++)
            if (piano_roll[base + s]) pc_counts[pitch % 12] += 1.0;
    }

    if (verbose) {
        static const char* PC_NAMES[12] = {"C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"};
        std::cerr << std::fixed << std::setprecision(1)
                  << "[CPP track=" << track_num << "] KEY_PC_COUNTS:";
        for (int i = 0; i < 12; i++)
            std::cerr << " " << PC_NAMES[i] << "=" << pc_counts[i];
        std::cerr << "\n";
    }

    const auto& maj_kp  = c_major_key_pos();
    const auto& min_kp  = a_minor_key_pos();
    double best_dist  = 1e18;
    int    best_shift = 0;  // the actual key_shift_for_ce to return
    bool   best_is_minor = false;

    // Helper: compute centroid shifted by shift_chrom, measure distance to ref_kp
    auto try_key = [&](int shift_chrom, const Point3D& ref_kp,
                       const char* key_name_dbg = nullptr) -> double {
        Point3D ce = {0.0, 0.0, 0.0};
        double total = 0.0;
        for (int pc = 0; pc < 12; pc++) {
            if (pc_counts[pc] == 0.0) continue;
            int shifted_pc = (pc - shift_chrom + 12) % 12;
            auto pos = sa_pos(PC_TO_SPIRAL[shifted_pc]);
            ce[0] += pc_counts[pc] * pos[0];
            ce[1] += pc_counts[pc] * pos[1];
            ce[2] += pc_counts[pc] * pos[2];
            total  += pc_counts[pc];
        }
        if (total > 0.0) { ce[0]/=total; ce[1]/=total; ce[2]/=total; }
        double dist = sa_dist(ce, ref_kp);
        if (verbose && key_name_dbg)
            std::cerr << std::fixed << std::setprecision(6)
                      << "[CPP track=" << track_num << "] KEY_CANDIDATE key=" << key_name_dbg
                      << " shift=" << shift_chrom
                      << " ce=(" << ce[0] << "," << ce[1] << "," << ce[2] << ")"
                      << " dist=" << dist << "\n";
        return dist;
    };

    // Python all_key_names order:
    // Major: C G D A E B F B- E- A- D- G-
    // Minor: A E B F# C# G# D G C F B- E-
    static const char* MAJOR_KEY_NAMES[12] = {
        "C major","G major","D major","A major","E major","B major",
        "F major","B- major","E- major","A- major","D- major","G- major"
    };
    static const char* MINOR_KEY_NAMES[12] = {
        "A minor","E minor","B minor","F# minor","C# minor","G# minor",
        "D minor","G minor","C minor","F minor","B- minor","E- minor"
    };

    for (int r = 0; r < 12; r++) {
        // Major key with chromatic root r: shift = r (rotate so root maps to C)
        double d_maj = try_key(r, maj_kp, verbose ? MAJOR_KEY_NAMES[r] : nullptr);
        if (d_maj < best_dist) { best_dist = d_maj; best_shift = r; best_is_minor = false; }

        // Minor key with chromatic root r:
        // Python: key_index = pitch_name_to_pitch_index[root] = PC_TO_SPIRAL[r]
        //         key_index -= 3  (maps to relative major spiral index)
        //         shift_for_ce = chromatic_pc_of(key_index)
        // The shift is the chromatic PC of the relative major root.
        int s_minor  = PC_TO_SPIRAL[r];              // spiral index of minor root
        int s_rel    = s_minor - 3;                  // relative major spiral index
        int shift_m  = spiral_to_chromatic(s_rel);   // chromatic PC of relative major
        double d_min = try_key(shift_m, min_kp, verbose ? MINOR_KEY_NAMES[r] : nullptr);
        if (d_min < best_dist) { best_dist = d_min; best_shift = shift_m; best_is_minor = true; }
    }

    if (verbose)
        std::cerr << "[CPP track=" << track_num << "] KEY_BEST_SHIFT=" << best_shift
                  << " best_dist=" << best_dist
                  << " is_minor=" << (best_is_minor ? "true" : "false") << "\n";

    if (is_minor_out) *is_minor_out = best_is_minor;
    return best_shift;
}

} // namespace spiral_array

// ================================================
// FARBOOD TREND-SALIENCE TENSION MODEL
// ================================================
namespace farbood {

static constexpr int SAMPLE_RATE = 10;
static constexpr double MEMORY_WINDOW_DUR = 3.0;
static constexpr double ATTENTIONAL_WINDOW_DUR = 3.0;
static constexpr double WINDOW_SHIFT = 0.25;

// Python-compatible fill boundary: first sample s such that s*10000 >= int(t*100000).
// Matches Python: int(sampleIndex / sampleRate * 100000) < int(t * 100000)
// where sampleRate=10, so the condition is sampleIndex*10000 < int(t*100000).
// We stop when sampleIndex*10000 >= floor(t*100000), i.e. s1 = ceil(floor(t*100000)/10000).
inline int py_fill_s1(double t) {
    long long n = (long long)(t * 100000.0);   // Python int() = truncation toward zero
    return (int)((n + 9999LL) / 10000LL);       // ceil(n / 10000)
}
static constexpr double MEMORY_WEIGHT = 2.0;
static constexpr double EPSILON = 0.0001;
static constexpr double DECAY = 0.001;

static const double WEIGHTS_PITCHED[6] = {2.0, 3.0, 3.0, 2.0, 1.0, 1.0};
static const double WEIGHTS_DRUMS[6]   = {2.0, 0.0, 3.0, 2.0, 0.0, 0.0};

static const double INTERVAL_DISSONANCE[12] = {
    0.0, 0.85, 0.4, 0.255, 0.225, 0.15, 0.275, 0.075, 0.275, 0.175, 0.225, 0.4
};

// Tempo map: mirrors Python _get_tempo_map / tick_to_time in tension_calculation.py.
// Ticks are in SPQ (piece resolution) units, matching bar_start_ticks().
struct TempoMap {
    std::vector<int>    ticks;  // sorted ascending
    std::vector<double> secs;   // cumulative seconds at each tempo change
    std::vector<double> qpm;    // BPM at each segment
    int                 res;    // piece resolution (SPQ)
};

// Build a TempoMap from a Piece. Falls back to the single `tempo` field (or 120 BPM)
// when tempo_changes is empty (e.g. pieces loaded from dataset, not parsed from MIDI).
inline TempoMap build_tempo_map(const midi::Piece* x) {
    TempoMap tm;
    tm.res = x->has_resolution() ? x->resolution() : 12;

    if (x->tempo_changes_size() > 0) {
        tm.ticks.reserve(x->tempo_changes_size());
        tm.qpm.reserve(x->tempo_changes_size());
        for (int i = 0; i < x->tempo_changes_size(); i++) {
            tm.ticks.push_back(x->tempo_changes(i).tick());
            tm.qpm.push_back((double)x->tempo_changes(i).qpm());
        }
        // Sort by tick (tempo events in the proto should already be sorted, but be safe)
        std::vector<int> idx(tm.ticks.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return tm.ticks[a] < tm.ticks[b]; });
        std::vector<int>    sorted_t; sorted_t.reserve(tm.ticks.size());
        std::vector<double> sorted_q; sorted_q.reserve(tm.qpm.size());
        for (int i : idx) { sorted_t.push_back(tm.ticks[i]); sorted_q.push_back(tm.qpm[i]); }
        tm.ticks = std::move(sorted_t);
        tm.qpm   = std::move(sorted_q);
    } else {
        double fallback = x->has_tempo() ? (double)x->tempo() : 120.0;
        tm.ticks = {0};
        tm.qpm   = {fallback};
    }

    // Build cumulative seconds at each tempo-change boundary
    tm.secs.resize(tm.ticks.size(), 0.0);
    for (int i = 1; i < (int)tm.ticks.size(); i++) {
        int dticks = tm.ticks[i] - tm.ticks[i-1];
        double sec_per_tick = 60.0 / (tm.qpm[i-1] * tm.res);
        tm.secs[i] = tm.secs[i-1] + dticks * sec_per_tick;
    }
    return tm;
}

// Convert absolute SPQ tick to seconds using the tempo map.
// Mirrors Python tick_to_time() with binary search over tempo segments.
inline double tick_to_sec_map(const TempoMap& tm, int tick) {
    // Find last tempo change <= tick
    auto it  = std::upper_bound(tm.ticks.begin(), tm.ticks.end(), tick);
    int  idx = (int)std::distance(tm.ticks.begin(), it) - 1;
    if (idx < 0) idx = 0;
    double sec_per_tick = 60.0 / (tm.qpm[idx] * tm.res);
    return tm.secs[idx] + (tick - tm.ticks[idx]) * sec_per_tick;
}

// Helper to map ticks to seconds using global tempo and resolution (legacy, single-tempo).
// Kept for any callers that don't yet have a TempoMap.
inline double tick_to_sec(const midi::Piece* x, int tick) {
    double bpm = x->has_tempo() ? x->tempo() : 120.0;
    double res = x->has_resolution() ? x->resolution() : 12.0;
    return (double)tick * 60.0 / (bpm * res);
}

inline double polyfit_slope(const std::vector<double>& y, int start, int end) {
    int n = end - start;
    if (n < 2) return 0.0;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < n; i++) {
        double xi = i;
        double yi = y[start + i];
        sum_x += xi; sum_y += yi; sum_xy += xi * yi; sum_x2 += xi * xi;
    }
    double denom = (n * sum_x2 - sum_x * sum_x);
    if (std::abs(denom) < 1e-12) return 0.0;
    return (n * sum_xy - sum_x * sum_y) / denom;
}

inline void z_score_normalize(std::vector<double>& v) {
    if (v.empty()) return;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - mean) * (x - mean);
    double std = std::sqrt(sq_sum / std::max(1.0, (double)v.size() - 1.0));
    if (std < 1e-12) std::fill(v.begin(), v.end(), 0.0);
    else for (double& x : v) x = (x - mean) / std;
}

// Z-score normalize with ddof=0, matching Python dataProcessing.normalize()
inline void normalize_feature(std::vector<double>& v) {
    if (v.empty()) return;
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - mean) * (x - mean);
    double std = std::sqrt(sq_sum / std::max(1.0, (double)v.size())); // ddof=0
    if (std < 1e-12) std::fill(v.begin(), v.end(), 0.0);
    else for (double& x : v) x = (x - mean) / std;
}

// Returns the bar duration in internal ticks for bar b (= internalBeatLength * resolution).
inline int bar_duration_ticks(const midi::Piece* x, int track_num, int bar_num) {
    double beat_len = x->tracks(track_num).bars(bar_num).has_internal_beat_length()
                      ? x->tracks(track_num).bars(bar_num).internal_beat_length() : 4.0;
    int res = x->has_resolution() ? x->resolution() : 12;
    return (int)std::round(beat_len * res);
}

// Cumulative bar start ticks (absolute) for each bar in the track.
inline std::vector<int> bar_start_ticks(const midi::Piece* x, int track_num) {
    const auto& track = x->tracks(track_num);
    std::vector<int> starts(track.bars_size(), 0);
    for (int b = 1; b < track.bars_size(); b++)
        starts[b] = starts[b-1] + bar_duration_ticks(x, track_num, b-1);
    return starts;
}

// Beat-granular harmony: matches Python _harmonic_tension_per_track_symusic(window_size=2).
//
// Algorithm (mirrors cal_tension / cal_centroid / merge_tension in tension_calculation.py):
//   1. Build piano roll at sixteenth-note granularity (step = res/4 ticks).
//      Notes are active for every step they sustain (not just onsets).
//   2. Compute centroid at each step (average spiral-array position of active notes; {0,0,0} if silent).
//   3. Merge centroids over 2-beat windows (average of all steps in [beat_i, beat_{i+2})).
//      Range: i = 0, 2, 4, ..., n_beats-2  (Python: range(0, len(beat_indices)-window_size, window_size)).
//   4. Compute distance from merged centroid to key position (0 if centroid norm < 0.1 = silent).
//   5. Fill 10 Hz output via step-hold starting at the window's beat time.
//
// Writes results into harmony_out[0..n_samples).
inline void fill_harmony_beat_granular(
    const midi::Piece* x, int track_num, int key_shift, bool is_minor,
    int n_samples, const std::vector<int>& bar_starts,
    const TempoMap& tm,
    std::vector<double>& harmony_out,
    int note_end_max_tick = 0,  // from extract_features_10hz; used for n_beats (window count)
    bool verbose = false)
{
    const auto& track = x->tracks(track_num);
    int res = x->has_resolution() ? x->resolution() : 12;
    int last_bar = track.bars_size() - 1;
    int max_tick = bar_starts[last_bar] + bar_duration_ticks(x, track_num, last_bar);
    // Use note-end max_tick for beat counting (matches Python which counts beats up to last note)
    // Fall back to bar-boundary max_tick if note_end_max_tick is not provided.
    int beats_max_tick = (note_end_max_tick > 0) ? note_end_max_tick : max_tick;

    // ── Step 1: build note list with durations ───────────────────────────────
    struct Note { int start, end, pitch; };
    std::vector<Note> notes;
    notes.reserve(256);
    for (int b = 0; b < track.bars_size(); b++) {
        for (int eid : track.bars(b).events()) {
            const auto& ev = x->events(eid);
            if (ev.velocity() == 0) continue;
            int abs_start = bar_starts[b] + ev.time();
            int dur = ev.has_internal_duration() ? ev.internal_duration() : res;
            notes.push_back({abs_start, abs_start + std::max(1, dur), ev.pitch()});
        }
    }
    if (notes.empty()) return;

    if (verbose) {
        for (const auto& n : notes) {
            double t_start = tick_to_sec_map(tm, n.start);
            double t_end   = tick_to_sec_map(tm, n.end);
            std::cerr << std::fixed << std::setprecision(6)
                      << "[CPP track=" << track_num << "] HARM_NOTE"
                      << " start_tick=" << n.start << " end_tick=" << n.end
                      << " t_start=" << t_start << " t_end=" << t_end
                      << " pitch=" << n.pitch << "\n";
        }
    }

    // ── Step 2: sixteenth-note step grid ─────────────────────────────────────
    // Python beat_division=4: 4 steps per beat.
    int step          = std::max(1, res / 4);          // ticks per sixteenth note
    int steps_per_beat = res / step;                   // = 4 for typical resolutions
    int n_steps       = (max_tick + step - 1) / step;  // total sixteenth steps

    // Build binary piano roll (128 x n_steps) — OR semantics per pitch, matching
    // Python get_piano_roll which does roll[p, i0:i1] = 1 (not +=1).
    // This prevents double-counting when two notes of the same pitch overlap.
    // i0 = ceil(start / step), i1 = floor((end-1) / step) + 1 = ceil(end / step)
    // which matches Python searchsorted(..., "left") on a uniform tick grid.
    std::vector<bool> piano_roll_bits((size_t)128 * n_steps, false);
    for (const auto& note : notes) {
        int pitch = note.pitch & 127;
        int i0 = std::max(0, (note.start + step - 1) / step);
        int i1 = std::min(n_steps, (note.end - 1) / step + 1);
        for (int i = i0; i < i1; i++)
            piano_roll_bits[(size_t)pitch * n_steps + i] = true;
    }

    // Compute per-step centroid from binary piano roll
    std::vector<double> ce_x(n_steps, 0.0), ce_y(n_steps, 0.0), ce_z(n_steps, 0.0);
    std::vector<int>    ce_n(n_steps, 0);
    for (int pitch = 0; pitch < 128; pitch++) {
        int pc = ((pitch % 12) - key_shift + 12) % 12;
        auto pos = spiral_array::sa_pos(spiral_array::PC_TO_SPIRAL[pc]);
        size_t base = (size_t)pitch * n_steps;
        for (int i = 0; i < n_steps; i++) {
            if (piano_roll_bits[base + i]) {
                ce_x[i] += pos[0]; ce_y[i] += pos[1]; ce_z[i] += pos[2];
                ce_n[i]++;
            }
        }
    }
    // Normalise per-step centroids (silent steps remain {0,0,0})
    for (int i = 0; i < n_steps; i++) {
        if (ce_n[i] > 0) {
            ce_x[i] /= ce_n[i];  ce_y[i] /= ce_n[i];  ce_z[i] /= ce_n[i];
        }
    }

    // ── Step 3 & 4: 2-beat window merge + key distance ───────────────────────
    // Python: range(0, len(beat_indices) - window_size, window_size)
    //       = range(0, n_beats - window_size, window_size)
    //       = range(0, n_beats - 2, 2)
    // Python get_beat_time uses range(0, end_tick+1, beat_ticks) = floor(end/beat)+1 beats.
    // C++ must add +1 to match: beats are indexed 0..floor(end/beat) inclusive.
    int n_beats = beats_max_tick / res + 1;  // matches Python len(beat_indices)
    constexpr int WIN_BEATS = 2;           // window_size in Python

    // Use C major (spiral 0) for major keys and A minor (spiral 3) for minor keys.
    // Matches Python cal_tension: key_pos = major_key_position(0) or minor_key_position(3).
    const auto& kp = is_minor ? spiral_array::a_minor_key_pos()
                               : spiral_array::c_major_key_pos();

    struct WinVal { int start_beat; double tension; };
    std::vector<WinVal> windows;
    windows.reserve((n_beats + 1) / WIN_BEATS + 1);

    for (int bi = 0; bi < n_beats - WIN_BEATS; bi += WIN_BEATS) {
        int s0 = bi * steps_per_beat;
        int s1 = std::min(n_steps, (bi + WIN_BEATS) * steps_per_beat);

        // Average per-step centroids over the window (include silent/zero steps)
        double ax = 0, ay = 0, az = 0;
        int cnt = s1 - s0;
        for (int s = s0; s < s1; s++) { ax += ce_x[s]; ay += ce_y[s]; az += ce_z[s]; }
        if (cnt > 0) { ax /= cnt; ay /= cnt; az /= cnt; }

        double norm = std::sqrt(ax*ax + ay*ay + az*az);
        double val  = 0.0;
        if (norm >= 0.1) {
            spiral_array::Point3D avg = {ax, ay, az};
            val = spiral_array::sa_dist(avg, kp);
        }
        if (verbose) {
            double beat_sec = tick_to_sec_map(tm, bi * res);
            std::cerr << std::fixed << std::setprecision(6)
                      << "[CPP track=" << track_num << "] HARM_WIN bi=" << bi
                      << " beat_sec=" << beat_sec
                      << " centroid=(" << ax << "," << ay << "," << az << ")"
                      << " norm=" << norm << " tension=" << val << "\n";
        }
        windows.push_back({bi, val});
    }

    if (windows.empty()) return;

    // ── Step 5: fill 10 Hz output via step-hold (Python-compatible cumulative) ──
    // Python featureAnalysis.py uses cumulative sampleIndex and py_fill_s1 boundaries.
    {
        int cumSI = 0;
        for (size_t wi = 0; wi + 1 < windows.size(); wi++) {
            double t_next = tick_to_sec_map(tm, windows[wi + 1].start_beat * res);
            int s1 = py_fill_s1(t_next);
            for (int s = cumSI; s < s1 && s < n_samples; s++) harmony_out[s] = windows[wi].tension;
            cumSI = std::min(s1, n_samples);
        }
        // Tail: last window holds to end
        for (int s = cumSI; s < n_samples; s++) harmony_out[s] = windows.back().tension;
    }
}

inline std::vector<std::vector<double>> extract_features_10hz(const midi::Piece* x, int track_num, bool is_drum) {
    const auto& track = x->tracks(track_num);
    if (track.bars_size() == 0) return std::vector<std::vector<double>>(6);

    // Build absolute bar start ticks — event.time is bar-relative, not absolute
    auto bar_starts = bar_start_ticks(x, track_num);

    // Compute total duration from note end times only (mirrors Python featureAnalysis
    // which uses max note end time, NOT bar boundaries).
    // Fall back to last bar boundary only if no notes exist.
    int last_bar = track.bars_size() - 1;
    int max_tick = 0;
    int res = x->has_resolution() ? x->resolution() : 12;
    for (int b = 0; b < track.bars_size(); b++) {
        for (int eid : track.bars(b).events()) {
            if (x->events(eid).velocity() == 0) continue;
            int dur = x->events(eid).has_internal_duration()
                      ? x->events(eid).internal_duration() : res;
            max_tick = std::max(max_tick,
                                bar_starts[b] + x->events(eid).time() + std::max(1, dur));
        }
    }
    if (max_tick == 0)
        max_tick = bar_starts[last_bar] + bar_duration_ticks(x, track_num, last_bar);
    // Build tempo map once — used for all tick→second conversions below.
    // Mirrors Python tick_to_time() which reads the full symusic tempo map.
    TempoMap tm = build_tempo_map(x);

    double end_sec = tick_to_sec_map(tm, max_tick);
    // Use floor (same as Python int() cast) to avoid floating-point ceil rounding up
    // e.g. 9.6 sec * 10 Hz = 96.0000...05 → ceil gives 97, int gives 96.
    int n_samples = (int)(end_sec * SAMPLE_RATE);
    if (n_samples <= 0) n_samples = 1;
    std::vector<std::vector<double>> features(6, std::vector<double>(n_samples, 0.0));

    // Check verbose logging flag
    const bool verbose = [] {
        const char* v = std::getenv("TENSION_VERBOSE");
        return v && v[0] && v[0] != '0';
    }();

    // Build onset map keyed by absolute seconds (note-on events only).
    // Round to microseconds (1e-6 s) to match Python: k_us = int(round(sec * 1e6)) / 1e6
    std::map<double, std::vector<int>> onsets;
    for (int b = 0; b < track.bars_size(); b++) {
        for (int eid : track.bars(b).events()) {
            if (x->events(eid).velocity() == 0) continue;
            int abs_tick = bar_starts[b] + x->events(eid).time();
            double sec = tick_to_sec_map(tm, abs_tick);
            sec = std::round(sec * 1e6) / 1e6;   // match Python microsecond rounding
            onsets[sec].push_back(eid);
        }
    }
    if (onsets.empty()) return features;

    if (verbose) {
        std::cerr << "[CPP track=" << track_num << "] n_samples=" << n_samples
                  << " n_onsets=" << onsets.size() << "\n";
        for (auto const& [t, eids] : onsets) {
            std::cerr << std::fixed << std::setprecision(6)
                      << "[CPP track=" << track_num << "] ONSET t=" << t;
            for (int eid : eids)
                std::cerr << " p=" << x->events(eid).pitch()
                          << " v=" << x->events(eid).velocity();
            std::cerr << "\n";
        }
    }

    // ── Onset Frequency ──────────────────────────────────────────────────────
    {
        std::vector<double> onset_times;
        for (auto const& [t, _] : onsets) onset_times.push_back(t);
        std::vector<double> onset_freqs(onset_times.size(), 0.0);
        for (size_t i = 1; i < onset_times.size(); i++) {
            double diff = onset_times[i] - onset_times[i-1];
            if (diff > 1e-6) onset_freqs[i] = 1.0 / diff;
        }
        if (verbose) {
            for (size_t i = 0; i < onset_times.size(); i++)
                std::cerr << std::fixed << std::setprecision(6)
                          << "[CPP track=" << track_num << "] ONSET_FREQ i=" << i
                          << " t=" << onset_times[i] << " freq=" << onset_freqs[i]
                          << " (diff=" << (i>0 ? onset_times[i]-onset_times[i-1] : 0.0) << ")\n";
        }
        {
            int cumSI = 0;
            for (size_t i = 1; i < onset_times.size(); i++) {
                int s1 = py_fill_s1(onset_times[i]);
                if (verbose)
                    std::cerr << "[CPP track=" << track_num << "] ONSET_FREQ_FILL i=" << i-1
                              << " s0=" << cumSI << " s1=" << s1 << " val=" << onset_freqs[i-1] << "\n";
                for (int s = cumSI; s < s1 && s < n_samples; s++) features[0][s] = onset_freqs[i-1];
                cumSI = std::min(s1, n_samples);
            }
            if (!onset_times.empty()) {
                double last_f = onset_freqs.back();
                if (verbose)
                    std::cerr << "[CPP track=" << track_num << "] ONSET_FREQ_TAIL last_s=" << cumSI
                              << " val=" << last_f << "\n";
                for (int s = cumSI; s < n_samples; s++) features[0][s] = last_f;
            }
        }
    }

    // ── Melodic Contour ──────────────────────────────────────────────────────
    // Implements Python getMelodicLine() filtering from noteObj.py:
    //   include onset if: (prevEndTime - t < 0.01) OR (prevEndTime > t AND currPitch > prevPitch)
    if (!is_drum) {
        // Build highest-pitch note info per onset: t → (high_pitch, high_end_sec)
        std::map<double, std::pair<int,double>> onset_high_info;
        for (int b = 0; b < track.bars_size(); b++) {
            for (int eid : track.bars(b).events()) {
                if (x->events(eid).velocity() == 0) continue;
                int abs_tick = bar_starts[b] + x->events(eid).time();
                double sec = tick_to_sec_map(tm, abs_tick);
                sec = std::round(sec * 1e6) / 1e6;
                int dur = x->events(eid).has_internal_duration()
                          ? x->events(eid).internal_duration() : res;
                double end_sec = tick_to_sec_map(tm, abs_tick + std::max(1, dur));
                int pitch = x->events(eid).pitch();
                auto it = onset_high_info.find(sec);
                if (it == onset_high_info.end() || pitch > it->second.first)
                    onset_high_info[sec] = {pitch, end_sec};
            }
        }

        // Apply getMelodicLine filtering
        const double MIN_ONSET_DIFF = 0.01;
        int mel_prev_pitch = -1;
        double mel_prev_end = -1.0;
        std::map<double, int> melodic_line;  // filtered: onset_time → pitch

        for (auto const& [t, info] : onset_high_info) {
            int curr_pitch   = info.first;
            double curr_end  = info.second;
            bool include = (mel_prev_end - t < MIN_ONSET_DIFF) ||
                           (mel_prev_end > t && curr_pitch > mel_prev_pitch);
            if (verbose)
                std::cerr << std::fixed << std::setprecision(6)
                          << "[CPP track=" << track_num << "] MELODIC_FILTER t=" << t
                          << " curr_pitch=" << curr_pitch
                          << " curr_end=" << curr_end
                          << " prev_end=" << mel_prev_end
                          << " prev_pitch=" << mel_prev_pitch
                          << " include=" << include << "\n";
            if (include) {
                melodic_line[t] = curr_pitch;
                mel_prev_pitch  = curr_pitch;
                mel_prev_end    = curr_end;
            }
        }

        // Step-fill features[1] from filtered melodic_line (cumulative, Python-compatible)
        int prev_pitch = 0;
        int cumSI_mel = 0;
        for (auto const& [t, pitch] : melodic_line) {
            int s1 = py_fill_s1(t);
            if (verbose)
                std::cerr << std::fixed << std::setprecision(6)
                          << "[CPP track=" << track_num << "] MELODIC t=" << t
                          << " high_pitch=" << pitch << " fill_s0=" << cumSI_mel << " s1=" << s1
                          << " fill_val=" << prev_pitch << "\n";
            for (int s = cumSI_mel; s < s1 && s < n_samples; s++) features[1][s] = prev_pitch;
            prev_pitch = pitch;
            cumSI_mel = std::min(s1, n_samples);
        }
        for (int s = cumSI_mel; s < n_samples; s++) features[1][s] = prev_pitch;
        {
            int first_nonzero = 0;
            while (first_nonzero < n_samples - 1 && features[1][first_nonzero] == 0.0) first_nonzero++;
            if (first_nonzero > 0 && first_nonzero < n_samples - 1) {
                double first_pitch = features[1][first_nonzero];
                if (verbose)
                    std::cerr << "[CPP track=" << track_num << "] MELODIC_LEAD_FILL first_nonzero="
                              << first_nonzero << " pitch=" << first_pitch << "\n";
                for (int s = 0; s < first_nonzero; s++) features[1][s] = first_pitch;
            }
        }
    }

    // ── Loudness ─────────────────────────────────────────────────────────────
    {
        double prev_loud = 0;
        int cumSI_loud = 0;
        for (auto const& [t, eids] : onsets) {
            int max_v = 0; double sum_v = 0;
            for (int eid : eids) {
                int v = x->events(eid).velocity();
                sum_v += 0.1 * v;
                max_v = std::max(max_v, v);
            }
            double loud = max_v + (sum_v - 0.1 * max_v);
            int s1 = py_fill_s1(t);
            if (verbose)
                std::cerr << std::fixed << std::setprecision(6)
                          << "[CPP track=" << track_num << "] LOUDNESS t=" << t
                          << " max_v=" << max_v << " sum_v=" << sum_v
                          << " loud=" << loud << " fill_s0=" << cumSI_loud << " s1=" << s1
                          << " fill_val=" << prev_loud << "\n";
            for (int s = cumSI_loud; s < s1 && s < n_samples; s++) features[2][s] = prev_loud;
            prev_loud = loud;
            cumSI_loud = std::min(s1, n_samples);
        }
        for (int s = cumSI_loud; s < n_samples; s++) features[2][s] = prev_loud;
    }

    // ── Tempo ─────────────────────────────────────────────────────────────────
    {
        int n_segs = (int)tm.ticks.size();
        for (int i = 0; i < n_segs; i++) {
            double t_start = tm.secs[i];
            double t_end   = (i + 1 < n_segs)
                ? tm.secs[i + 1]
                : tick_to_sec_map(tm, max_tick);
            int s0 = (int)std::round(t_start * SAMPLE_RATE);
            int s1 = (int)std::round(t_end   * SAMPLE_RATE);
            if (verbose)
                std::cerr << std::fixed << std::setprecision(6)
                          << "[CPP track=" << track_num << "] TEMPO seg=" << i
                          << " t_start=" << t_start << " t_end=" << t_end
                          << " qpm=" << tm.qpm[i] << " s0=" << s0 << " s1=" << s1 << "\n";
            for (int s = s0; s < s1 && s < n_samples; s++) features[3][s] = tm.qpm[i];
        }
        double t_last = tm.secs.back();
        int last_s = (int)std::round(t_last * SAMPLE_RATE);
        for (int s = last_s; s < n_samples; s++) features[3][s] = tm.qpm.back();
    }

    // ── Harmony (beat-granular piano roll, matching Python window_size=2) ─────
    // ── Dissonance ────────────────────────────────────────────────────────────
    if (!is_drum) {
        bool key_is_minor = false;
        int key_shift = spiral_array::detect_key_shift_spiral(x, track_num, bar_starts, verbose, &key_is_minor);
        if (verbose)
            std::cerr << "[CPP track=" << track_num << "] KEY_SHIFT=" << key_shift
                      << " key_is_minor=" << (key_is_minor ? "true" : "false") << "\n";
        fill_harmony_beat_granular(x, track_num, key_shift, key_is_minor, n_samples, bar_starts, tm, features[4], max_tick, verbose);

        double prev_diss = 0;
        int cumSI_diss = 0;
        for (auto const& [t, eids] : onsets) {
            double d = 0;
            std::vector<int> pitches;
            for (int eid : eids) pitches.push_back(x->events(eid).pitch());
            for (size_t i = 0; i < pitches.size(); i++)
                for (size_t j = i + 1; j < pitches.size(); j++)
                    d += INTERVAL_DISSONANCE[std::abs(pitches[i] - pitches[j]) % 12];
            int s1 = py_fill_s1(t);
            if (verbose) {
                std::cerr << std::fixed << std::setprecision(6)
                          << "[CPP track=" << track_num << "] DISSONANCE t=" << t
                          << " pitches=";
                for (int p : pitches) std::cerr << p << ",";
                std::cerr << " diss=" << d << " fill_s0=" << cumSI_diss << " s1=" << s1
                          << " fill_val=" << prev_diss << "\n";
            }
            for (int s = cumSI_diss; s < s1 && s < n_samples; s++) features[5][s] = prev_diss;
            prev_diss = d;
            cumSI_diss = std::min(s1, n_samples);
        }
        for (int s = cumSI_diss; s < n_samples; s++) features[5][s] = prev_diss;
    }
    if (verbose) {
        static const char* FEAT_NAMES[6] = {"OnsetFreq","MelodicContour","Loudness","Tempo","Harmony","Dissonance"};
        int n_s = (int)features[0].size();
        for (int fi = 0; fi < 6; fi++) {
            std::cerr << "[CPP track=" << track_num << "] FEAT_RAW " << FEAT_NAMES[fi] << " n=" << n_s << " vals:";
            for (int s = 0; s < n_s; s++)
                std::cerr << std::fixed << std::setprecision(6) << " " << features[fi][s];
            std::cerr << "\n";
        }
    }

    for (int i = 0; i < 6; i++) normalize_feature(features[i]);

    if (verbose) {
        static const char* FEAT_NAMES2[6] = {"OnsetFreq","MelodicContour","Loudness","Tempo","Harmony","Dissonance"};
        int n_s = (int)features[0].size();
        for (int fi = 0; fi < 6; fi++) {
            std::cerr << "[CPP track=" << track_num << "] FEAT_NORM " << FEAT_NAMES2[fi] << " n=" << n_s << " vals:";
            for (int s = 0; s < n_s; s++)
                std::cerr << std::fixed << std::setprecision(6) << " " << features[fi][s];
            std::cerr << "\n";
        }
    }

    return features;
}

inline std::vector<double> run_farbood_model(const std::vector<std::vector<double>>& features, bool is_drum,
                                               bool verbose = false, int track_num = -1) {
    if (features.empty() || features[0].empty()) return {};
    int n_points = (int)features[0].size();
    const double* w_vals = is_drum ? WEIGHTS_DRUMS : WEIGHTS_PITCHED;
    double scale_weight = 0.0;
    for (int i = 0; i < 6; i++) scale_weight += std::abs(w_vals[i]);
    std::vector<double> weights(6);
    for (int i = 0; i < 6; i++) weights[i] = w_vals[i] / (scale_weight + 1e-12);
    int samples_attn = (int)std::round(ATTENTIONAL_WINDOW_DUR * SAMPLE_RATE);
    int samples_mem = (int)std::round(MEMORY_WINDOW_DUR * SAMPLE_RATE);
    int shift = (int)std::round(WINDOW_SHIFT * SAMPLE_RATE);
    if (shift <= 0) shift = 1;
    std::vector<double> prediction(n_points, 0.0);
    int written_len = 0; // mirrors Python's growing prediction array size
    // Python sets endReached=True when endpt first hits numPoints-1, then skips
    // all subsequent iterations. Mirror that here.
    bool end_reached = false;
    int win_idx = 0;
    for (int i = -samples_attn + 2; i < n_points; i += shift) {
        int startpt = std::max(0, i); int endpt = std::min(n_points - 1, samples_attn + i - 1);
        if (endpt <= startpt) continue;
        int win_len = endpt - startpt + 1;
        if (!end_reached) {
            double slope_total = 0.0;
            double feat_slopes[6] = {};
            for (int j = 0; j < 6; j++) {
                feat_slopes[j] = polyfit_slope(features[j], startpt, endpt + 1);
                slope_total += weights[j] * feat_slopes[j];
            }
            int mem_end = startpt - 1; int mem_start = std::max(0, mem_end - samples_mem + 1);
            double prev_slope = 0.0;
            bool mem_active = false;
            if (mem_end >= 3) {
                prev_slope = polyfit_slope(prediction, mem_start, mem_end + 1);
                mem_active = true;
                // Match Python condition order: check decay (both near-zero) FIRST,
                // then strengthen (same sign). C++ polyfit roundoff (~1e-17) would
                // otherwise make tiny-positive slopes fire "strengthen" instead of "decay".
                if      (std::abs(slope_total) < EPSILON && std::abs(prev_slope) < EPSILON) slope_total -= DECAY;
                else if ((slope_total > 0 && prev_slope > 0) || (slope_total < 0 && prev_slope < 0)) slope_total *= MEMORY_WEIGHT;
            }
            if (verbose) {
                std::cerr << std::fixed << std::setprecision(6)
                          << "[CPP track=" << track_num << "] FARBOOD win=" << win_idx
                          << " i=" << i << " startpt=" << startpt << " endpt=" << endpt
                          << " feat_slopes:";
                for (int j = 0; j < 6; j++) std::cerr << " " << feat_slopes[j];
                std::cerr << " slope_raw=" << (slope_total / (mem_active ? 1.0 : 1.0))
                          << " mem_active=" << mem_active << " prev_slope=" << prev_slope
                          << " slope_final=" << slope_total << "\n";
            }
            std::vector<double> y(win_len);
            for (int k = 0; k < win_len; k++) y[k] = slope_total * k;
            if (startpt == 0) {
                for (int k = 0; k < win_len; k++) prediction[k] = y[k];
                written_len = win_len;
            } else {
                double original_start_val = prediction[startpt];
                int overlap = std::max(0, std::min(written_len - startpt, win_len));
                std::vector<double> middle(overlap);
                for (int k = 0; k < overlap; k++) middle[k] = (y[k] + prediction[startpt + k]) / 2.0;
                if (overlap > 0) { double offset1 = original_start_val - middle[0]; for (int k = 0; k < overlap; k++) middle[k] += offset1; }
                for (int k = 0; k < overlap; k++) prediction[startpt + k] = middle[k];
                if (win_len > overlap) {
                    double offset2 = overlap > 0 ? middle.back() - y[overlap] : 0.0;
                    for (int k = overlap; k < win_len && (startpt + k) < n_points; k++) prediction[startpt + k] = y[k] + offset2;
                }
                written_len = std::min(n_points, std::max(written_len, startpt + win_len));
            }
        }
        if (endpt == n_points - 1) end_reached = true;
        win_idx++;
    }
    if (verbose) {
        std::cerr << "[CPP track=" << track_num << "] FARBOOD_RAW n=" << n_points << " vals:";
        for (int s = 0; s < n_points; s++)
            std::cerr << std::fixed << std::setprecision(6) << " " << prediction[s];
        std::cerr << "\n";
    }
    z_score_normalize(prediction);
    if (verbose) {
        std::cerr << "[CPP track=" << track_num << "] FARBOOD_NORM n=" << n_points << " vals:";
        for (int s = 0; s < n_points; s++)
            std::cerr << std::fixed << std::setprecision(6) << " " << prediction[s];
        std::cerr << "\n";
    }
    return prediction;
}

inline std::vector<int> tension_to_bins_fixed(const std::vector<double>& raw, int n_bins) {
    if (raw.empty()) return {};
    const double THEORETICAL_MIN = -3.0; const double THEORETICAL_MAX = 3.0;
    std::vector<int> bins(raw.size(), 0);
    for (size_t i = 0; i < raw.size(); i++) {
        double norm = (raw[i] - THEORETICAL_MIN) / (THEORETICAL_MAX - THEORETICAL_MIN);
        bins[i] = std::clamp((int)std::floor(norm * n_bins), 0, n_bins - 1);
    }
    return bins;
}

// Aggregate continuous tension (10 Hz) into per-bar averages using actual
// time-based bar boundaries (mirrors Python interval_level_average).
// s_start = int(bar_start_sec * 10), s_end = int(next_bar_start_sec * 10).
inline std::vector<double> aggregate_to_bars(
    const std::vector<double>& continuous,
    const midi::Piece* x, int track_num,
    bool verbose = false)
{
    const auto& track = x->tracks(track_num);
    int n_bars = track.bars_size();
    std::vector<double> bar_tensions(n_bars, 0.0);
    if (continuous.empty()) return bar_tensions;

    auto bar_starts = bar_start_ticks(x, track_num);
    TempoMap tm = build_tempo_map(x);
    int last_bar = n_bars - 1;
    int max_tick = bar_starts[last_bar] + bar_duration_ticks(x, track_num, last_bar);
    int n_samples = (int)continuous.size();

    if (verbose)
        std::cerr << "[CPP track=" << track_num << "] AGG n_bars=" << n_bars
                  << " n_samples=" << n_samples << " max_tick=" << max_tick << "\n";

    for (int b = 0; b < n_bars; b++) {
        double t_start = tick_to_sec_map(tm, bar_starts[b]);
        double t_end   = (b + 1 < n_bars)
            ? tick_to_sec_map(tm, bar_starts[b + 1])
            : tick_to_sec_map(tm, max_tick);
        // Mirror Python: sample index = int(t * SAMPLE_RATE) (floor)
        int s0 = (int)(t_start * SAMPLE_RATE);
        int s1 = (int)(t_end   * SAMPLE_RATE);
        s1 = std::min(s1, n_samples);
        if (s1 > s0) {
            double sum = 0; for (int s = s0; s < s1; s++) sum += continuous[s];
            bar_tensions[b] = sum / (s1 - s0);
        }
        if (verbose)
            std::cerr << std::fixed << std::setprecision(6)
                      << "[CPP track=" << track_num << "] AGG_BAR b=" << b
                      << " t_start=" << t_start << " t_end=" << t_end
                      << " s0=" << s0 << " s1=" << s1
                      << " mean=" << bar_tensions[b] << "\n";
    }
    return bar_tensions;
}

inline std::vector<double> track_tonal_tension_raw(const midi::Piece* x, int track_num) {
    const bool verbose = [] {
        const char* v = std::getenv("TENSION_VERBOSE");
        return v && v[0] && v[0] != '0';
    }();
    auto features = extract_features_10hz(x, track_num, false);
    auto continuous = run_farbood_model(features, false, verbose, track_num);
    return aggregate_to_bars(continuous, x, track_num, verbose);
}

inline std::vector<double> track_drum_tension_raw(const midi::Piece* x, int track_num) {
    const bool verbose = [] {
        const char* v = std::getenv("TENSION_VERBOSE");
        return v && v[0] && v[0] != '0';
    }();
    auto features = extract_features_10hz(x, track_num, true);
    auto continuous = run_farbood_model(features, true, verbose, track_num);
    return aggregate_to_bars(continuous, x, track_num, verbose);
}

inline void precompute_instrument_tension(midi::Piece* x, int n_bins = 10) {
    for (int t = 0; t < x->tracks_size(); t++) {
        if (x->tracks(t).track_type() == midi::STANDARD_DRUM_TRACK) continue;
        auto raw  = track_tonal_tension_raw(x, t);
        auto bins = tension_to_bins_fixed(raw, n_bins);
        for (int b = 0; b < x->tracks(t).bars_size(); b++) {
            auto bf = util_protobuf::GetBarFeatures(x->mutable_tracks(t), b);
            bf->set_tension(bins[b]);
            bf->set_tension_raw(raw[b]);
        }
    }
}

inline void precompute_drum_tension(midi::Piece* x, int n_bins = 10) {
    for (int t = 0; t < x->tracks_size(); t++) {
        if (x->tracks(t).track_type() != midi::STANDARD_DRUM_TRACK) continue;
        auto raw  = track_drum_tension_raw(x, t);
        auto bins = tension_to_bins_fixed(raw, n_bins);
        for (int b = 0; b < x->tracks(t).bars_size(); b++) {
            auto bf = util_protobuf::GetBarFeatures(x->mutable_tracks(t), b);
            bf->set_tension_drum(bins[b]);
            bf->set_tension_drum_raw(raw[b]);
        }
    }
}

} // namespace farbood
