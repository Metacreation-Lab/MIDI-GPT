#include "midi_reader.h"
#include <symusic.h>
#include <symusic/io/midi.h>
#include <algorithm>
#include <fstream>
#include <map>
#include <cmath> // For round
#include "../core/logging.h" // Include logging (corrected path)

namespace midigpt::io {

Score MidiReader::read(const std::string& path) const {
    std::ifstream file(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    return read_bytes(bytes);
}

Score MidiReader::read_bytes(const std::vector<uint8_t>& bytes) const {
    symusic::Score<symusic::Tick> s;
    try {
        s = symusic::Score<symusic::Tick>::parse<::symusic::DataFormat::MIDI>(bytes);
    } catch (const std::exception& e) {
        LOG_WARNING("Symusic parse failed, attempting with default parameters: " + std::string(e.what()));
        s = symusic::Score<symusic::Tick>::parse<::symusic::DataFormat::MIDI>(bytes); // Removed extra args
    }
    return from_symusic(s);
}

Score MidiReader::from_symusic(const symusic::Score<symusic::Tick>& s) const {
    Score out;
    out.resolution = resolution_; // Target resolution
    out.tempo = s.tempos->empty() ? 500000 : (*s.tempos)[0].mspq; // Use first tempo event for now

    int tpq = s.ticks_per_quarter;
    if (tpq <= 0) {
        LOG_ERROR("Invalid ticks_per_quarter: " + std::to_string(tpq) + ". Defaulting to 480.");
        tpq = 480;
    }

    int max_tick = 0;
    for (const auto& track_ptr : *s.tracks) {
        for (const auto& note : *track_ptr->notes) {
            max_tick = std::max(max_tick, (int)(note.time + note.duration));
        }
    }

    // Accumulate time signatures and bar lengths
    std::map<int, std::tuple<int, int, int>> timesigs; // time -> (length, numerator, denominator)
    for (const auto& ts : *s.time_signatures) {
        // Calculate ticks_per_bar based on actual numerator and denominator
        int ticks_per_bar = (tpq * 4 * ts.numerator) / ts.denominator;
        timesigs[ts.time] = std::make_tuple(ticks_per_bar, ts.numerator, ts.denominator);
        max_tick = std::max(max_tick, (int)ts.time);
    }

    // Ensure we have a time signature at tick 0
    if (timesigs.find(0) == timesigs.end()) {
        timesigs[0] = std::make_tuple(tpq * 4, 4, 4);
    }

    // Map ticks to bars up to max_tick
    std::map<int, std::tuple<int, int, int, int>> bars_map; // start_tick -> (length, index, num, den)
    int bar_count = 0;
    auto it = timesigs.begin();
    while (it != timesigs.end()) {
        auto next_it = std::next(it);
        
        int start = it->first;
        int bar_len = std::get<0>(it->second);
        int num = std::get<1>(it->second);
        int den = std::get<2>(it->second);

        if (bar_len > 0) {
            if (next_it == timesigs.end()) {
                // Last time signature segment: generate bars that contain content.
                // A bar [start, start+bar_len) contains content only if start < max_tick.
                // When max_tick lands exactly on a bar boundary, do not create the empty
                // trailing bar starting at max_tick.
                bool emitted_any = false;
                while (start < max_tick) {
                    bars_map[start] = std::make_tuple(bar_len, bar_count++, num, den);
                    start += bar_len;
                    emitted_any = true;
                }
                // Guarantee at least one bar even for empty scores.
                if (!emitted_any) {
                    bars_map[start] = std::make_tuple(bar_len, bar_count++, num, den);
                }
            } else {
                // Intermediate segment: generate bars up to the next time signature
                int end = next_it->first;
                for (int t = start; t < end; t += bar_len) {
                    bars_map[t] = std::make_tuple(bar_len, bar_count++, num, den);
                }
            }
        }
        it = next_it;
    }

    auto get_bar_info = [&](int tick) -> std::tuple<int, int, int, int> {
        auto bit = bars_map.upper_bound(tick);
        if (bit == bars_map.begin()) return {0, 0, 4, 4};
        return prev(bit)->second;
    };

    // Pre-calculate cumulative bar offsets in target resolution using double precision
    std::vector<double> bar_starts_raw;
    double current_abs = 0;
    for (const auto& binfo : bars_map) {
        if (std::get<0>(binfo.second) == 0) continue; // skip sentinel
        bar_starts_raw.push_back(current_abs);
        current_abs += (double)out.resolution * std::get<0>(binfo.second) / tpq;
    }

    for (const auto& strack_ptr : *s.tracks) {
        const auto& strack = *strack_ptr;
        Track t;
        t.instrument = strack.program;
        t.type = strack.is_drum ? TrackType::Drum : TrackType::Melodic;
        
        // Initialize all bars for this track
        // We only add bars if they are actually found in bars_map
        for (const auto& binfo : bars_map) {
            if (std::get<0>(binfo.second) == 0) continue; // skip sentinel
            Bar b;
            b.ts_numerator = std::get<2>(binfo.second);
            b.ts_denominator = std::get<3>(binfo.second);
            b.beat_length = (double)std::get<0>(binfo.second) / tpq;
            b.has_notes = false; // Will be updated later
            t.bars.push_back(b);
        }

        for (const auto& snote : *strack.notes) {
            Note n;
            n.pitch = snote.pitch;
            n.velocity = snote.velocity;
            
            auto binfo = get_bar_info(snote.time);
            int bar_idx = std::get<1>(binfo);
            int bar_start_tick_symusic = 0;
            // Find the actual symusic start tick for this bar
            for (const auto& b : bars_map) {
                if (std::get<1>(b.second) == bar_idx) {
                    bar_start_tick_symusic = b.first;
                    break;
                }
            }

            int offset_tick_symusic = snote.time + snote.duration;
            auto offset_binfo = get_bar_info(offset_tick_symusic);
            int offset_bar_idx = std::get<1>(offset_binfo);
            int offset_bar_start_tick_symusic = 0;
            for (const auto& b : bars_map) {
                if (std::get<1>(b.second) == offset_bar_idx) {
                    offset_bar_start_tick_symusic = b.first;
                    break;
                }
            }

            // High-precision absolute tick calculation for onset and offset
            double onset_bar_abs_raw = bar_starts_raw[bar_idx];
            double offset_bar_abs_raw = bar_starts_raw[offset_bar_idx];

            double rel_onset_raw = (double)(snote.time - bar_start_tick_symusic) / tpq * out.resolution;
            n.onset_ticks = (int)round(rel_onset_raw);

            // Match Yellow original: a note that rounds to bar_len_target
            // (i.e. exactly the start of the next bar) is dropped, not
            // clamped into this bar. Without this, ref keeps a note that
            // orig discards and the two encoders diverge by ±1 note token.
            int bar_len_target = (int)round((double)out.resolution * std::get<0>(binfo) / tpq);
            if (bar_len_target > 0 && n.onset_ticks >= bar_len_target) {
                continue;
            }
            n.onset_ticks = std::max(0, n.onset_ticks);

            double rel_offset_raw = (double)(offset_tick_symusic - offset_bar_start_tick_symusic) / tpq * out.resolution;
            int offset_quantized = (int)round(rel_offset_raw);
            
            // Duration calculation matches MidiWriter's absolute tick reconstruction exactly
            int onset_abs_quantized = (int)round(onset_bar_abs_raw) + n.onset_ticks;
            int offset_abs_quantized = (int)round(offset_bar_abs_raw) + offset_quantized;

            n.duration_ticks = std::max(1, offset_abs_quantized - onset_abs_quantized);

            // Microtiming residual: how far the true onset falls from the
            // nearest out.resolution grid point, expressed as an integer
            // fraction of one grid cell (scaled by out.resolution so the
            // magnitude fits the Delta token domain, sized resolution/2).
            // Harmless for configs that don't emit Delta tokens: encoder.cpp
            // only reads note.delta when emit_delta_tokens is set, and
            // tokenizer.py's resample_delta ignores it entirely otherwise.
            double onset_residual = rel_onset_raw - std::round(rel_onset_raw);
            n.delta = (int)std::lround(onset_residual * out.resolution);

            // Debugging output
            LOG_DEBUG("Note: pitch=" + std::to_string(snote.pitch)
                      + ", sym_time=" + std::to_string(snote.time)
                      + ", sym_duration=" + std::to_string(snote.duration)
                      + ", bar_idx=" + std::to_string(bar_idx)
                      + ", sym_bar_start=" + std::to_string(bar_start_tick_symusic)
                      + ", rel_onset_raw=" + std::to_string(rel_onset_raw)
                      + ", n.onset_ticks=" + std::to_string(n.onset_ticks)
                      + ", onset_bar_abs_raw=" + std::to_string(onset_bar_abs_raw)
                      + ", onset_abs_quantized=" + std::to_string(onset_abs_quantized)
                      + ", offset_abs_quantized=" + std::to_string(offset_abs_quantized)
                      + ", n.duration_ticks=" + std::to_string(n.duration_ticks));

            out.notes.push_back(n);
            if (bar_idx < (int)t.bars.size()) {
                t.bars[bar_idx].note_indices.push_back(out.notes.size() - 1);
                t.bars[bar_idx].has_notes = true;
            }
        }
        out.tracks.push_back(t);
    }
    return out;
}

} // namespace midigpt::io
