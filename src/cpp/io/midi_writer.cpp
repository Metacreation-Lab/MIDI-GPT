#include "midi_writer.h"
#include <symusic.h>
#include <symusic/io/midi.h>
#include <fstream>
#include "../core/logging.h"

namespace midigpt::io {

void MidiWriter::write(const Score& score, const std::string& path) const {
    auto bytes = write_bytes(score);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::vector<uint8_t> MidiWriter::write_bytes(const Score& score) const {
    auto s = to_symusic(score);
    return s.dumps<::symusic::DataFormat::MIDI>();
}

symusic::Score<symusic::Tick> MidiWriter::to_symusic(const Score& score) const {
    symusic::Score<symusic::Tick> s;
    s.ticks_per_quarter = score.resolution;
    
    symusic::Tempo<symusic::Tick> tempo(0, score.tempo);
    s.tempos.push_back(tempo);

    // Track cumulative absolute ticks for bar starts using double precision
    // to match MidiReader's bar_starts_raw calculation logic exactly.
    std::vector<int> bar_starts;
    double current_abs_double = 0.0;
    
    const Track* ref_track = nullptr;
    size_t max_bars = 0;
    for (const auto& t : score.tracks) {
        if (t.bars.size() > max_bars) {
            max_bars = t.bars.size();
            ref_track = &t;
        }
    }

    if (ref_track) {
        for (size_t i = 0; i < ref_track->bars.size(); ++i) {
            const auto& b = ref_track->bars[i];
            int bar_start_quantized = (int)round(current_abs_double);
            bar_starts.push_back(bar_start_quantized);
            
            if (i == 0 || 
                b.ts_numerator != ref_track->bars[i-1].ts_numerator ||
                b.ts_denominator != ref_track->bars[i-1].ts_denominator) {
                s.time_signatures.push_back(symusic::TimeSignature<symusic::Tick>(
                    bar_start_quantized, b.ts_numerator, b.ts_denominator));
            }

            // Exactly match the reader's accumulation step: current_abs += resolution * length
            current_abs_double += (double)score.resolution * b.beat_length;
        }
    }

    for (const auto& t : score.tracks) {
        symusic::Track<symusic::Tick> strack;
        strack.program = t.instrument;
        strack.is_drum = (t.type == TrackType::Drum);
        
        for (size_t b_idx = 0; b_idx < t.bars.size(); ++b_idx) {
            const auto& b = t.bars[b_idx];
            int bar_offset = (b_idx < bar_starts.size()) ? bar_starts[b_idx] : 0;
            
            for (int idx : b.note_indices) {
                const auto& n = score.notes[idx];
                // MidiReader calculates absolute tick as:
                // onset_abs = round(onset_bar_abs_raw + rel_ticks_raw)
                // where rel_ticks_raw = (snote.time - bar_start_tick) / tpq * resolution
                // We have n.onset_ticks = round(rel_ticks_raw).
                // However, adding round(A) + round(B) is NOT necessarily round(A + B).
                // This was causing notes to shift across bar boundaries.

                // To achieve bit-perfect roundtrip parity, we MUST ensure the resulting
                // absolute MIDI tick is exactly what we read.
                // Since we don't store the raw unquantized onset, we must assume
                // that n.onset_ticks is the correct relative position.
                // But wait, if we used onset_abs = round(bar_start_raw + rel_raw),
                // then the "relative" onset we store should actually be:
                // stored_rel = round(bar_start_raw + rel_raw) - round(bar_start_raw)
                // This is what MidiReader now does.

                strack.notes.push_back(symusic::Note<symusic::Tick>(
                    (int)bar_offset + n.onset_ticks, n.duration_ticks, n.pitch, n.velocity));
            }
        }
        s.tracks.push_back(strack);
    }
    return s;
}

} // namespace midigpt::io
