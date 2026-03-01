#pragma once
// symusic-backed MIDI parser.  Drop-in replacement for the midifile-based
// MidiParsedData + Parser when compiled with -DMIDIGPT_USE_SYMUSIC.
//
// The public API (ParseSong / write_midi) is identical so that callers in
// encoder_base.h, lib.cpp, feature_extraction.h need no changes.

#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <set>
#include <fstream>
#include <filesystem>

#include <symusic.h>

#include "../../common/midi_parsing/util_protobuf.h"
#include "../../common/data_structures/track_type.h"
#include "../../common/data_structures/encoder_config.h"
#include "../../common/midi_parsing/adjacent_range.h"

#include <google/protobuf/util/json_util.h>

// START OF NAMESPACE
namespace midi_io {

// ── helpers shared with the midifile backend ────────────────────────────────

float quantize_beat_float(double x, double TPQ, double SPQ, double cut=.5) {
  return (int)((x / TPQ * SPQ) + (1.-cut)) * (TPQ / SPQ);
}

int quantize_beat(double x, double TPQ, double SPQ, double cut=.5) {
  return (int)quantize_beat_float(x, TPQ, SPQ, cut);
}

int get_time_difference(double x, double y, double xpq, double spq, double tempo, int beats_per_note) {
  return (int)(1000 * 60 * beats_per_note * (y - x) )/(4 * xpq * tempo);
}

bool event_comparator(const midi::Event a, const midi::Event b) {
  if (a.time() != b.time()) {
    return a.time() < b.time();
  }
  if (std::min(a.velocity(),1) != std::min(b.velocity(),1)) {
    return std::min(a.velocity(),1) < std::min(b.velocity(),1);
  }
  return a.pitch() < b.pitch();
}
bool event_pair_comparator(const std::pair<midi::Event, int> a, const std::pair<midi::Event, int> b) {
  if (a.first.time() != b.first.time()) {
    return a.first.time() < b.first.time();
  }
  if (std::min(a.first.velocity(),1) != std::min(b.first.velocity(),1)) {
    return std::min(a.first.velocity(),1) < std::min(b.first.velocity(),1);
  }
  return a.first.pitch() < b.first.pitch();
}

using TRACK_IDENTIFIER = std::tuple<int,int,int,int>;

// ── MidiParsedData (symusic) ────────────────────────────────────────────────
// Wraps symusic::Score<symusic::Tick>.  Exposes the same two public fields
// that the midifile MidiParsedData does: track_count, ticks_per_quarter_note.

class MidiParsedData {
public:
    symusic::Score<symusic::Tick> score;
    int track_count;
    int ticks_per_quarter_note;

    MidiParsedData(std::string file_path)
        : score(symusic::Score<symusic::Tick>::parse<symusic::DataFormat::MIDI>(
              symusic::read_file(std::filesystem::path(file_path))))
    {
        track_count = static_cast<int>(score.tracks->size());
        ticks_per_quarter_note = score.ticks_per_quarter;
    }
};

// ── Parser (symusic) ────────────────────────────────────────────────────────

class Parser {
public:
  Parser(std::string filepath, midi::Piece *piece, const std::shared_ptr<data_structures::EncoderConfig> &config) {
      Parse(filepath, piece, config);
  }

  static const int DRUM_CHANNEL = 9;
  std::shared_ptr<data_structures::EncoderConfig> ec;
  int TPQ;
  int SPQ;
  int max_tick;
  int tempo;
  std::map<TRACK_IDENTIFIER,int> track_map;
  std::map<int,TRACK_IDENTIFIER> rev_track_map;
  std::map<int,std::tuple<int,int,int>> timesigs;
  std::map<int,std::tuple<int,int,int,int>> bars;
  std::vector<std::vector<midi::Event>> events;

  void SetMemberVariables(const std::shared_ptr<data_structures::EncoderConfig> &config, MidiParsedData* parsed_file) {
      ec = config;
      TPQ = parsed_file->ticks_per_quarter_note;
      SPQ = ec->resolution;
      if (TPQ < SPQ) {
          throw std::runtime_error("MIDI FILE HAS INVALID TICKS PER QUARTER.");
      }
  }

  // Symusic gives us note-level data.  We must generate the same onset +
  // offset midi::Event pairs that the midifile parser produces.
  void FillPiece(midi::Piece* piece, MidiParsedData* parsed_file,
                 const std::shared_ptr<data_structures::EncoderConfig> &/*config*/) {
      piece->set_resolution(SPQ);
      piece->set_internal_ticks_per_quarter(TPQ);
      max_tick = 0;

      // ── extract time signatures ──
      for (const auto &ts : *parsed_file->score.time_signatures) {
          int numerator = ts.numerator;
          int denominator = ts.denominator;
          int barlength = static_cast<int>(
              static_cast<double>(TPQ) * 4 * numerator / denominator);
          if (barlength >= 0) {
              timesigs[static_cast<int>(ts.time)] =
                  std::make_tuple(barlength, numerator, denominator);
          }
      }

      // ── extract tempo (use last tempo event, matching midifile behaviour) ──
      for (const auto &t : *parsed_file->score.tempos) {
          tempo = static_cast<int>(t.qpm());
          piece->set_tempo(tempo);
      }

      // ── process tracks and notes ──
      // symusic tracks are already split by (channel, program).
      // We assign each symusic track a sequential track index (0-based) just
      // like the midifile parser does via its track_map.
      for (int trk_idx = 0; trk_idx < parsed_file->track_count; trk_idx++) {
          const auto &trk = *(*parsed_file->score.tracks)[trk_idx];
          int instrument = trk.program;
          bool is_drum = trk.is_drum;
          int track_type = is_drum ? midi::STANDARD_DRUM_TRACK : midi::STANDARD_TRACK;

          // Build TRACK_IDENTIFIER.  symusic doesn't preserve the original
          // MIDI track index; we use trk_idx as a stand-in.  The channel
          // field is set to DRUM_CHANNEL for drum tracks, 0 otherwise (the
          // actual channel doesn't matter beyond drum detection).
          int channel = is_drum ? DRUM_CHANNEL : 0;
          TRACK_IDENTIFIER track_info = std::make_tuple(
              trk_idx, channel, instrument, track_type);

          // Register track (should always be new since trk_idx is unique)
          int current_size = static_cast<int>(track_map.size());
          track_map[track_info] = current_size;
          rev_track_map[current_size] = track_info;
          events.push_back(std::vector<midi::Event>());

          for (const auto &note : *trk.notes) {
              int onset_tick = static_cast<int>(note.time);
              int offset_tick = static_cast<int>(note.time + note.duration);
              int pitch = static_cast<int>(note.pitch);
              int velocity = static_cast<int>(note.velocity);

              max_tick = std::max(max_tick, offset_tick);

              // ── onset ──
              {
                  int tick = onset_tick;
                  float float_tick = static_cast<float>(onset_tick);
                  int unquantized_tick = onset_tick;
                  if (!ec->unquantized) {
                      tick = quantize_beat(onset_tick, TPQ, SPQ);
                      float_tick = quantize_beat_float(onset_tick, TPQ, SPQ);
                  }

                  int delta = 0;
                  if (ec->use_microtiming) {
                      delta = ec->step_to_delta(unquantized_tick - float_tick, TPQ);
                  }

                  if (is_drum) {
                      // Drum: add onset + immediate short offset
                      add_event(track_info, tick, pitch, velocity, delta);
                      add_event(track_info, tick + (TPQ/SPQ), pitch, 0, delta);
                  } else {
                      add_event(track_info, tick, pitch, velocity, delta);
                  }
              }

              // ── offset (non-drum only) ──
              if (!is_drum) {
                  int tick = offset_tick;
                  float float_tick = static_cast<float>(offset_tick);
                  int unquantized_tick = offset_tick;
                  if (!ec->unquantized) {
                      tick = quantize_beat(offset_tick, TPQ, SPQ);
                      float_tick = quantize_beat_float(offset_tick, TPQ, SPQ);
                  }

                  // Skip offsets at tick 0 (same as midifile parser)
                  if (tick == 0) continue;

                  int delta = 0;
                  if (ec->use_microtiming) {
                      delta = ec->step_to_delta(unquantized_tick - float_tick, TPQ);
                  }

                  add_event(track_info, tick, pitch, 0, delta);
              }
          }
      }

      if (max_tick <= 0) {
          throw std::runtime_error("MIDI FILE HAS NO NOTES");
      }

      piece->set_internal_has_time_signatures(timesigs.size() > 0);
  }

  void ProcessTimeSignatures(MidiParsedData* parsed_file) {
      int count = 0;
      if (timesigs.find(0) == timesigs.end()) {
          timesigs[0] = std::make_tuple(
              parsed_file->ticks_per_quarter_note * 4, 4, 4);
      }
      timesigs[max_tick] = std::make_tuple(0, 0, 0);
      for (const auto& p : midi_parsing::make_adjacent_range(timesigs)) {
          if (std::get<0>(p.first.second) > 0) {
              for (int t = p.first.first; t < p.second.first;
                   t += std::get<0>(p.first.second)) {
                  auto ts = p.first.second;
                  bars[t] = std::make_tuple(
                      std::get<0>(ts), count, std::get<1>(ts), std::get<2>(ts));
                  count++;
              }
          }
      }
  }

  void CreateMidiPiece(midi::Piece* piece, MidiParsedData* parsed_file) {
      midi::Track* track = NULL;
      midi::Bar* bar = NULL;
      midi::Event* event = NULL;

      for (int track_num = 0; track_num < (int)events.size(); track_num++) {
          std::sort(events[track_num].begin(), events[track_num].end(),
                    event_comparator);

          track = piece->add_tracks();
          track->set_instrument(std::get<2>(rev_track_map[track_num]));
          track->set_track_type(
              (midi::TRACK_TYPE)std::get<3>(rev_track_map[track_num]));

          for (const auto& bar_info : bars) {
              bar = track->add_bars();
              bar->set_internal_beat_length(
                  std::get<0>(bar_info.second) /
                  parsed_file->ticks_per_quarter_note);
              bar->set_ts_numerator(std::get<2>(bar_info.second));
              bar->set_ts_denominator(std::get<3>(bar_info.second));
          }

          for (int j = 0; j < (int)events[track_num].size(); j++) {
              int velocity = events[track_num][j].velocity();
              int tick = events[track_num][j].time();
              auto bar_info = get_bar_info(tick, velocity > 0);

              bar = track->mutable_bars(std::get<2>(bar_info));
              bar->set_internal_has_notes(true);

              bar->add_events(piece->events_size());
              event = piece->add_events();
              event->CopyFrom(events[track_num][j]);

              int rel_tick = round(
                  (double)(tick - std::get<0>(bar_info)) /
                  parsed_file->ticks_per_quarter_note * SPQ);
              event->set_time(rel_tick);
          }
      }
  }

  void Parse(std::string filepath, midi::Piece* piece,
             const std::shared_ptr<data_structures::EncoderConfig> &config) {
      MidiParsedData parsed_file = MidiParsedData(filepath);
      SetMemberVariables(config, &parsed_file);
      FillPiece(piece, &parsed_file, config);
      ProcessTimeSignatures(&parsed_file);
      CreateMidiPiece(piece, &parsed_file);
  }

  std::tuple<int,int,int> get_bar_info(int tick, bool is_onset) {
      auto it = bars.upper_bound(tick);
      if (it == bars.begin()) {
          throw std::runtime_error("CAN'T GET BAR INFO FOR TICK!");
      }
      it = prev(it);
      if ((it->first == tick) && (!is_onset)) {
          if (it == bars.begin()) {
              throw std::runtime_error("CAN'T GET BAR INFO FOR TICK!");
          }
          it = prev(it);
      }
      return std::make_tuple(
          it->first, std::get<0>(it->second), std::get<1>(it->second));
  }

  void add_event(TRACK_IDENTIFIER &track_info, int tick, int pitch,
                 int velocity, int delta) {
      midi::Event event;
      event.set_time(tick);
      event.set_pitch(pitch);
      event.set_velocity(velocity);
      event.set_delta(delta);
      events[track_map[track_info]].push_back(event);
  }
};

void ParseSong(std::string filepath, midi::Piece *midi_piece,
               const std::shared_ptr<data_structures::EncoderConfig> &encoder_config) {
    Parser parser(filepath, midi_piece, encoder_config);
}

// ── write_midi (symusic) ────────────────────────────────────────────────────
// Converts a midi::Piece back to a Standard MIDI File via symusic.

void write_midi(midi::Piece* p, std::string& path, int single_track = -1) {
    static const int DRUM_CHANNEL = 9;

    if (p->tracks_size() >= 15) {
        throw std::runtime_error("TOO MANY TRACKS FOR MIDI OUTPUT");
    }

    symusic::Score<symusic::Tick> score(p->resolution());
    score.tempos->push_back(
        symusic::Tempo<symusic::Tick>(0, symusic::Tempo<symusic::Tick>::from_qpm(0, p->tempo()).mspq));

    int track_num = 0;
    for (const auto &track : p->tracks()) {
        if ((single_track < 0) || (track_num == single_track)) {
            auto out_track = std::make_shared<symusic::Track<symusic::Tick>>();
            out_track->program = track.instrument();
            out_track->is_drum = data_structures::is_drum_track(track.track_type());

            int bar_start_time = 0;
            for (const auto &bar : track.bars()) {
                for (const auto &event_index : bar.events()) {
                    const midi::Event e = p->events(event_index);
                    int time = bar_start_time + e.time();
                    int vel = e.velocity();
                    int pitch = e.pitch();

                    if (vel > 0) {
                        out_track->notes->push_back(
                            symusic::Note<symusic::Tick>(
                                time, 0, static_cast<int8_t>(pitch),
                                static_cast<int8_t>(vel)));
                    } else {
                        for (auto it = out_track->notes->rbegin();
                             it != out_track->notes->rend(); ++it) {
                            if (it->pitch == pitch && it->duration == 0) {
                                it->duration = time - it->time;
                                break;
                            }
                        }
                    }
                }
                bar_start_time += bar.internal_beat_length() * p->resolution();
            }
            score.tracks->push_back(out_track);
        }
        track_num++;
    }

    auto midi_bytes = score.dumps<symusic::DataFormat::MIDI>();
    symusic::write_file(std::filesystem::path(path),
                        std::span<const uint8_t>(midi_bytes.data(), midi_bytes.size()));
}

}
// END OF NAMESPACE
