#pragma once

#include <google/protobuf/util/json_util.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include "../../common/data_structures/track_type.h"
#include "../../common/data_structures/encoder_config.h"
#include "../../common/data_structures/verbosity.h"

#include "../../inference/enum/density.h"
#include "../../inference/enum/constants.h"
#include "../../inference/enum/gm.h"
#include "../../inference/random.h"

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

// START OF NAMESPACE
namespace util_protobuf {

	// Checks if bar has features and returns them
	midi::BarFeatures* GetBarFeatures(midi::Track *track, int bar_num) {
		if ((bar_num < 0) || (bar_num >= track->bars_size())) {
			throw std::runtime_error("BAR FEATURE REQUEST OUT OF RANGE");
		}
		midi::Bar* bar = track->mutable_bars(bar_num);
		if (bar->internal_features_size() == 0) {
			return bar->add_internal_features();
		}
		return bar->mutable_internal_features(0);
	}

	// Checks if tracks has features and returns them
	midi::TrackFeatures* GetTrackFeatures(midi::Piece* midi_piece, int track_num) {
		if ((track_num < 0) || (track_num >= midi_piece->tracks_size())) {
			throw std::runtime_error("TRACK FEATURE REQUEST OUT OF RANGE");
		}
		//we return a pointer to the mutable track object with index track_num and we store the pointer in midi_track
		midi::Track* midi_track = midi_piece->mutable_tracks(track_num);
		if (midi_track->internal_features_size() == 0) {
			//adds new element to end of field and returns a pointer. The returned track features is mutable and will have none of its fields set.
			return midi_track->add_internal_features();
		}
		//returns a pointer to the underlying mutable track object with index track_num and we return the pointer
		return midi_track->mutable_internal_features(0);
	}

	midi::PieceFeatures* GetPieceFeatures(midi::Piece* midi_piece) {
		if (midi_piece->internal_features_size() == 0) {
			return midi_piece->add_internal_features();
		}
		return midi_piece->mutable_internal_features(0);
	}

	// Get the number of bars in a piece
	int GetNumBars(midi::Piece* midi_piece) {
		if (midi_piece->tracks_size() == 0) {
			return 0;
		}
		std::set<int> track_num_bars;
		for (const auto &track : midi_piece->tracks()) {
			track_num_bars.insert(track.bars_size());
		}
		if (track_num_bars.size() > 1) {
			throw std::runtime_error("Each track must have the same number of bars!");
		}
		//we dereference the pointer to the first element in the set (in this case the only element)
		return *track_num_bars.begin();
	}

	// ================================================================
	// Functions to update the note_polyphony field in the midi::Tracks of a midi::Piece
	// ================================================================

	midi::Note CreateNote(int start, int end, int pitch) {
		midi::Note note;
		note.set_start(start);
		note.set_end(end);
		note.set_pitch(pitch);
		return note;
	}

	// slightly different way to get notes
	std::vector<midi::Note> getNotes(midi::Piece* piece, int track_start, int track_end, int bar_start, int bar_end, bool onset_only_drums) {
		midi::Event current_midi_event;
		std::vector<midi::Note> notes;
		std::map<int, int> onsets; // key = pitch, value = start time
		for (int track_num=track_start; track_num<track_end; track_num++) {
			int current_time = 0;
			for (int bar_num=bar_start; bar_num<bar_end; bar_num++) {
				assert(track_num < piece->tracks_size());
				assert(bar_num < piece->tracks(track_num).bars_size());
				const midi::Track track = piece->tracks(track_num);
				const midi::Bar bar = track.bars(bar_num);
				for (const int event_id : bar.events()) {
					current_midi_event = piece->events(event_id);
					if (current_midi_event.velocity() > 0) {
					// need to account for bar offset to get correct start time
						int start_time = current_time + current_midi_event.time();
						if ((data_structures::is_drum_track(track.track_type())) && (onset_only_drums)) {
							notes.push_back(util_protobuf::CreateNote(start_time, start_time + 1, current_midi_event.pitch()));
						}
						else {
							onsets[current_midi_event.pitch()] = start_time;
						}
					}
					else {
						auto last_event_with_pitch = onsets.find(current_midi_event.pitch());
						int end_time = current_time + current_midi_event.time();
						if (last_event_with_pitch != onsets.end()) {
							notes.push_back(util_protobuf::CreateNote(last_event_with_pitch->second, end_time, last_event_with_pitch->first));
							onsets.erase(last_event_with_pitch);
						}
					}
				}
				current_time += piece->resolution() * bar.internal_beat_length();
			}
		}
		return notes;
	}

	// Go over all the bars and convert midi::events to midi::notes
	std::vector<midi::Note> IterateAndConvert(midi::Piece* midi_piece, const midi::Track* current_track, bool bool_drum_track, int* duration_in_ticks) {
		midi::Event current_midi_event;
		std::vector<midi::Note> notes;
		std::map<int, int> onsets;
		int bar_start = 0;
		for (int bar_num = 0; bar_num < current_track->bars_size(); bar_num++) {
			const midi::Bar bar = current_track->bars(bar_num);
			for (auto event_id : bar.events()) {
				current_midi_event = midi_piece->events(event_id);
				if (current_midi_event.velocity() > 0) {
					// need to account for bar offset to get correct start time
					onsets[current_midi_event.pitch()] = bar_start + current_midi_event.time();
				}
				else {
					auto last_event_with_pitch = onsets.find(current_midi_event.pitch());
					// need to account for bar offset to get correct end time
					int end_time = bool_drum_track ? last_event_with_pitch->second + 1 : bar_start + current_midi_event.time();
					if (last_event_with_pitch != onsets.end()) {
						midi::Note note = CreateNote(last_event_with_pitch->second, end_time, last_event_with_pitch->first);
						notes.push_back(note);
						onsets.erase(last_event_with_pitch);
					}
				}
				*duration_in_ticks = std::max(*duration_in_ticks, bar_start + current_midi_event.time());
			}
			bar_start += midi_piece->resolution() * bar.internal_beat_length();
		}
		return notes;
	}

	// Get a specific track from a midi piece and convert its midi::events to midi::notes
	std::vector<midi::Note> TrackEventsToNotes(midi::Piece* midi_piece, int track_num, int* duration_in_ticks) {
		bool bool_drum_track = data_structures::is_drum_track(midi_piece->tracks(track_num).track_type()); //TODO: this should be renamed is_drum_track = check_if_drum_track()... refactor
		const midi::Track* current_track = &(midi_piece->tracks(track_num));
		std::vector<midi::Note> notes = IterateAndConvert(midi_piece, current_track, bool_drum_track, duration_in_ticks); //TODO: This is a mayor change, but maybe the .proto shouldn't keep the events int the Piece, and instead keep them in the track message type
		return notes;
	}

	// Get the notes playing simultaneously per tick and return the tick with most note count.
	int GetTrackMaxPolyphony(std::vector<midi::Note>& notes, int duration_in_ticks) {
		int max_polyphony = 0;
		std::vector<int> flat_roll(duration_in_ticks, 0);
		for (const auto &note : notes) {
			for (int tick = note.start(); tick < note.end(); tick++) {
				flat_roll[tick]++;
				max_polyphony = std::max(flat_roll[tick], max_polyphony);
			}
		}
		return max_polyphony;
	}

	// ================================================================
	// Functions to convert a polyphonic track to a monophonic one
	// ================================================================

	// We create an array of monophonic events
	// we iterate over events
	// if an event starts, we flag it.
	// if another event starts before the flag is down, we force the first event to end and 
	// be pushed in the array. We then flag the new event as being played
	// if the event ends before another starts, we just push it in the array

	void UpdateHasNotes(midi::Piece* midi_piece) {
		int track_num = 0;
		for (const auto &track : midi_piece->tracks()) {
			int bar_num = 0;
			for (const auto &bar : track.bars()) {
				bool has_notes = false;
				for (const auto &event_index : bar.events()) {
					if (midi_piece->events(event_index).velocity() > 0) {
						has_notes = true;
						break;
					}
				}
				midi_piece->mutable_tracks(track_num)->mutable_bars(bar_num)->set_internal_has_notes(has_notes);
				bar_num++;
			}
			track_num++;
		}
	}

	// ========================================================================
	// RANDOM SEGMENT SELECTION FOR TRAINING
	// 
	// 1. we select an index of a random segment


	void UpdateValidSegments(midi::Piece* midi_piece, int seglen, int min_tracks) {
		UpdateHasNotes(midi_piece);
		midi_piece->clear_internal_valid_segments();
		midi_piece->clear_internal_valid_tracks();

		if (midi_piece->tracks_size() < min_tracks) { return; } // no valid tracks

		int min_non_empty_bars = round(seglen * .75);
		int num_bars = GetNumBars(midi_piece);

		for (int start = 0; start < num_bars - seglen + 1; start++) {

			// check that all time sigs are supported
			bool is_four_four = true;

			// check which tracks are valid
			midi::ValidTrack vtracks;
			std::map<int, int> used_track_types;
			for (int track_num = 0; track_num < midi_piece->tracks_size(); track_num++) {
				int non_empty_bars = 0;
				for (int k = 0; k < seglen; k++) {
					if (midi_piece->tracks(track_num).bars(start + k).internal_has_notes()) {
						non_empty_bars++;
					}
				}
				if (non_empty_bars >= min_non_empty_bars) {
					vtracks.add_tracks(track_num);
				}
			}

			// check if there are enough tracks
			bool enough_tracks = vtracks.tracks_size() >= min_tracks;

			if (enough_tracks && is_four_four) {
				midi::ValidTrack* v = midi_piece->add_internal_valid_tracks_v2();
				v->CopyFrom(vtracks);
				midi_piece->add_internal_valid_segments(start);
			}
		}
	}

	// ================================================================
	// Non-factorized functions for inference
	// ================================================================

	inline double midigpt_log2(const long double x) {
		return std::log(x) * M_LOG2E;
	}

	template <typename T>
	T clip(const T& n, const T& lower, const T& upper) {
		return std::max(lower, std::min(n, upper));
	}

	template<typename T>
	std::vector<T> quantile(std::vector<T>& x, std::vector<double> qs) {
		std::vector<T> vals;
		for (const auto &q : qs) {
			if (x.size()) {
				int index = std::min((int)round((double)x.size() * q), (int)x.size() - 1);
				std::nth_element(x.begin(), x.begin() + index, x.end());
				vals.push_back(x[index]);
			}
			else {
				vals.push_back(0);
			}
		}
		return vals;
	}

	template<typename T>
	T min_value(std::vector<T>& x) {
		auto result = std::min_element(x.begin(), x.end());
		if (result == x.end()) {
			return 0;
		}
		return *result;
	}

	template<typename T>
	T max_value(std::vector<T>& x) {
		auto result = std::max_element(x.begin(), x.end());
		if (result == x.end()) {
			return 0;
		}
		return *result;
	}

	template <typename T>
	std::string protobuf_to_string(T* x) {
		std::string output;
		google::protobuf::util::JsonPrintOptions opt;
		opt.add_whitespace = true;
		google::protobuf::util::MessageToJsonString(*x, &output, opt);
		return output;
	}

	std::vector<int> get_note_durations(std::vector<midi::Note>& notes) {
		std::vector<int> durations;
		for (const auto &note : notes) {
			double d = note.end() - note.start();
			durations.push_back((int)clip(midigpt_log2(std::max(d / 3., 1e-6)) + 1, 0., 5.));
		}
		return durations;
	}

	std::tuple<double, double, double, double, double, double> av_polyphony_inner(std::vector<midi::Note>& notes, int max_tick, midi::TrackFeatures* f) {
		int nonzero_count = 0;
		double count = 0;
		std::vector<int> flat_roll(max_tick, 0);
		for (const auto &note : notes) {
			for (int t = note.start(); t < std::min(note.end(), max_tick - 1); t++) {
				if (flat_roll[t] == 0) {
					nonzero_count += 1;
				}
				flat_roll[t]++;
				count++;
			}
		}

		std::vector<int> nz;
		for (const auto &x : flat_roll) {
			if (x > 0) {
				nz.push_back(x);
				if (f) {
					f->add_polyphony_distribution(x);
				}
			}
		}

		double silence = max_tick - nonzero_count;

		std::vector<int> poly_qs = quantile<int>(nz, { .15,.85 });

		double min_polyphony = min_value(nz);
		double max_polyphony = max_value(nz);

		double av_polyphony = count / std::max(nonzero_count, 1);
		double av_silence = silence / std::max(max_tick, 1);
		return std::make_tuple(av_polyphony, av_silence, poly_qs[0], poly_qs[1], min_polyphony, max_polyphony);
	}

	double note_duration_inner(std::vector<midi::Note>& notes) {
		double total_diff = 0;
		for (const auto &note : notes) {
			total_diff += (note.end() - note.start());
		}
		return total_diff / std::max((int)notes.size(), 1);
	}

	// function to get note density value
	int get_note_density_target(midi::Track* track, int bin) {
		int qindex = track->instrument();
		int tt = track->track_type();
		if (data_structures::is_drum_track(tt)) {
			qindex = 128;
		}
		return enums::DENSITY_QUANTILES[qindex][bin];
	}

	void update_note_density(midi::Piece* x) {

		int track_num = 0;
		int num_notes;
		for (const auto &track : x->tracks()) {

			// calculate average notes per bar
			num_notes = 0;
			int bar_num = 0;
			std::set<int> valid_bars;
			for (const auto &bar : track.bars()) {
				for (const auto &event_index : bar.events()) {
					if (x->events(event_index).velocity()) {
						valid_bars.insert(bar_num);
						num_notes++;
					}
				}
				bar_num++;
			}
			int num_bars = std::max((int)valid_bars.size(), 1);
			double av_notes_fp = (double)num_notes / num_bars;
			int av_notes = round(av_notes_fp);

			// calculate the density bin
			int qindex = track.instrument();
			int bin = 0;

			if (data_structures::is_drum_track(track.track_type())) {
				qindex = 128;
			}
			while (av_notes > enums::DENSITY_QUANTILES[qindex][bin]) {
				bin++;
			}

			// update protobuf
			midi::TrackFeatures* tf = GetTrackFeatures(x, track_num);
			tf->set_note_density_v2(bin);
			tf->set_note_density_value(av_notes_fp);
			track_num++;


		}
	}

	// adding note durations to events
	void calculate_note_durations(midi::Piece* p) {
		// to start set all durations == 0
		for (int i = 0; i < p->events_size(); i++) {
			p->mutable_events(i)->set_internal_duration(0);
		}

		for (const auto &track : p->tracks()) {
			// pitches to (abs_time, event_index)
			std::map<int, std::tuple<int, int>> onsets;
			int bar_start = 0;
			for (const auto &bar : track.bars()) {
				for (auto event_id : bar.events()) {
					midi::Event e = p->events(event_id);
					//data_structures::LOGGER( "PROC EVENT :: " , e.pitch() , " " , e.velocity() , " " , e.time() );
					if (e.velocity() > 0) {
						if (data_structures::is_drum_track(track.track_type())) {
							// drums always have duration of 1 timestep
							p->mutable_events(event_id)->set_internal_duration(1);
						}
						else {
							onsets[e.pitch()] = std::make_tuple(bar_start + e.time(), event_id);
						}
					}
					else {
						auto it = onsets.find(e.pitch());
						if (it != onsets.end()) {
							int index = std::get<1>(it->second);
							int duration = (bar_start + e.time()) - std::get<0>(it->second);
							p->mutable_events(index)->set_internal_duration(duration);
						}
					}
				}
				// move forward a bar
				bar_start += p->resolution() * bar.internal_beat_length();
			}
		}
	}

	void update_av_polyphony_and_note_duration(midi::Piece* p) {
		for (int track_num = 0; track_num < p->tracks_size(); track_num++) {
			int max_tick = 0;
			std::vector<midi::Note> notes = TrackEventsToNotes(
				p, track_num, &max_tick);
			std::vector<int> durations = get_note_durations(notes);
			midi::TrackFeatures* f = GetTrackFeatures(p, track_num);
			auto stat = av_polyphony_inner(notes, max_tick, f);
			f->set_note_duration(note_duration_inner(notes));
			f->set_av_polyphony(std::get<0>(stat));
			f->set_min_polyphony_q(
				std::max(std::min((int)std::get<2>(stat), 10), 1) - 1);
			f->set_max_polyphony_q(
				std::max(std::min((int)std::get<3>(stat), 10), 1) - 1);

			std::vector<int> dur_qs = quantile(durations, { .15,.85 });
			f->set_min_note_duration_q(dur_qs[0]);
			f->set_max_note_duration_q(dur_qs[1]);

			// new hard upper lower limits
			f->set_min_polyphony_hard(std::get<4>(stat));
			f->set_max_polyphony_hard(std::get<5>(stat));

			f->set_min_note_duration_hard(min_value(durations));
			f->set_max_note_duration_hard(max_value(durations));

		}
	}

	std::tuple<int, int> get_pitch_extents(midi::Piece* x) {
		int min_pitch = INT_MAX;
		int max_pitch = 0;
		for (const auto &track : x->tracks()) {
			if (!data_structures::is_drum_track(track.track_type())) {
				for (const auto &bar : track.bars()) {
					for (const auto &event_index : bar.events()) {
						int pitch = x->events(event_index).pitch();
						min_pitch = std::min(pitch, min_pitch);
						max_pitch = std::max(pitch, max_pitch);
					}
				}
			}
		}
		return std::make_pair(min_pitch, max_pitch);
	}

	void select_random_segment_indices(midi::Piece* x, int num_bars, int min_tracks, int max_tracks, std::mt19937* engine, std::vector<int>& valid_tracks, int* start) {
		UpdateValidSegments(x, num_bars, min_tracks);

		if (x->internal_valid_segments_size() == 0) {
			throw std::runtime_error("NO VALID SEGMENTS");
		}

		int index = random_on_range(x->internal_valid_segments_size(), engine);
		(*start) = x->internal_valid_segments(index);
		for (const auto &track_num : x->internal_valid_tracks_v2(index).tracks()) {
			valid_tracks.push_back(track_num);
		}
		shuffle(valid_tracks.begin(), valid_tracks.end(), *engine);

		// limit the tracks
		int ntracks = std::min((int)valid_tracks.size(), max_tracks);
		valid_tracks.resize(ntracks);
	}

	void prune_tracks(midi::Piece* x, std::vector<int> tracks, std::vector<int> bars) {

		if (x->tracks_size() == 0) {
			return;
		}

		midi::Piece tmp(*x);

		int num_bars = GetNumBars(x);
		bool remove_bars = (int)bars.size() > 0;
		x->clear_tracks();
		x->clear_events();

		std::vector<int> tracks_to_keep;
		for (const auto &track_num : tracks) {
			if ((track_num >= 0) && (track_num < tmp.tracks_size())) {
				tracks_to_keep.push_back(track_num);
			}
		}

		std::vector<int> bars_to_keep;
		for (const auto &bar_num : bars) {
			if ((bar_num >= 0) && (bar_num < num_bars)) {
				bars_to_keep.push_back(bar_num);
			}
		}

		for (const auto &track_num : tracks_to_keep) {
			const midi::Track track = tmp.tracks(track_num);
			midi::Track* t = x->add_tracks();
			t->CopyFrom(track);
			if (remove_bars) {
				t->clear_bars();
				for (const auto &bar_num : bars_to_keep) {
					const midi::Bar bar = track.bars(bar_num);
					midi::Bar* b = t->add_bars();
					b->CopyFrom(bar);
					b->clear_events();
					for (const auto &event_index : bar.events()) {
						b->add_events(x->events_size());
						midi::Event* e = x->add_events();
						e->CopyFrom(tmp.events(event_index));
					}
				}
			}
		}
	}

	void select_random_segment(midi::Piece* x, int num_bars, int min_tracks, int max_tracks, std::mt19937* engine) {
		int start;
		std::vector<int> valid_tracks;
		select_random_segment_indices(
			x, num_bars, min_tracks, max_tracks, engine, valid_tracks, &start);
		std::vector<int> bars = arange(start, start + num_bars, 1);
		prune_tracks(x, valid_tracks, bars);
	}

	std::set<std::tuple<int, int>> make_bar_mask(midi::Piece* x, float proportion, std::mt19937* engine) {
		int num_tracks = x->tracks_size();
		int num_bars = GetNumBars(x);
		int max_filled_bars = (int)round(num_tracks * num_bars * proportion);
		int n_fill = random_on_range(max_filled_bars, engine);
		std::vector<std::tuple<int, int>> choices;
		for (int track_num = 0; track_num < num_tracks; track_num++) {
			for (int bar_num = 0; bar_num < num_bars; bar_num++) {
				choices.push_back(std::make_pair(track_num, bar_num));
			}
		}
		std::set<std::tuple<int, int>> mask;
		shuffle(choices.begin(), choices.end(), *engine);
		for (int i = 0; i < n_fill; i++) {
			mask.insert(choices[i]);
		}
		return mask;
	}

	std::string get_piece_string(midi::Piece* x) {
		std::string output;
		google::protobuf::util::JsonPrintOptions opt;
		opt.add_whitespace = true;
		google::protobuf::util::MessageToJsonString(*x, &output, opt);
		return output;
	}

	void print_piece(midi::Piece* x) {
		data_structures::LOGGER( get_piece_string(x) );
	}

	void print_piece_summary(midi::Piece* x) {
		midi::Piece c(*x);
		c.clear_events();
		for (int track_num = 0; track_num < c.tracks_size(); track_num++) {
			c.mutable_tracks(track_num)->clear_bars();
		}
		print_piece(&c);
	}

	void reorder_tracks(midi::Piece* x, std::vector<int> track_order) {
		int num_tracks = x->tracks_size();
		if (num_tracks != (int)track_order.size()) {
			data_structures::LOGGER(data_structures::to_str( num_tracks , " " , track_order.size() ));
			throw std::runtime_error("Track order does not match midi::Piece.");
		}
		for (int track_num = 0; track_num < num_tracks; track_num++) {
			GetTrackFeatures(x, track_num)->set_order(track_order[track_num]);
		}
		std::sort(
			x->mutable_tracks()->begin(),
			x->mutable_tracks()->end(),
			[](const midi::Track& a, const midi::Track& b) {
				return a.internal_features(0).order() < b.internal_features(0).order();
			}
		);
	}

	template <typename T>
	void string_to_protobuf(std::string& s, T* x) {
		google::protobuf::util::JsonParseOptions opt;
    	opt.ignore_unknown_fields = true;
		google::protobuf::util::JsonStringToMessage(s, x, opt);
	}

	template <typename T>
	std::string enum_to_string(const T &value) {
		const google::protobuf::EnumDescriptor *descriptor = google::protobuf::GetEnumDescriptor<T>();
		return descriptor->FindValueByNumber(value)->name();
	}

	template <typename T>
	T string_to_enum(const std::string &name) {
		const google::protobuf::EnumDescriptor *descriptor = google::protobuf::GetEnumDescriptor<T>();
		return static_cast<T>(descriptor->FindValueByName(name)->number());
	}

	template <typename T>
	void print_protobuf(T* x) {
		data_structures::LOGGER( protobuf_to_string(x) );
	}

	void pad_piece_with_status(midi::Piece* p, midi::Status* s, int min_bars) {
		// add tracks when status references ones that do not exist
		for (const auto &track : s->tracks()) {
			midi::Track* t = NULL;
			if (track.track_id() >= p->tracks_size()) {
				t = p->add_tracks();
				t->set_track_type(track.track_type());
				data_structures::LOGGER(data_structures::to_str( "adding track " , track.track_id() ));
			}
			else {
				data_structures::LOGGER(data_structures::to_str( "using track " , track.track_id() ));
				t = p->mutable_tracks(track.track_id());
			}
			for (int i = t->bars_size(); i < 5; i++) {} // WHAT IS THIS ???
			data_structures::LOGGER(data_structures::to_str( "track " , track.track_id() , " has " , t->bars_size() , " bars" ));
			int num_bars = std::max(track.selected_bars_size(), min_bars);
			data_structures::LOGGER(data_structures::to_str( "adding " , num_bars , " bars" ));
			for (int i = t->bars_size(); i < num_bars; i++) {
				data_structures::LOGGER(data_structures::to_str( "adding bar " , i ));
				midi::Bar* b = t->add_bars();
				data_structures::LOGGER(data_structures::to_str( "check " , i ));
				b->set_internal_beat_length(4);
				b->set_ts_numerator(4);
				b->set_ts_denominator(4);
			}
			data_structures::LOGGER( "end" );
		}
	}


	midi::GM_TYPE gm_inst_to_string(int track_type, int instrument) {
		return enums::GM_REV[data_structures::is_drum_track(track_type) * 128 + instrument];
	}

	void status_from_piece(midi::Piece *piece, midi::Status *status) {
		status->Clear();
		int track_num = 0;
		for (const auto &track : piece->tracks()) {
			midi::StatusTrack *strack = status->add_tracks();
			strack->set_track_id( track_num );
			strack->set_track_type( track.track_type() );
			strack->set_density( midi::DENSITY_ANY ); //
			strack->set_instrument(gm_inst_to_string(track.track_type(),track.instrument()));
			strack->set_polyphony_hard_limit( 10 );
			strack->set_temperature( 1. );
			for (int i=0; i<track.bars_size(); i++) {
				strack->add_selected_bars( false );
				strack->add_bars(); // for bar level controls
			}
			track_num++;
		}
	}

	std::string status_from_piece_py(std::string &piece_str) {
		midi::Piece p;
		midi::Status s;
		string_to_protobuf(piece_str, &p);
		status_from_piece(&p, &s);
		return protobuf_to_string(&s);
	}

	midi::HyperParam default_sample_param() {
		midi::HyperParam param;
		param.set_tracks_per_step( 1 );
		param.set_bars_per_step( 2 );
		param.set_model_dim( 4 );
		param.set_shuffle( true );
		param.set_percentage( 100 );
		param.set_temperature( 1. );
		param.set_batch_size( 1 );
		param.set_max_steps( 0 );
		param.set_verbose( false );
		param.set_polyphony_hard_limit( 5 );
		return param;
	}

	std::string default_sample_param_py() {
		midi::HyperParam param = default_sample_param();
		return protobuf_to_string(&param);
	}

	std::string prune_tracks_py(std::string json_string, std::vector<int> tracks, std::vector<int> bars) {
		midi::Piece p;
		string_to_protobuf(json_string, &p);
		prune_tracks(&p, tracks, bars);
		return protobuf_to_string(&p);
	}

}
// END OF NAMESPACE
