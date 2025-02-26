#pragma once

#include "midi.pb.h"

// START OF NAMESPACE
namespace data_structures {

std::map<midi::TRACK_TYPE,bool> TRACK_TYPE_IS_DRUM = {
  {midi::AUX_DRUM_TRACK, true},
  {midi::AUX_INST_TRACK, false},
  {midi::STANDARD_TRACK, false},
  {midi::STANDARD_DRUM_TRACK, true},
};

bool is_drum_track(int tt) {
  return TRACK_TYPE_IS_DRUM[static_cast<midi::TRACK_TYPE>(tt)];
}

}
// END OF NAMESPACE
