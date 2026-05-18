#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "../core/score.h"
#include <symusic.h>

namespace midigpt::io {

class MidiReader {
public:
    explicit MidiReader(int resolution = 480) : resolution_(resolution) {}
    
    Score read(const std::string& path) const;
    Score read_bytes(const std::vector<uint8_t>& bytes) const;
private:
    Score from_symusic(const symusic::Score<symusic::Tick>& s) const;
    int resolution_;
};

} // namespace midigpt::io
