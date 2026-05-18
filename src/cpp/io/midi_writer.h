#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "../core/score.h"
#include <symusic.h>

namespace midigpt::io {

class MidiWriter {
public:
    void write(const Score& score, const std::string& path) const;
    std::vector<uint8_t> write_bytes(const Score& score) const;
private:
    symusic::Score<symusic::Tick> to_symusic(const Score& score) const;
};

} // namespace midigpt::io
