#pragma once

#include "../../common/encoder/encoder_all.h"
#include <string>

namespace enums {

enum ENCODER_TYPE {
  EXPRESSIVE_ENCODER,
  NO_ENCODER
};

std::unique_ptr<encoder::ENCODER> getEncoder(ENCODER_TYPE et) {
  switch (et) {
    case EXPRESSIVE_ENCODER: return std::make_unique<encoder::ExpressiveEncoder>();
    case NO_ENCODER: return NULL;
  }
  return NULL;
}

ENCODER_TYPE getEncoderType(const std::string &s) {
  if (s == "EXPRESSIVE_ENCODER") return EXPRESSIVE_ENCODER;
  return NO_ENCODER;
}

std::vector<std::string> getEncoderTypeList() {
  std::vector<std::string> list;
  list.push_back("EXPRESSIVE_ENCODER");
  return list;
}

int getEncoderSize(ENCODER_TYPE et) {
  std::unique_ptr<encoder::ENCODER> encoder = getEncoder(et);
  if (!encoder) {
    return 0;
  }
  int size = encoder->rep->max_token();
  return size;
}

// helper for unit tests
inline bool starts_with(std::string const & value, std::string const & match) {
  if (match.size() > value.size()) return false;
  return std::equal(match.begin(), match.end(), value.begin());
}

std::unique_ptr<encoder::ENCODER> getEncoderFromString(const std::string &s) {
  return getEncoder(getEncoderType(s));
}

}
