#pragma once

#include "../../common/encoder/encoder_all.h"
#include <string>

namespace enums {

enum ENCODER_TYPE {
  EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER,
  EXPRESSIVE_ENCODER,
  STEINBERG_WPCS_ENCODER,
  GHOST_ENCODER,
  SPECTER_ENCODER,
  ORACLE_ENCODER,
  NO_ENCODER
};

std::unique_ptr<encoder::ENCODER> getEncoder(ENCODER_TYPE et) {
  switch (et) {
    case EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER: return std::make_unique<encoder::ElVelocityDurationPolyphonyYellowEncoder>();
    case EXPRESSIVE_ENCODER: return std::make_unique<encoder::ExpressiveEncoder>();
    case STEINBERG_WPCS_ENCODER: return std::make_unique<encoder::SteinbergWPCSEncoder>();
    case GHOST_ENCODER: return std::make_unique<encoder::GhostEncoder>();
    case SPECTER_ENCODER: return std::make_unique<encoder::SpecterEncoder>();
    case ORACLE_ENCODER: return std::make_unique<encoder::OracleEncoder>();
    case NO_ENCODER: return NULL;
  }
  return NULL;
}

ENCODER_TYPE getEncoderType(const std::string &s) {
  if (s == "EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER") return EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER;
  if (s == "EXPRESSIVE_ENCODER") return EXPRESSIVE_ENCODER;
  if (s == "STEINBERG_WPCS_ENCODER") return STEINBERG_WPCS_ENCODER;
  if (s == "STEINBERG_W_P_C_S_ENCODER") return STEINBERG_WPCS_ENCODER;
  if (s == "GHOST_ENCODER") return GHOST_ENCODER;
  if (s == "SPECTER_ENCODER") return SPECTER_ENCODER;
  if (s == "ORACLE_ENCODER") return ORACLE_ENCODER;
  return NO_ENCODER;
}

std::vector<std::string> getEncoderTypeList() {
  std::vector<std::string> list;
  list.push_back("EL_VELOCITY_DURATION_POLYPHONY_YELLOW_ENCODER");
  list.push_back("EXPRESSIVE_ENCODER");
  list.push_back("STEINBERG_WPCS_ENCODER");
  list.push_back("GHOST_ENCODER");
  list.push_back("SPECTER_ENCODER");
  list.push_back("ORACLE_ENCODER");
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
