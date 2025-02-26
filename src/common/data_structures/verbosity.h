// control verbosity levels in the code to make things cleaner

#pragma once

#include <sstream>

namespace data_structures {

enum VERBOSITY_LEVEL {
  VERBOSITY_LEVEL_QUIET,
  VERBOSITY_LEVEL_VERBOSE,
  VERBOSITY_LEVEL_DEBUG,
  VERBOSITY_LEVEL_TRACE
};

VERBOSITY_LEVEL GLOBAL_VERBOSITY_LEVEL = VERBOSITY_LEVEL_QUIET;

inline void setGlobalVerbosityLevel(VERBOSITY_LEVEL vl) {
  GLOBAL_VERBOSITY_LEVEL = vl;
}

template<typename T>
std::string to_str(const T& value){
  std::ostringstream tmp_str;
  tmp_str << value;
  return tmp_str.str();
}

template<typename T, typename ... Args >
std::string to_str(const T& value, const Args& ... args){
  return to_str(value) + to_str(args...);
}

template<typename T>
inline void LOGGER(T x) {
    if (GLOBAL_VERBOSITY_LEVEL >= VERBOSITY_LEVEL_VERBOSE) {
        std::cout << x << std::endl;
    }
}

template<typename T>
inline void LOGGER(VERBOSITY_LEVEL vl, T x) {
    if (vl <= GLOBAL_VERBOSITY_LEVEL) {
        std::cout << x << std::endl;
    }
}

template<typename T>
inline void LOGGER(T x, bool newline) {
    if (GLOBAL_VERBOSITY_LEVEL >= VERBOSITY_LEVEL_VERBOSE) {
        std::cout << x;
    }
}

}