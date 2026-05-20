// control verbosity levels in the code to make things cleaner

#pragma once

#include <cstdlib>
#include <sstream>
#include <iostream>

namespace data_structures {

enum VERBOSITY_LEVEL {
  VERBOSITY_LEVEL_QUIET = 0,
  VERBOSITY_LEVEL_SEQUENCES = 1,
  VERBOSITY_LEVEL_VERBOSE = 2,
  VERBOSITY_LEVEL_DEBUG = 3,
  VERBOSITY_LEVEL_TRACE = 4
};

inline VERBOSITY_LEVEL get_verbosity_from_env() {
    const char* env_v = std::getenv("MIDIGPT_VERBOSITY");
    if (env_v) {
        int v = std::atoi(env_v);
        if (v <= 0) return VERBOSITY_LEVEL_QUIET;
        if (v == 1) return VERBOSITY_LEVEL_SEQUENCES;
        if (v == 2) return VERBOSITY_LEVEL_VERBOSE;
        if (v == 3) return VERBOSITY_LEVEL_DEBUG;
        return VERBOSITY_LEVEL_TRACE;
    }
    return VERBOSITY_LEVEL_TRACE; // Default
}

inline VERBOSITY_LEVEL GLOBAL_VERBOSITY_LEVEL = get_verbosity_from_env();
inline int GLOBAL_LOGGER_INDENT = 0;

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