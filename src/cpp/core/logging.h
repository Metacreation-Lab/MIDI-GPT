#pragma once

#include <iostream>
#include <string>

// Windows headers (wingdi.h, winbase.h, dbgapi.h) define these as macros and
// collide with our scoped enum below. Undefine if present.
#ifdef ERROR
#undef ERROR
#endif
#ifdef TRACE
#undef TRACE
#endif
#ifdef DEBUG
#undef DEBUG
#endif
#ifdef WARNING
#undef WARNING
#endif

namespace midigpt {

enum class LogLevel {
    OFF = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5
};

class Logger {
public:
    static void set_level(LogLevel level) {
        get_instance().level_ = level;
    }

    static LogLevel get_level() {
        return get_instance().level_;
    }

    static void log(LogLevel level, const std::string& message) {
        if (level <= get_instance().level_ && level != LogLevel::OFF) {
            std::ostream& os = (level <= LogLevel::WARNING) ? std::cerr : std::cout;
            os << "[" << level_to_string(level) << "] " << message << std::endl;
        }
    }

private:
    Logger() : level_(LogLevel::WARNING) {}
    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }

    static std::string level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::ERROR:   return "ERROR";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::INFO:    return "INFO";
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::TRACE:   return "TRACE";
            default:                return "UNKNOWN";
        }
    }

    LogLevel level_;
};

#define LOG_ERROR(msg)   Logger::log(LogLevel::ERROR,   msg)
#define LOG_WARNING(msg) Logger::log(LogLevel::WARNING, msg)
#define LOG_INFO(msg)    Logger::log(LogLevel::INFO,    msg)
#define LOG_DEBUG(msg)   Logger::log(LogLevel::DEBUG,   msg)
#define LOG_TRACE(msg)   Logger::log(LogLevel::TRACE,   msg)

} // namespace midigpt
