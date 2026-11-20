// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: see log.h.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>

#include "common/logging/log.h"

namespace Common::Log {
namespace {

constexpr const char* LevelNames[] = {"trace", "debug", "info", "warning", "error", "critical"};

Level Threshold() {
    static const Level threshold = [] {
        const char* env = std::getenv("BB_GPU_LOG");
        if (env) {
            for (int i = 0; i < 6; ++i) {
                if (std::strcmp(env, LevelNames[i]) == 0) {
                    return Level(i);
                }
            }
        }
        return Level::warn;
    }();
    return threshold;
}

bool ExternalTimestamps() {
    static const bool ext = [] {
        const char* env = std::getenv("BB_LOG_TIMESTAMPS");
        return env && env[0] == '1';
    }();
    return ext;
}

const std::chrono::steady_clock::time_point g_log_start = std::chrono::steady_clock::now();

} // namespace

bool ShouldLog(Level level) {
    return level >= Threshold();
}

void Write(std::string_view log_class, Level level, const char* file, int line,
           std::string&& message) {
    static std::mutex mutex;
    const char* slash = std::strrchr(file, '/');
    const int index = int(level) >= 0 && int(level) < 6 ? int(level) : 5;
    std::scoped_lock lock{mutex};
    if (!ExternalTimestamps()) {
        const auto sys_now = std::chrono::system_clock::now();
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()) %
            1000;
        const std::time_t tt = std::chrono::system_clock::to_time_t(sys_now);
        std::tm tm_now{};
#ifdef _WIN32
        localtime_s(&tm_now, &tt);
#else
        localtime_r(&tt, &tm_now);
#endif
        const double elapsed_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - g_log_start).count();
        std::fprintf(stderr, "[%02d:%02d:%02d.%03d +%06.3fs] gpu %s (%.*s): %s [%s:%d]\n",
                     tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec, static_cast<int>(ms.count()),
                     elapsed_s, LevelNames[index], int(log_class.size()), log_class.data(),
                     message.c_str(), slash ? slash + 1 : file, line);
    } else {
        std::fprintf(stderr, "gpu %s (%.*s): %s [%s:%d]\n", LevelNames[index], int(log_class.size()),
                     log_class.data(), message.c_str(), slash ? slash + 1 : file, line);
    }
}

} // namespace Common::Log
