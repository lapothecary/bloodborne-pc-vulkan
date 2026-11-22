// SPDX-License-Identifier: GPL-2.0-or-later
// Cross-platform (Linux + Windows) privacy-safe hardware/OS passport and live RAM/CPU sensors.
// Strictly ZERO PII: never logs usernames, hostnames, serials, MAC/IP, or unrelated processes.

#pragma once

#include <string>
#include <string_view>

namespace BbSysInfo {

struct LiveSysSensors {
    bool has_ram = false;
    float ram_proc_gb = 0.0f;
    float ram_sys_used_gb = 0.0f;
    float ram_sys_total_gb = 0.0f;
    int proc_cpu_pct = -1; // Process CPU % across all logical threads (0..100, matches System Monitor)
    int sys_cpu_pct = -1;  // Total system CPU % across all logical threads (0..100)
    int cpu_clock_mhz = 0; // Average active CPU clock in MHz (0 if unavailable)
    int cpu_temp_c = 0;    // Package/Tctl CPU temperature in C (0 if unavailable)
};

/// Replaces the current user's home directory prefix ($HOME on Linux/macOS, %USERPROFILE% on
/// Windows) with "~" or "%USERPROFILE%" so log files never leak the OS account name.
std::string RedactHomePath(std::string_view path);

/// Logs a compact, privacy-safe 4-line System/OS/CPU/RAM/Settings passport once per process.
void LogSystemPassport();

/// Samples live process/system RAM, process/system CPU %, CPU frequency, and CPU temp.
/// Intended to run on the background telemetry thread (never on Present / Main threads).
void PollLiveSystemSensors(LiveSysSensors& out);

} // namespace BbSysInfo
