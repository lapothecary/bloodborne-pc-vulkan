// Native C++20 zero-Python Windows starter for Bloodborne PC Vulkan (root Bloodborne.exe).
//
// Design:
//   - Steady-state launches (< 5 ms, 0% Python): verifies out/launch_env.txt & out/boot-linked.bin
//     are up to date, opens logs/<timestamp>.log, spawns bin/Bloodborne.exe directly with
//     CREATE_NO_WINDOW, and exits immediately so only ONE Bloodborne.exe process runs.
//   - First launch / changed config, mods, or patches: runs bin/bb-helper.exe --prepare-only,
//     waits for preparation to finish and exit, then launches bin/Bloodborne.exe and exits.
//   - Backward compatibility: forwards `--run-script ...` transparently to bin/bb-helper.exe.

#ifndef _WIN32
int main() { return 0; }
#else

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <ctime>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::wstring Utf8ToWide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return {};
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

std::wstring QuoteArg(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;
    }
    std::wstring out = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
        } else if (c == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(L'"');
            backslashes = 0;
        } else {
            if (backslashes > 0) {
                out.append(backslashes, L'\\');
                backslashes = 0;
            }
            out.push_back(c);
        }
    }
    if (backslashes > 0) {
        out.append(backslashes * 2, L'\\');
    }
    out.push_back(L'"');
    return out;
}

std::wstring BuildCmdLine(const std::vector<std::wstring>& args) {
    std::wstring cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) cmd.push_back(L' ');
        cmd += QuoteArg(args[i]);
    }
    return cmd;
}

struct LaunchManifest {
    std::wstring exe;
    std::wstring cwd;
    std::wstring boot;
    std::wstring content;
    std::wstring patches;
    std::wstring app0;
    std::wstring user;
    std::wstring timeout = L"0";
    std::vector<std::pair<std::wstring, std::wstring>> env;
};

bool ParseManifest(const fs::path& path, LaunchManifest& m) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("ENV:", 0) == 0) {
            size_t eq = line.find('=', 4);
            if (eq != std::string::npos) {
                std::string k = line.substr(4, eq - 4);
                std::string v = line.substr(eq + 1);
                if (!k.empty()) {
                    m.env.emplace_back(Utf8ToWide(k), Utf8ToWide(v));
                }
            }
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        if (k == "EXE") m.exe = Utf8ToWide(v);
        else if (k == "CWD") m.cwd = Utf8ToWide(v);
        else if (k == "BOOT") m.boot = Utf8ToWide(v);
        else if (k == "CONTENT") m.content = Utf8ToWide(v);
        else if (k == "PATCHES") m.patches = Utf8ToWide(v);
        else if (k == "APP0") m.app0 = Utf8ToWide(v);
        else if (k == "USER") m.user = Utf8ToWide(v);
        else if (k == "TIMEOUT") m.timeout = Utf8ToWide(v);
    }
    return !m.exe.empty() && !m.boot.empty() && !m.content.empty() &&
           !m.patches.empty() && !m.app0.empty() && !m.user.empty();
}

bool FileNewerThan(const fs::path& p, fs::file_time_type ref_time) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    auto t = fs::last_write_time(p, ec);
    return !ec && t > ref_time;
}

bool EnvEquals(const wchar_t* name, const wchar_t* expected) {
    wchar_t buf[64]{};
    DWORD n = GetEnvironmentVariableW(name, buf, 64);
    if (n == 0 || n >= 64) return false;
    return wcscmp(buf, expected) == 0;
}

bool IniRequestsOnline(const fs::path& ini_path) {
    std::ifstream f(ini_path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return std::string{};
            size_t b = s.find_last_not_of(" \t\r\n");
            return s.substr(a, b - a + 1);
        };
        k = trim(k);
        v = trim(v);
        if (k == "online_pool" && !v.empty() && v != "offline") return true;
    }
    return false;
}

bool NeedsPrepare(const fs::path& pkg_dir, const fs::path& manifest_path, LaunchManifest& m) {
    std::error_code ec;
    if (!fs::is_regular_file(manifest_path, ec)) return true;
    if (!ParseManifest(manifest_path, m)) return true;

    if (!fs::is_regular_file(fs::path(m.exe), ec)) return true;
    if (!fs::is_regular_file(fs::path(m.boot), ec)) return true;
    if (!fs::is_regular_file(fs::path(m.content), ec)) return true;
    if (!fs::is_regular_file(fs::path(m.patches), ec)) return true;
    if (!fs::is_regular_file(fs::path(m.app0) / L"eboot.bin", ec)) return true;

    if (EnvEquals(L"BB_ONLINE", L"1") || IniRequestsOnline(pkg_dir / L"bbport.ini")) {
        return true;
    }

    auto ref_time = fs::last_write_time(manifest_path, ec);
    if (ec) return true;

    const fs::path watch_files[] = {
        pkg_dir / L"bbport.ini",
        pkg_dir / L"patches.json",
        pkg_dir / L"mods.json",
        pkg_dir / L"launcher-win.json",
        pkg_dir / L"test-env.txt",
        pkg_dir / L"scripts" / L"patches.py",
        pkg_dir / L"scripts" / L"prepare.py",
        pkg_dir / L"scripts" / L"mods.py",
    };
    for (const auto& wf : watch_files) {
        if (FileNewerThan(wf, ref_time)) return true;
    }

    const fs::path mods_dir = pkg_dir / L"mods";
    if (fs::is_directory(mods_dir, ec)) {
        if (FileNewerThan(mods_dir, ref_time)) return true;
        for (const auto& entry : fs::directory_iterator(mods_dir, ec)) {
            if (FileNewerThan(entry.path(), ref_time)) return true;
        }
    }
    return false;
}

int RunProcessAndWait(const fs::path& exe,
                      const std::vector<std::wstring>& args,
                      const fs::path& cwd,
                      bool inherit_stdio) {
    std::wstring cmdline = BuildCmdLine(args);
    std::vector<wchar_t> cmd_buf(cmdline.begin(), cmdline.end());
    cmd_buf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    const DWORD flags = inherit_stdio ? 0 : CREATE_NO_WINDOW;
    if (!CreateProcessW(
            exe.wstring().c_str(),
            cmd_buf.data(),
            nullptr,
            nullptr,
            inherit_stdio ? TRUE : FALSE,
            flags,
            nullptr,
            cwd.wstring().c_str(),
            &si,
            &pi)) {
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(exit_code);
}

HANDLE OpenTimestampedLog(const fs::path& data_dir) {
    std::error_code ec;
    fs::path logs_dir = data_dir / L"logs";
    fs::create_directories(logs_dir, ec);

    std::time_t now = std::time(nullptr);
    std::tm tm_now{};
    localtime_s(&tm_now, &now);
    wchar_t name[64]{};
    swprintf_s(name, L"%04d%02d%02d_%02d%02d%02d.log",
               tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday,
               tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec);

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    fs::path log_path = logs_dir / name;
    HANDLE h = CreateFileW(
        log_path.wstring().c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        &sa,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    return h;
}

}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    wchar_t exe_buf[MAX_PATH * 4]{};
    DWORD n = GetModuleFileNameW(nullptr, exe_buf, static_cast<DWORD>(std::size(exe_buf)));
    if (n == 0) return 1;
    const fs::path pkg_dir = fs::path(exe_buf).parent_path();
    const fs::path helper_exe = pkg_dir / L"bin" / L"bb-helper.exe";

    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::wstring> raw_args;
    if (argv) {
        for (int i = 1; i < argc; ++i) {
            raw_args.emplace_back(argv[i]);
        }
        LocalFree(argv);
    }

    // Legacy compatibility: forward `--run-script ...` straight to bin/bb-helper.exe.
    if (!raw_args.empty() && raw_args[0] == L"--run-script") {
        std::vector<std::wstring> fwd;
        fwd.push_back(helper_exe.wstring());
        for (const auto& a : raw_args) fwd.push_back(a);
        return RunProcessAndWait(helper_exe, fwd, pkg_dir, true);
    }

    // Special online mode with local Python BBSC server: let bb-helper manage its lifecycle.
    if (EnvEquals(L"BB_ONLINE", L"1") && EnvEquals(L"BB_BBSC", L"1")) {
        std::vector<std::wstring> fwd;
        fwd.push_back(helper_exe.wstring());
        for (const auto& a : raw_args) {
            if (a != L"--play") fwd.push_back(a);
        }
        return RunProcessAndWait(helper_exe, fwd, pkg_dir, false);
    }

    std::error_code ec;
    fs::path data_dir = pkg_dir;
    wchar_t data_env[MAX_PATH * 4]{};
    if (GetEnvironmentVariableW(L"BB_DATA_DIR", data_env, static_cast<DWORD>(std::size(data_env))) > 0) {
        data_dir = fs::path(data_env);
    }
    const fs::path manifest_path = data_dir / L"out" / L"launch_env.txt";

    LaunchManifest manifest;
    if (NeedsPrepare(pkg_dir, manifest_path, manifest)) {
        if (!fs::is_regular_file(helper_exe, ec)) {
            MessageBoxW(nullptr,
                        L"Missing bin\\bb-helper.exe required for first-time game preparation.",
                        L"Bloodborne Vulkan Remaster",
                        MB_OK | MB_ICONERROR);
            return 1;
        }
        const int prep_rc = RunProcessAndWait(
            helper_exe,
            {helper_exe.wstring(), L"--prepare-only"},
            pkg_dir,
            false);
        if (prep_rc != 0) {
            return prep_rc;
        }
        manifest = LaunchManifest{};
        if (!ParseManifest(manifest_path, manifest)) {
            MessageBoxW(nullptr,
                        L"Game preparation finished without generating out\\launch_env.txt.\n"
                        L"Please check logs\\ or open Launcher.exe to select your game folder.",
                        L"Bloodborne Vulkan Remaster",
                        MB_OK | MB_ICONERROR);
            return 1;
        }
    }

    // Apply prepared BB_* / driver environment variables (preserving explicit user overrides).
    for (const auto& kv : manifest.env) {
        if (GetEnvironmentVariableW(kv.first.c_str(), nullptr, 0) == 0) {
            SetEnvironmentVariableW(kv.first.c_str(), kv.second.c_str());
        }
    }

    std::vector<std::wstring> engine_args = {
        manifest.exe,
        manifest.boot,
        L"--content-profile", manifest.content,
        L"--patches", manifest.patches,
        L"--app0", manifest.app0,
        L"--user", manifest.user,
        L"--timeout", manifest.timeout,
    };
    for (const auto& a : raw_args) {
        if (a != L"--play" && a != L"--prepare-only") {
            engine_args.push_back(a);
        }
    }

    std::wstring cmdline = BuildCmdLine(engine_args);
    std::vector<wchar_t> cmd_buf(cmdline.begin(), cmdline.end());
    cmd_buf.push_back(L'\0');

    HANDLE hLog = OpenTimestampedLog(data_dir);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    BOOL inherit = FALSE;
    if (hLog != INVALID_HANDLE_VALUE) {
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = hLog;
        si.hStdError = hLog;
        si.hStdInput = INVALID_HANDLE_VALUE;
        inherit = TRUE;
    }

    PROCESS_INFORMATION pi{};
    const std::wstring cwd = manifest.cwd.empty() ? data_dir.wstring() : manifest.cwd;
    if (!CreateProcessW(
            manifest.exe.c_str(),
            cmd_buf.data(),
            nullptr,
            nullptr,
            inherit,
            CREATE_NO_WINDOW,
            nullptr,
            cwd.c_str(),
            &si,
            &pi)) {
        if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
        MessageBoxW(nullptr,
                    L"Failed to launch bin\\Bloodborne.exe.",
                    L"Bloodborne Vulkan Remaster",
                    MB_OK | MB_ICONERROR);
        return 1;
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
    // Exit immediately: only bin\Bloodborne.exe remains in Task Manager (0% Python in RAM).
    return 0;
}

#endif
