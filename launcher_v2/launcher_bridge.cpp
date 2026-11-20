#include "launcher_bridge.h"

#include "fsb_ui_sfx.h"
#include "gfx_font_extract.h"
#include "tpf_hero_art.h"

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace bb_launcher {
namespace {

namespace fs = std::filesystem;

std::string Trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::string ExtractJsonField(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return {};
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size()) return {};
    if (json[pos] == '"') {
        ++pos;
        std::string out;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                ++pos;
                switch (json[pos]) {
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(json[pos]); break;
                }
            } else {
                out.push_back(json[pos]);
            }
            ++pos;
        }
        return out;
    }
    size_t end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' &&
           !std::isspace(static_cast<unsigned char>(json[end]))) {
        ++end;
    }
    return json.substr(pos, end - pos);
}

std::string ExtractKvToken(const std::string& line, const std::string& key) {
    const std::string pat = key + "=";
    size_t pos = line.find(pat);
    if (pos == std::string::npos) return {};
    pos += pat.size();
    size_t end = pos;
    while (end < line.size() && !std::isspace(static_cast<unsigned char>(line[end]))) ++end;
    return line.substr(pos, end - pos);
}

#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& s) {
    if (s.empty()) return {};
    int need = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (need <= 0) return {};
    std::wstring w(static_cast<size_t>(need), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), need);
    return w;
}

std::wstring QuoteWinArg(const std::wstring& a) {
    if (a.empty()) return L"\"\"";
    if (a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring out = L"\"";
    for (wchar_t c : a) {
        if (c == L'"') out += L"\\\"";
        else out.push_back(c);
    }
    out += L"\"";
    return out;
}

void SpawnDetached(const std::vector<std::string>& argv, const fs::path& cwd) {
    if (argv.empty()) return;
    std::wstring cmd;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i > 0) cmd += L' ';
        cmd += QuoteWinArg(Utf8ToWide(argv[i]));
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::wstring cwd_w = cwd.wstring();
    if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                       DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, nullptr,
                       cwd_w.empty() ? nullptr : cwd_w.c_str(), &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
}

int RunProcessStreaming(const std::vector<std::string>& argv,
                        const fs::path& cwd,
                        std::atomic<int>* pid_out,
                        const std::function<void(const std::string&)>& on_line) {
    if (argv.empty()) return -1;
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE read_h = nullptr, write_h = nullptr;
    if (!CreatePipe(&read_h, &write_h, &sa, 0)) return -1;
    SetHandleInformation(read_h, HANDLE_FLAG_INHERIT, 0);

    std::wstring cmd;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i > 0) cmd += L' ';
        cmd += QuoteWinArg(Utf8ToWide(argv[i]));
    }

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_h;
    si.hStdError = write_h;
    si.hStdInput = nullptr;

    PROCESS_INFORMATION pi{};
    std::wstring cwd_w = cwd.wstring();
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr,
                             cwd_w.empty() ? nullptr : cwd_w.c_str(), &si, &pi);
    CloseHandle(write_h);
    if (!ok) {
        CloseHandle(read_h);
        return -1;
    }

    if (pid_out) pid_out->store(static_cast<int>(pi.dwProcessId));

    std::string pending;
    char buf[1024];
    DWORD got = 0;
    while (ReadFile(read_h, buf, sizeof(buf), &got, nullptr) && got > 0) {
        pending.append(buf, got);
        size_t nl = 0;
        while ((nl = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, nl);
            pending.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (on_line) on_line(line);
        }
    }
    if (!pending.empty()) {
        if (pending.back() == '\r') pending.pop_back();
        if (on_line) on_line(pending);
    }
    CloseHandle(read_h);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (pid_out) pid_out->store(-1);
    return static_cast<int>(exit_code);
}
#else
void SpawnDetached(const std::vector<std::string>& argv, const fs::path& cwd) {
    if (argv.empty()) return;
    pid_t pid = fork();
    if (pid == 0) {
        setsid();
        if (!cwd.empty()) (void)chdir(cwd.c_str());
        std::vector<char*> cargs;
        cargs.reserve(argv.size() + 1);
        for (const auto& a : argv) cargs.push_back(const_cast<char*>(a.c_str()));
        cargs.push_back(nullptr);
        execvp(cargs[0], cargs.data());
        _exit(127);
    }
}

int RunProcessStreaming(const std::vector<std::string>& argv,
                        const fs::path& cwd,
                        std::atomic<int>* pid_out,
                        const std::function<void(const std::string&)>& on_line) {
    int pipefd[2]{-1, -1};
    if (pipe(pipefd) != 0) return -1;

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        if (!cwd.empty()) (void)chdir(cwd.c_str());
        std::vector<char*> cargs;
        cargs.reserve(argv.size() + 1);
        for (const auto& a : argv) cargs.push_back(const_cast<char*>(a.c_str()));
        cargs.push_back(nullptr);
        execvp(cargs[0], cargs.data());
        _exit(127);
    }

    close(pipefd[1]);
    if (pid_out) pid_out->store(static_cast<int>(pid));

    FILE* fp = fdopen(pipefd[0], "r");
    if (fp) {
        char buf[1024];
        while (std::fgets(buf, sizeof(buf), fp)) {
            std::string line(buf);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (on_line) on_line(line);
        }
        std::fclose(fp);
    } else {
        close(pipefd[0]);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (pid_out) pid_out->store(-1);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

}  // namespace

std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

LauncherBridge::LauncherBridge(LauncherContext ctx) : ctx_(std::move(ctx)) {}

LauncherBridge::~LauncherBridge() {
    CancelShaderBake();
    if (bake_thread_.joinable()) bake_thread_.join();
    if (upd_thread_.joinable()) upd_thread_.join();
}

bool LauncherBridge::IsValidGameDir(const fs::path& p) const {
    if (p.empty()) return false;
    std::error_code ec;
    return fs::is_regular_file(p / "eboot.bin", ec) &&
           fs::is_directory(p / "dvdroot_ps4", ec);
}

fs::path LauncherBridge::ResolveGameDir() const {
    const std::string saved = ReadIniValue("game_dir");
    if (!saved.empty() && IsValidGameDir(saved)) return fs::path(saved);

    if (const char* env = std::getenv("BB_GAME_DIR")) {
        if (*env && IsValidGameDir(env)) return fs::path(env);
    }

    const fs::path parent = ctx_.package_dir.parent_path();
    const std::vector<fs::path> candidates = {
        ctx_.package_dir / "CUSA03173",
        ctx_.package_dir / "CUSA00900",
        parent / "CUSA03173",
        parent / "CUSA00900",
        parent / "CUSA00207",
    };
    for (const auto& c : candidates) {
        if (IsValidGameDir(c)) return c;
    }
    return saved.empty() ? fs::path{} : fs::path(saved);
}

fs::path LauncherBridge::ResolveUserDir() const {
    if (const char* env = std::getenv("BB_USER_DIR")) {
        if (*env) return fs::path(env);
    }
    return ctx_.package_dir / "user";
}

std::string LauncherBridge::ReadIniValue(const std::string& key, const std::string& fallback) const {
    const fs::path ini = ctx_.package_dir / "bbport.ini";
    std::ifstream f(ini);
    if (!f) return fallback;
    std::string line;
    while (std::getline(f, line)) {
        std::string t = Trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';' || t[0] == '[') continue;
        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        std::string k = Trim(t.substr(0, eq));
        if (k == key) return Trim(t.substr(eq + 1));
    }
    return fallback;
}

void LauncherBridge::WriteIniValue(const std::string& key, const std::string& value) const {
    const fs::path ini = ctx_.package_dir / "bbport.ini";
    std::vector<std::string> lines;
    bool updated = false;
    {
        std::ifstream f(ini);
        std::string line;
        while (std::getline(f, line)) {
            std::string t = Trim(line);
            size_t eq = t.find('=');
            if (!t.empty() && t[0] != '#' && t[0] != ';' && t[0] != '[' && eq != std::string::npos) {
                std::string k = Trim(t.substr(0, eq));
                if (k == key) {
                    lines.push_back(key + " = " + value);
                    updated = true;
                    continue;
                }
            }
            lines.push_back(line);
        }
    }
    if (!updated) lines.push_back(key + " = " + value);
    std::ofstream out(ini, std::ios::trunc);
    for (const auto& l : lines) out << l << "\n";
}

int LauncherBridge::CountCachedShaders(const fs::path& user_dir) const {
    const fs::path cache_root = user_dir / "cache";
    std::error_code ec;
    if (!fs::is_directory(cache_root, ec)) return 0;
    int base_spv = 0;
    for (const auto& entry : fs::recursive_directory_iterator(cache_root, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        if (entry.path().extension() != ".spv") continue;
        const std::string fn = entry.path().filename().string();
        if (fn.rfind("opt", 0) == 0) continue;
        ++base_spv;
    }
    return (std::min)(base_spv, 5761);
}

std::string LauncherBridge::ReadVersionTag() const {
    const fs::path vf = ctx_.package_dir / ".bbport-version.json";
    std::ifstream f(vf);
    if (!f) return "v1.1.9";
    std::stringstream ss;
    ss << f.rdbuf();
    std::string tag = ExtractJsonField(ss.str(), "tag");
    return tag.empty() ? "v1.1.9" : tag;
}

std::string LauncherBridge::BuildModsJson() const {
    const fs::path mods_dir = ctx_.package_dir / "mods";
    std::error_code ec;
    std::vector<std::pair<std::string, bool>> mods;
    if (fs::is_directory(mods_dir, ec)) {
        for (const auto& entry : fs::directory_iterator(mods_dir, ec)) {
            if (!entry.is_directory(ec)) continue;
            std::string name = entry.path().filename().string();
            if (name.empty() || name[0] == '.') continue;
            bool disabled = (name.size() > 9 && name.substr(name.size() - 9) == ".disabled");
            std::string display = disabled ? name.substr(0, name.size() - 9) : name;
            mods.emplace_back(display, !disabled);
        }
    }
    std::sort(mods.begin(), mods.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string out = "[";
    for (size_t i = 0; i < mods.size(); ++i) {
        if (i > 0) out += ",";
        out += "{\"name\":\"" + JsonEscape(mods[i].first) + "\",\"enabled\":" +
               (mods[i].second ? "true" : "false") + "}";
    }
    out += "]";
    return out;
}

void LauncherBridge::ToggleMod(const std::string& mod_name, bool enabled) const {
    if (mod_name.empty() || mod_name.find('/') != std::string::npos ||
        mod_name.find('\\') != std::string::npos || mod_name.find("..") != std::string::npos) {
        return;
    }
    const fs::path mods_dir = ctx_.package_dir / "mods";
    const fs::path active = mods_dir / mod_name;
    const fs::path inactive = mods_dir / (mod_name + ".disabled");
    std::error_code ec;
    if (enabled && fs::is_directory(inactive, ec)) {
        fs::rename(inactive, active, ec);
    } else if (!enabled && fs::is_directory(active, ec)) {
        fs::rename(active, inactive, ec);
    }
}

bool LauncherBridge::ClearShaderCache() const {
    const fs::path cache_root = ResolveUserDir() / "cache";
    std::error_code ec;
    if (!fs::is_directory(cache_root, ec)) return true;
    for (const auto& entry : fs::directory_iterator(cache_root, ec)) {
        const auto fn = entry.path().filename();
        if (fn == "ui_sfx" || fn == "ui_assets" || fn == "webview2") continue;
        fs::remove_all(entry.path(), ec);
    }
    return true;
}

bool LauncherBridge::ResetGraphicsSafeMode() const {
    WriteIniValue("output_res", "auto");
    WriteIniValue("window_mode", "borderless");
    WriteIniValue("upscaler", "off");
    WriteIniValue("aa_mode", "off");
    WriteIniValue("fps_limit", "60");
    WriteIniValue("ssao_quality", "0");
    WriteIniValue("ssr_quality", "0");
    WriteIniValue("contact_shadows", "0");
    WriteIniValue("true_black_hdr", "0");
    WriteIniValue("ui_upscale", "1");
    return true;
}

bool LauncherBridge::OpenSystemFolder(const std::string& which) const {
    fs::path target = ctx_.package_dir;
    if (which == "mods") target = ctx_.package_dir / "mods";
    else if (which == "saves") target = ResolveUserDir() / "savedata";
    else if (which == "logs") target = ctx_.package_dir / "logs";
    std::error_code ec;
    fs::create_directories(target, ec);
#ifdef _WIN32
    ShellExecuteW(nullptr, L"open", target.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    SpawnDetached({"xdg-open", target.string()}, ctx_.package_dir);
#endif
    return true;
}

bool LauncherBridge::LaunchGameProcess() const {
#ifdef _WIN32
    const fs::path exe = ctx_.package_dir / "Bloodborne.exe";
    auto u8 = exe.u8string();
    std::string exe_utf8(reinterpret_cast<const char*>(u8.data()), u8.size());
    SpawnDetached({exe_utf8}, ctx_.package_dir);
#else
    const fs::path sh = ctx_.package_dir / "play.sh";
    SpawnDetached({"/bin/bash", sh.string()}, ctx_.package_dir);
#endif
    return true;
}

void LauncherBridge::StartShaderBake() {
    {
        std::lock_guard<std::mutex> lk(bake_mu_);
        if (bake_.running) return;
        bake_ = BakeProgress{};
        bake_.running = true;
        bake_.phase = "scanning";
    }
    if (bake_thread_.joinable()) bake_thread_.join();

    const fs::path game_dir = ResolveGameDir();
    const fs::path user_dir = ResolveUserDir();
    const fs::path pkg_dir = ctx_.package_dir;

    bake_thread_ = std::thread([this, game_dir, user_dir, pkg_dir]() {
        std::error_code ec;
        if (!IsValidGameDir(game_dir)) {
            std::lock_guard<std::mutex> lk(bake_mu_);
            bake_.running = false;
            bake_.error = "Game folder with eboot.bin not configured";
            return;
        }
        auto to_utf8 = [](const fs::path& p) -> std::string {
            auto u8 = p.u8string();
            return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
        };
#ifdef _WIN32
        fs::path probe = pkg_dir / "bin" / "Bloodborne.exe";
        if (!fs::is_regular_file(probe, ec)) probe = pkg_dir / "bin" / "bb-probe.exe";
        if (!fs::is_regular_file(probe, ec)) probe = pkg_dir / "out" / "bb-probe.exe";
#else
        fs::path probe = pkg_dir / "bin" / "bb-vulkan";
        if (!fs::is_regular_file(probe, ec)) probe = pkg_dir / "out" / "bb-vulkan";
        if (!fs::is_regular_file(probe, ec)) probe = pkg_dir / "bin" / "bb-probe";
        if (!fs::is_regular_file(probe, ec)) probe = pkg_dir / "out" / "bb-probe";
#endif
        if (!fs::is_regular_file(probe, ec)) {
            std::lock_guard<std::mutex> lk(bake_mu_);
            bake_.running = false;
            bake_.error = "Engine binary not found in bin/";
            return;
        }

        const int rc = RunProcessStreaming(
            {to_utf8(probe), "--bake-shaders", to_utf8(game_dir), to_utf8(user_dir)},
            pkg_dir, &bake_pid_, [this](const std::string& line) {
                if (line.find("[BB-BAKE] PROGRESS") != std::string::npos) {
                    std::lock_guard<std::mutex> lk(bake_mu_);
                    auto s_pct = ExtractKvToken(line, "pct");
                    auto s_done = ExtractKvToken(line, "done");
                    auto s_tot = ExtractKvToken(line, "total");
                    auto s_ph = ExtractKvToken(line, "phase");
                    if (!s_pct.empty()) bake_.pct = std::atoi(s_pct.c_str());
                    if (!s_done.empty()) bake_.done = std::atoi(s_done.c_str());
                    if (!s_tot.empty()) bake_.total = std::atoi(s_tot.c_str());
                    if (!s_ph.empty()) bake_.phase = s_ph;
                } else if (line.find("[BB-BAKE] DONE") != std::string::npos) {
                    std::lock_guard<std::mutex> lk(bake_mu_);
                    bake_.pct = 100;
                    bake_.finished = true;
                    bake_.phase = "done";
                    bake_.elapsed = ExtractKvToken(line, "elapsed_s");
                } else if (line.find("[BB-BAKE] ERROR:") != std::string::npos) {
                    std::lock_guard<std::mutex> lk(bake_mu_);
                    bake_.error = line;
                }
            });

        std::lock_guard<std::mutex> lk(bake_mu_);
        bake_.running = false;
        if (!bake_.cancelled && rc == 0) {
            bake_.pct = 100;
            bake_.finished = true;
        } else if (!bake_.cancelled && bake_.error.empty()) {
            bake_.error = "Shader baker exited with code " + std::to_string(rc);
        }
    });
}

void LauncherBridge::CancelShaderBake() {
    const int pid = bake_pid_.load();
    if (pid > 0) {
        {
            std::lock_guard<std::mutex> lk(bake_mu_);
            bake_.cancelled = true;
        }
#ifdef _WIN32
        if (HANDLE hp = OpenProcess(PROCESS_TERMINATE, FALSE, static_cast<DWORD>(pid))) {
            TerminateProcess(hp, 1);
            CloseHandle(hp);
        }
#else
        kill(static_cast<pid_t>(pid), SIGTERM);
#endif
    }
}

void LauncherBridge::StartUpdateCheck() {
    {
        std::lock_guard<std::mutex> lk(upd_mu_);
        if (upd_.checking || upd_.applying) return;
        upd_.checking = true;
        upd_.error.clear();
    }
    if (upd_thread_.joinable()) upd_thread_.join();

    const fs::path pkg_dir = ctx_.package_dir;
    const fs::path script = pkg_dir / "launcher" / "bbport_updater.py";

    upd_thread_ = std::thread([this, pkg_dir, script]() {
        auto to_utf8 = [](const fs::path& p) -> std::string {
            auto u8 = p.u8string();
            return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
        };
        std::vector<std::string> cmd;
#ifdef _WIN32
        std::error_code ec;
        fs::path runner = pkg_dir / "bin" / "bb-helper.exe";
        if (!fs::is_regular_file(runner, ec)) runner = pkg_dir / "Bloodborne.exe";
        if (fs::is_regular_file(runner, ec)) {
            cmd = {to_utf8(runner), "--run-script", to_utf8(script), "--check-json", to_utf8(pkg_dir)};
        } else {
            cmd = {"python", to_utf8(script), "--check-json", to_utf8(pkg_dir)};
        }
#else
        cmd = {"python3", to_utf8(script), "--check-json", to_utf8(pkg_dir)};
#endif
        std::string last_json;
        const int rc = RunProcessStreaming(
            cmd, pkg_dir, nullptr, [&](const std::string& line) {
                if (!line.empty() && line.front() == '{') last_json = line;
            });
        std::lock_guard<std::mutex> lk(upd_mu_);
        upd_.checking = false;
        upd_.checked = true;
        if (rc == 0 && !last_json.empty()) {
            upd_.ok = ExtractJsonField(last_json, "ok") == "true";
            upd_.up_to_date = ExtractJsonField(last_json, "up_to_date") == "true";
            upd_.remote_tag = ExtractJsonField(last_json, "tag");
            if (upd_.remote_tag.empty()) upd_.remote_tag = ExtractJsonField(last_json, "remote_tag");
            upd_.local_tag = ExtractJsonField(last_json, "local_tag");
            if (upd_.local_tag.empty()) upd_.local_tag = ReadVersionTag();
            upd_.error = ExtractJsonField(last_json, "error");
        } else {
            upd_.ok = false;
            upd_.error = "Could not reach GitHub Releases";
        }
    });
}

void LauncherBridge::StartUpdateApply() {
    {
        std::lock_guard<std::mutex> lk(upd_mu_);
        if (upd_.applying || upd_.checking) return;
        upd_.applying = true;
        upd_.pct = 0;
        upd_.stage = "download";
        upd_.detail = "Starting update...";
        upd_.error.clear();
    }
    if (upd_thread_.joinable()) upd_thread_.join();

    const fs::path pkg_dir = ctx_.package_dir;
    const fs::path script = pkg_dir / "launcher" / "bbport_updater.py";

    upd_thread_ = std::thread([this, pkg_dir, script]() {
        auto to_utf8 = [](const fs::path& p) -> std::string {
            auto u8 = p.u8string();
            return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
        };
        std::vector<std::string> cmd;
#ifdef _WIN32
        std::error_code ec;
        fs::path runner = pkg_dir / "bin" / "bb-helper.exe";
        if (!fs::is_regular_file(runner, ec)) runner = pkg_dir / "Bloodborne.exe";
        if (fs::is_regular_file(runner, ec)) {
            cmd = {to_utf8(runner), "--run-script", to_utf8(script), "--apply-json", to_utf8(pkg_dir)};
        } else {
            cmd = {"python", to_utf8(script), "--apply-json", to_utf8(pkg_dir)};
        }
#else
        cmd = {"python3", to_utf8(script), "--apply-json", to_utf8(pkg_dir)};
#endif
        std::string result_json;
        const int rc = RunProcessStreaming(
            cmd, pkg_dir, nullptr, [&](const std::string& line) {
                if (line.rfind("[UPD] ", 0) == 0) {
                    std::string rest = line.substr(6);
                    size_t p1 = rest.find('|');
                    size_t p2 = (p1 != std::string::npos) ? rest.find('|', p1 + 1) : std::string::npos;
                    std::lock_guard<std::mutex> lk(upd_mu_);
                    if (p1 != std::string::npos) {
                        upd_.pct = std::atoi(rest.substr(0, p1).c_str());
                        upd_.stage = (p2 != std::string::npos) ? rest.substr(p1 + 1, p2 - p1 - 1)
                                                               : rest.substr(p1 + 1);
                        if (p2 != std::string::npos) upd_.detail = rest.substr(p2 + 1);
                    }
                } else if (line.rfind("[UPD-RESULT] ", 0) == 0) {
                    result_json = line.substr(13);
                }
            });
        std::lock_guard<std::mutex> lk(upd_mu_);
        upd_.applying = false;
        if (rc == 0 && ExtractJsonField(result_json, "ok") == "true") {
            upd_.pct = 100;
            upd_.up_to_date = true;
            upd_.restart_needed = true;
            upd_.stage = "done";
        } else {
            upd_.error = ExtractJsonField(result_json, "error");
            if (upd_.error.empty()) upd_.error = "Update failed";
        }
    });
}

std::string LauncherBridge::BuildBakeJson() const {
    std::lock_guard<std::mutex> lk(bake_mu_);
    std::ostringstream os;
    os << "{\"running\":" << (bake_.running ? "true" : "false")
       << ",\"finished\":" << (bake_.finished ? "true" : "false")
       << ",\"cancelled\":" << (bake_.cancelled ? "true" : "false")
       << ",\"pct\":" << bake_.pct
       << ",\"done\":" << bake_.done
       << ",\"total\":" << bake_.total
       << ",\"phase\":\"" << JsonEscape(bake_.phase) << "\""
       << ",\"elapsed\":\"" << JsonEscape(bake_.elapsed) << "\""
       << ",\"error\":\"" << JsonEscape(bake_.error) << "\"}";
    return os.str();
}

std::string LauncherBridge::BuildUpdateJson() const {
    std::lock_guard<std::mutex> lk(upd_mu_);
    std::ostringstream os;
    os << "{\"checking\":" << (upd_.checking ? "true" : "false")
       << ",\"applying\":" << (upd_.applying ? "true" : "false")
       << ",\"checked\":" << (upd_.checked ? "true" : "false")
       << ",\"ok\":" << (upd_.ok ? "true" : "false")
       << ",\"upToDate\":" << (upd_.up_to_date ? "true" : "false")
       << ",\"restartNeeded\":" << (upd_.restart_needed ? "true" : "false")
       << ",\"pct\":" << upd_.pct
       << ",\"remoteTag\":\"" << JsonEscape(upd_.remote_tag) << "\""
       << ",\"localTag\":\"" << JsonEscape(upd_.local_tag) << "\""
       << ",\"stage\":\"" << JsonEscape(upd_.stage) << "\""
       << ",\"detail\":\"" << JsonEscape(upd_.detail) << "\""
       << ",\"error\":\"" << JsonEscape(upd_.error) << "\"}";
    return os.str();
}

std::string LauncherBridge::BuildStateJson() const {
    const fs::path game_dir = ResolveGameDir();
    const bool valid = IsValidGameDir(game_dir);
    const fs::path user_dir = ResolveUserDir();
    const int shaders = CountCachedShaders(user_dir);

    std::string lang = ReadIniValue("launcher_lang", "en");
    if (lang != "ru" && lang != "en") lang = "en";
    const bool sfx_on = ReadIniValue("launcher_sfx", "1") != "0";
    std::string sfx_vol = ReadIniValue("launcher_sfx_vol", "0.30");
    const bool bgm_on = ReadIniValue("launcher_bgm", "1") != "0";
    std::string bgm_vol = ReadIniValue("launcher_bgm_vol", "0.28");
    std::string bgm_track = ReadIniValue("launcher_bgm_track", "hunters_dream");
    if (bgm_track != "hunters_dream" && bgm_track != "omen") bgm_track = "hunters_dream";

    std::string online_pool = ReadIniValue("online_pool", "offline");
    bool online_upnp = ReadIniValue("online_upnp", "1") != "0";

    std::ostringstream os;
    os << "{"
       << "\"gameDir\":\"" << JsonEscape(game_dir.string()) << "\","
       << "\"gameValid\":" << (valid ? "true" : "false") << ","
       << "\"version\":\"" << JsonEscape(ReadVersionTag()) << "\","
       << "\"lang\":\"" << JsonEscape(lang) << "\","
       << "\"sfxEnabled\":" << (sfx_on ? "true" : "false") << ","
       << "\"sfxVolume\":" << (sfx_vol.empty() ? "0.30" : sfx_vol) << ","
       << "\"bgmEnabled\":" << (bgm_on ? "true" : "false") << ","
       << "\"bgmVolume\":" << (bgm_vol.empty() ? "0.28" : bgm_vol) << ","
       << "\"bgmTrack\":\"" << JsonEscape(bgm_track) << "\","
       << "\"shaderCount\":" << shaders << ","
       << "\"shaderTotal\":5761,"
       << "\"onlinePool\":\"" << JsonEscape(online_pool) << "\","
       << "\"onlineUpnp\":" << (online_upnp ? "true" : "false") << ","
       << "\"mods\":" << BuildModsJson()
       << "}";
    return os.str();
}

std::string LauncherBridge::BuildSfxJson() const {
    const fs::path game_dir = ResolveGameDir();
    const fs::path cache_dir = ResolveUserDir() / "cache";
    auto map = LoadOrExtractUiSfx(game_dir, cache_dir);
    std::string out = "{\"authentic\":" + std::string(map.empty() ? "false" : "true") + ",\"clips\":{";
    bool first = true;
    for (const auto& kv : map) {
        if (!first) out += ",";
        first = false;
        out += "\"" + JsonEscape(kv.first) + "\":\"" + kv.second + "\"";
    }
    out += "}}";
    return out;
}

std::string LauncherBridge::BuildHeroArtJson() const {
    const fs::path game_dir = ResolveGameDir();
    const fs::path cache_dir = ResolveUserDir() / "cache";
    const std::string url = LoadOrExtractHeroArtUrl(game_dir, cache_dir);
    if (url.empty()) {
        return "{\"authentic\":false,\"url\":\"\"}";
    }
    return "{\"authentic\":true,\"url\":\"" + JsonEscape(url) + "\"}";
}

std::string LauncherBridge::BuildGameFontJson() const {
    const fs::path game_dir = ResolveGameDir();
    const fs::path cache_dir = ResolveUserDir() / "cache";
    const std::string url = LoadOrExtractGameFontUrl(game_dir, cache_dir);
    if (url.empty()) {
        return "{\"authentic\":false,\"url\":\"\"}";
    }
    return "{\"authentic\":true,\"url\":\"" + JsonEscape(url) + "\"}";
}

std::string LauncherBridge::BuildBgmJson(const std::string& payload) const {
    const fs::path game_dir = ResolveGameDir();
    const fs::path cache_dir = ResolveUserDir() / "cache";
    std::string track = ExtractJsonField(payload, "track");
    if (track.empty()) track = ReadIniValue("launcher_bgm_track", "hunters_dream");
    const BgmTrackInfo info = LoadOrExtractBgmTrack(game_dir, cache_dir, track);
    if (!info.authentic) {
        return "{\"authentic\":false,\"url\":\"\",\"track\":\"" + JsonEscape(info.track_key) +
               "\",\"loopStart\":0,\"loopEnd\":0}";
    }
    std::ostringstream os;
    os << "{\"authentic\":true,\"track\":\"" << JsonEscape(info.track_key)
       << "\",\"url\":\"" << JsonEscape(info.url)
       << "\",\"loopStart\":" << info.loop_start_sec
       << ",\"loopEnd\":" << info.loop_end_sec << "}";
    return os.str();
}

std::string LauncherBridge::HandleCommand(const std::string& action, const std::string& payload) {
    if (action == "getState") return BuildStateJson();
    if (action == "getUiSfx") return BuildSfxJson();
    if (action == "getHeroArt") return BuildHeroArtJson();
    if (action == "getGameFont") return BuildGameFontJson();
    if (action == "getBgm") return BuildBgmJson(payload);
    if (action == "setUiPref") {
        const std::string lang = ExtractJsonField(payload, "lang");
        if (lang == "ru" || lang == "en") WriteIniValue("launcher_lang", lang);
        const std::string sfx = ExtractJsonField(payload, "sfxEnabled");
        if (sfx == "true" || sfx == "false") WriteIniValue("launcher_sfx", sfx == "true" ? "1" : "0");
        const std::string vol = ExtractJsonField(payload, "sfxVolume");
        if (!vol.empty()) WriteIniValue("launcher_sfx_vol", vol);
        const std::string bgm = ExtractJsonField(payload, "bgmEnabled");
        if (bgm == "true" || bgm == "false") WriteIniValue("launcher_bgm", bgm == "true" ? "1" : "0");
        const std::string bgm_vol = ExtractJsonField(payload, "bgmVolume");
        if (!bgm_vol.empty()) WriteIniValue("launcher_bgm_vol", bgm_vol);
        const std::string bgm_track = ExtractJsonField(payload, "bgmTrack");
        if (bgm_track == "hunters_dream" || bgm_track == "omen") {
            WriteIniValue("launcher_bgm_track", bgm_track);
        }
        return BuildStateJson();
    }
    if (action == "setOnline") {
        const std::string pool = ExtractJsonField(payload, "onlinePool");
        if (!pool.empty()) WriteIniValue("online_pool", pool);
        const std::string upnp = ExtractJsonField(payload, "onlineUpnp");
        if (upnp == "true" || upnp == "false") WriteIniValue("online_upnp", upnp == "true" ? "1" : "0");
        return BuildStateJson();
    }
    if (action == "pickGameFolder") {
        if (folder_picker_) {
            const std::string chosen = folder_picker_("Select folder containing eboot.bin");
            if (!chosen.empty()) WriteIniValue("game_dir", chosen);
        }
        return BuildStateJson();
    }
    if (action == "toggleMod") {
        ToggleMod(ExtractJsonField(payload, "name"), ExtractJsonField(payload, "enabled") == "true");
        return BuildStateJson();
    }
    if (action == "clearShaderCache") {
        ClearShaderCache();
        return BuildStateJson();
    }
    if (action == "resetSafeMode") {
        ResetGraphicsSafeMode();
        return BuildStateJson();
    }
    if (action == "openFolder") {
        OpenSystemFolder(ExtractJsonField(payload, "which"));
        return "{\"ok\":true}";
    }
    if (action == "launchGame") {
        LaunchGameProcess();
        return "{\"ok\":true}";
    }
    if (action == "startBake") {
        StartShaderBake();
        return BuildBakeJson();
    }
    if (action == "pollBake") {
        return BuildBakeJson();
    }
    if (action == "cancelBake") {
        CancelShaderBake();
        return BuildBakeJson();
    }
    if (action == "checkUpdate") {
        StartUpdateCheck();
        return BuildUpdateJson();
    }
    if (action == "applyUpdate") {
        StartUpdateApply();
        return BuildUpdateJson();
    }
    if (action == "pollUpdate") {
        return BuildUpdateJson();
    }
    return "{\"error\":\"unknown_action\"}";
}

}  // namespace bb_launcher
