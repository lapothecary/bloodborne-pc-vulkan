#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace bb_launcher {

struct LauncherContext {
    std::filesystem::path package_dir;
    std::filesystem::path ui_dir;
    std::filesystem::path icon_png_path;
};

struct BakeProgress {
    bool running = false;
    bool finished = false;
    bool cancelled = false;
    int pct = 0;
    int done = 0;
    int total = 5761;
    std::string phase;
    std::string elapsed;
    std::string error;
};

struct UpdateProgress {
    bool checking = false;
    bool applying = false;
    bool checked = false;
    bool ok = false;
    bool up_to_date = true;
    bool restart_needed = false;
    int pct = 0;
    std::string remote_tag;
    std::string local_tag;
    std::string stage;
    std::string detail;
    std::string error;
};

class LauncherBridge {
public:
    explicit LauncherBridge(LauncherContext ctx);
    ~LauncherBridge();

    // Executes a JSON RPC command from the WebView UI and returns a JSON response string.
    std::string HandleCommand(const std::string& action, const std::string& payload);

    // Native OS folder picker helper (set by platform main.cpp).
    void SetFolderPicker(std::function<std::string(const std::string& title)> picker) {
        folder_picker_ = std::move(picker);
    }

    const LauncherContext& ctx() const { return ctx_; }

private:
    std::filesystem::path ResolveGameDir() const;
    std::filesystem::path ResolveUserDir() const;
    bool IsValidGameDir(const std::filesystem::path& p) const;
    std::string ReadIniValue(const std::string& key, const std::string& fallback = "") const;
    void WriteIniValue(const std::string& key, const std::string& value) const;
    int CountCachedShaders(const std::filesystem::path& user_dir) const;
    std::string ReadVersionTag() const;
    std::string BuildStateJson() const;
    std::string BuildSfxJson() const;
    std::string BuildModsJson() const;
    std::string BuildBakeJson() const;
    std::string BuildUpdateJson() const;
    void ToggleMod(const std::string& mod_name, bool enabled) const;
    bool ClearShaderCache() const;
    bool ResetGraphicsSafeMode() const;
    bool OpenSystemFolder(const std::string& which) const;
    bool LaunchGameProcess() const;
    void StartShaderBake();
    void CancelShaderBake();
    void StartUpdateCheck();
    void StartUpdateApply();

    LauncherContext ctx_;
    std::function<std::string(const std::string&)> folder_picker_;

    mutable std::mutex bake_mu_;
    BakeProgress bake_{};
    std::atomic<int> bake_pid_{-1};
    std::thread bake_thread_;

    mutable std::mutex upd_mu_;
    UpdateProgress upd_{};
    std::thread upd_thread_;
};

std::string JsonEscape(const std::string& s);

}  // namespace bb_launcher
