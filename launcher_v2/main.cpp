#include "fsb_ui_sfx.h"
#include "launcher_bridge.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dwmapi.h>
#include <objbase.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <shlwapi.h>
#include <WebView2.h>
#else
#include <gtk/gtk.h>
#include <webkit2/webkit2.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

fs::path GetExecutableDir(const char* argv0) {
#ifdef _WIN32
    wchar_t wbuf[4096]{};
    DWORD n = GetModuleFileNameW(nullptr, wbuf, 4096);
    if (n > 0 && n < 4096) {
        return fs::path(wbuf).parent_path();
    }
#else
    char buf[4096]{};
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return fs::path(buf).parent_path();
    }
#endif
    std::error_code ec;
    return fs::weakly_canonical(fs::path(argv0), ec).parent_path();
}

bb_launcher::LauncherContext ResolveContext(const char* argv0) {
    bb_launcher::LauncherContext ctx;
    fs::path exe_dir = GetExecutableDir(argv0);

    fs::path probe = exe_dir;
    for (int i = 0; i < 4; ++i) {
        std::error_code ec;
        if (fs::is_regular_file(probe / "launcher_v2" / "ui" / "index.html", ec)) {
            ctx.package_dir = probe;
            ctx.ui_dir = probe / "launcher_v2" / "ui";
            break;
        }
        if (fs::is_regular_file(probe / "ui" / "index.html", ec)) {
            ctx.package_dir = probe.parent_path();
            ctx.ui_dir = probe / "ui";
            break;
        }
        if (!probe.has_parent_path() || probe == probe.parent_path()) break;
        probe = probe.parent_path();
    }

    if (ctx.package_dir.empty()) {
        ctx.package_dir = fs::current_path();
        ctx.ui_dir = ctx.package_dir / "launcher_v2" / "ui";
    }
    ctx.icon_png_path = ctx.package_dir / "launcher" / "bloodborne.png";
    return ctx;
}

std::string ExtractJsonStrField(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return {};
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (pos >= json.size()) return {};
    if (json[pos] == '"') {
        ++pos;
        std::string out;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                ++pos;
                out.push_back(json[pos]);
            } else {
                out.push_back(json[pos]);
            }
            ++pos;
        }
        return out;
    }
    size_t end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != ' ') ++end;
    return json.substr(pos, end - pos);
}

std::string ExtractRawPayload(const std::string& msg) {
    std::string payload = "{}";
    const size_t ppos = msg.find("\"payload\"");
    if (ppos != std::string::npos) {
        size_t colon = msg.find(':', ppos + 9);
        if (colon != std::string::npos) {
            payload = msg.substr(colon + 1);
            if (!payload.empty() && payload.back() == '}') payload.pop_back();
        }
    }
    return payload;
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

std::string WideToUtf8(const wchar_t* ws) {
    if (!ws || !*ws) return {};
    int need = WideCharToMultiByte(CP_UTF8, 0, ws, -1, nullptr, 0, nullptr, nullptr);
    if (need <= 1) return {};
    std::string out(static_cast<size_t>(need - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, -1, out.data(), need - 1, nullptr, nullptr);
    return out;
}

std::string PickFolderWin32(HWND owner, const std::string& title) {
    IFileOpenDialog* dlg = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dlg))) || !dlg) {
        return {};
    }
    DWORD opts = 0;
    dlg->GetOptions(&opts);
    dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    const std::wstring wtitle = Utf8ToWide(title);
    if (!wtitle.empty()) dlg->SetTitle(wtitle.c_str());

    std::string chosen;
    if (SUCCEEDED(dlg->Show(owner))) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dlg->GetResult(&item)) && item) {
            PWSTR path = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path) {
                chosen = WideToUtf8(path);
                CoTaskMemFree(path);
            }
            item->Release();
        }
    }
    dlg->Release();
    return chosen;
}

template <typename Iface, typename InvokeFn>
class SimpleComHandler : public Iface {
public:
    explicit SimpleComHandler(InvokeFn fn) : fn_(std::move(fn)) {}
    virtual ~SimpleComHandler() = default;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == __uuidof(Iface)) {
            *ppv = static_cast<Iface*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++ref_; }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG r = --ref_;
        if (r == 0) delete this;
        return r;
    }

protected:
    std::atomic<ULONG> ref_{1};
    InvokeFn fn_;
};

class EnvCompletedHandler
    : public SimpleComHandler<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler,
                              std::function<HRESULT(HRESULT, ICoreWebView2Environment*)>> {
public:
    using SimpleComHandler::SimpleComHandler;
    HRESULT STDMETHODCALLTYPE Invoke(HRESULT res, ICoreWebView2Environment* env) override {
        return fn_(res, env);
    }
};

class ControllerCompletedHandler
    : public SimpleComHandler<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler,
                              std::function<HRESULT(HRESULT, ICoreWebView2Controller*)>> {
public:
    using SimpleComHandler::SimpleComHandler;
    HRESULT STDMETHODCALLTYPE Invoke(HRESULT res, ICoreWebView2Controller* ctrl) override {
        return fn_(res, ctrl);
    }
};

class WebMessageHandler
    : public SimpleComHandler<ICoreWebView2WebMessageReceivedEventHandler,
                              std::function<HRESULT(ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs*)>> {
public:
    using SimpleComHandler::SimpleComHandler;
    HRESULT STDMETHODCALLTYPE Invoke(ICoreWebView2* wv,
                                     ICoreWebView2WebMessageReceivedEventArgs* args) override {
        return fn_(wv, args);
    }
};

struct WinAppState {
    HWND hwnd = nullptr;
    ICoreWebView2Controller* controller = nullptr;
    ICoreWebView2* webview = nullptr;
    std::unique_ptr<bb_launcher::LauncherBridge> bridge;
};

static WinAppState* g_win_app = nullptr;

LRESULT CALLBACK WinWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_SIZE:
            if (g_win_app && g_win_app->controller) {
                RECT rc{};
                GetClientRect(hwnd, &rc);
                g_win_app->controller->put_Bounds(rc);
            }
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

int RunWindowsWebView(bb_launcher::LauncherContext ctx) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    HINSTANCE hinst = GetModuleHandleW(nullptr);
    HICON hicon = LoadIconW(hinst, MAKEINTRESOURCEW(101));

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WinWndProc;
    wc.hInstance = hinst;
    wc.hIcon = hicon;
    wc.hIconSm = hicon;
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    wc.hbrBackground = CreateSolidBrush(RGB(9, 10, 13));
    wc.lpszClassName = L"BloodborneLauncherV2Wnd";
    RegisterClassExW(&wc);

    // Compute DPI-scaled window size centered on primary monitor
    const UINT dpi = GetDpiForSystem();
    const int win_w = MulDiv(1160, static_cast<int>(dpi), 96);
    const int win_h = MulDiv(720, static_cast<int>(dpi), 96);
    const int scr_w = GetSystemMetrics(SM_CXSCREEN);
    const int scr_h = GetSystemMetrics(SM_CYSCREEN);
    const int pos_x = (scr_w - win_w) / 2;
    const int pos_y = (scr_h - win_h) / 2;

    HWND hwnd = CreateWindowExW(
        0, wc.lpszClassName, L"Bloodborne Vulkan Remaster",
        WS_POPUP | WS_MINIMIZEBOX | WS_SYSMENU | WS_CLIPCHILDREN,
        pos_x, pos_y, win_w, win_h,
        nullptr, nullptr, hinst, nullptr);
    if (!hwnd) {
        CoUninitialize();
        return 1;
    }

    if (hicon) {
        SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hicon));
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hicon));
    }

    // Native Windows 11 rounded corners + subtle DWM shadow
    const DWORD corner_pref = 2;  // DWMWCP_ROUND
    DwmSetWindowAttribute(hwnd, 33 /* DWMWA_WINDOW_CORNER_PREFERENCE */, &corner_pref, sizeof(corner_pref));
    const MARGINS margins = {0, 0, 0, 1};
    DwmExtendFrameIntoClientArea(hwnd, &margins);

    WinAppState app;
    app.hwnd = hwnd;
    app.bridge = std::make_unique<bb_launcher::LauncherBridge>(ctx);
    app.bridge->SetFolderPicker([hwnd](const std::string& title) {
        return PickFolderWin32(hwnd, title);
    });
    g_win_app = &app;

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    // Keep WebView2 profile portable inside user/cache/webview2
    const fs::path wv2_data = ctx.package_dir / "user" / "cache" / "webview2";
    std::error_code ec;
    fs::create_directories(wv2_data, ec);

    const fs::path html_path = fs::weakly_canonical(ctx.ui_dir / "index.html", ec);
    wchar_t file_url[4096]{};
    DWORD url_len = 4096;
    if (FAILED(UrlCreateFromPathW(html_path.wstring().c_str(), file_url, &url_len, 0))) {
        std::wstring fallback = L"file:///" + html_path.wstring();
        wcsncpy_s(file_url, fallback.c_str(), _TRUNCATE);
    }
    const std::wstring target_url(file_url);

    auto* env_handler = new EnvCompletedHandler(
        [&app, hwnd, target_url](HRESULT hr, ICoreWebView2Environment* env) -> HRESULT {
            if (FAILED(hr) || !env) {
                MessageBoxW(hwnd,
                    L"Microsoft Edge WebView2 Runtime is required to open the Bloodborne Launcher.\n"
                    L"You can still launch the game directly via Bloodborne.exe.",
                    L"Bloodborne PC Vulkan", MB_OK | MB_ICONWARNING);
                PostQuitMessage(1);
                return hr;
            }
            auto* ctrl_handler = new ControllerCompletedHandler(
                [&app, hwnd, target_url](HRESULT chr, ICoreWebView2Controller* ctrl) -> HRESULT {
                    if (FAILED(chr) || !ctrl) {
                        PostQuitMessage(1);
                        return chr;
                    }
                    app.controller = ctrl;
                    ctrl->AddRef();

                    // Dark background before HTML first paint (zero white flash)
                    ICoreWebView2Controller2* ctrl2 = nullptr;
                    if (SUCCEEDED(ctrl->QueryInterface(IID_PPV_ARGS(&ctrl2))) && ctrl2) {
                        COREWEBVIEW2_COLOR dark{255, 9, 10, 13};
                        ctrl2->put_DefaultBackgroundColor(dark);
                        ctrl2->Release();
                    }

                    RECT rc{};
                    GetClientRect(hwnd, &rc);
                    ctrl->put_Bounds(rc);
                    ctrl->put_IsVisible(TRUE);

                    ICoreWebView2* wv = nullptr;
                    if (SUCCEEDED(ctrl->get_CoreWebView2(&wv)) && wv) {
                        app.webview = wv;
                        ICoreWebView2Settings* st = nullptr;
                        if (SUCCEEDED(wv->get_Settings(&st)) && st) {
                            st->put_AreDefaultContextMenusEnabled(FALSE);
                            st->put_AreDevToolsEnabled(FALSE);
                            st->put_IsStatusBarEnabled(FALSE);
                            st->put_IsZoomControlEnabled(FALSE);
                            st->Release();
                        }

                        const wchar_t* init_js = LR"JS(
                            window.__bbSeq = 1;
                            window.__bbPending = new Map();
                            window.__bbResolve = function(id, data) {
                                const cb = window.__bbPending.get(id);
                                if (cb) { window.__bbPending.delete(id); cb(data); }
                            };
                            window.bbInvoke = function(action, payload) {
                                return new Promise((resolve) => {
                                    const id = window.__bbSeq++;
                                    window.__bbPending.set(id, resolve);
                                    window.chrome.webview.postMessage(
                                        JSON.stringify({ id: id, action: action, payload: payload || {} })
                                    );
                                });
                            };
                        )JS";
                        wv->AddScriptToExecuteOnDocumentCreated(init_js, nullptr);

                        auto* msg_handler = new WebMessageHandler(
                            [&app, hwnd](ICoreWebView2* sender,
                                         ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                                LPWSTR raw_w = nullptr;
                                if (FAILED(args->TryGetWebMessageAsString(&raw_w)) || !raw_w) {
                                    return S_OK;
                                }
                                const std::string msg = WideToUtf8(raw_w);
                                CoTaskMemFree(raw_w);

                                const std::string req_id = ExtractJsonStrField(msg, "id");
                                const std::string action = ExtractJsonStrField(msg, "action");
                                const std::string payload = ExtractRawPayload(msg);

                                if (action == "quit" || action == "closeWindow") {
                                    PostQuitMessage(0);
                                    return S_OK;
                                }
                                if (action == "hideWindow") {
                                    ShowWindow(hwnd, SW_HIDE);
                                    return S_OK;
                                }
                                if (action == "minimizeWindow") {
                                    ShowWindow(hwnd, SW_MINIMIZE);
                                    return S_OK;
                                }
                                if (action == "startDrag") {
                                    ReleaseCapture();
                                    SendMessageW(hwnd, WM_NCLBUTTONDOWN, HTCAPTION, 0);
                                    return S_OK;
                                }

                                const std::string reply = app.bridge->HandleCommand(action, payload);
                                const std::wstring js = Utf8ToWide(
                                    "window.__bbResolve && window.__bbResolve(" + req_id + ", " + reply + ");");
                                sender->ExecuteScript(js.c_str(), nullptr);
                                return S_OK;
                            });
                        EventRegistrationToken tok{};
                        wv->add_WebMessageReceived(msg_handler, &tok);
                        msg_handler->Release();

                        wv->Navigate(target_url.c_str());
                    }
                    return S_OK;
                });
            env->CreateCoreWebView2Controller(hwnd, ctrl_handler);
            ctrl_handler->Release();
            return S_OK;
        });

    HRESULT init_hr = CreateCoreWebView2EnvironmentWithOptions(
        nullptr, wv2_data.wstring().c_str(), nullptr, env_handler);
    env_handler->Release();
    if (FAILED(init_hr)) {
        MessageBoxW(hwnd,
            L"Could not initialize Microsoft Edge WebView2 Runtime.\n"
            L"Install WebView2 Runtime or run Bloodborne.exe directly.",
            L"Bloodborne PC Vulkan", MB_OK | MB_ICONWARNING);
        DestroyWindow(hwnd);
        CoUninitialize();
        return 1;
    }

    MSG m{};
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }

    if (app.webview) app.webview->Release();
    if (app.controller) {
        app.controller->Close();
        app.controller->Release();
    }
    g_win_app = nullptr;
    CoUninitialize();
    return 0;
}

#else

struct AppState {
    GtkWidget* window = nullptr;
    WebKitWebView* webview = nullptr;
    std::unique_ptr<bb_launcher::LauncherBridge> bridge;
};

void OnScriptMessage(WebKitUserContentManager*,
                     WebKitJavascriptResult* js_result,
                     gpointer user_data) {
    auto* app = static_cast<AppState*>(user_data);
    JSCValue* val = webkit_javascript_result_get_js_value(js_result);
    if (!val || !jsc_value_is_string(val)) return;

    gchar* raw_str = jsc_value_to_string(val);
    if (!raw_str) return;
    std::string msg(raw_str);
    g_free(raw_str);

    const std::string req_id = ExtractJsonStrField(msg, "id");
    const std::string action = ExtractJsonStrField(msg, "action");
    const std::string payload = ExtractRawPayload(msg);

    if (action == "quit" || action == "closeWindow") {
        gtk_main_quit();
        return;
    }
    if (action == "hideWindow") {
        gtk_widget_hide(app->window);
        return;
    }
    if (action == "minimizeWindow") {
        gtk_window_iconify(GTK_WINDOW(app->window));
        return;
    }
    if (action == "startDrag") {
        GdkWindow* gdk_win = gtk_widget_get_window(app->window);
        if (gdk_win) {
            GdkDisplay* disp = gdk_window_get_display(gdk_win);
            GdkSeat* seat = gdk_display_get_default_seat(disp);
            GdkDevice* ptr = seat ? gdk_seat_get_pointer(seat) : nullptr;
            gint rx = 0, ry = 0;
            if (ptr) {
                gdk_device_get_position(ptr, nullptr, &rx, &ry);
                gtk_window_begin_move_drag(GTK_WINDOW(app->window), 1, rx, ry, GDK_CURRENT_TIME);
            }
        }
        return;
    }

    const std::string reply_json = app->bridge->HandleCommand(action, payload);
    const std::string script =
        "window.__bbResolve && window.__bbResolve(" + req_id + ", " + reply_json + ");";
    webkit_web_view_evaluate_javascript(app->webview, script.c_str(),
                                        static_cast<gssize>(script.size()), nullptr, nullptr,
                                        nullptr, nullptr, nullptr);
}

int RunLinuxWebView(bb_launcher::LauncherContext ctx, int argc, char** argv) {
    gtk_init(&argc, &argv);

    AppState app;
    app.bridge = std::make_unique<bb_launcher::LauncherBridge>(ctx);

    GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    app.window = window;
    gtk_window_set_title(GTK_WINDOW(window), "Bloodborne Vulkan Remaster");

    GdkScreen* screen = gtk_widget_get_screen(window);
    if (GdkVisual* rgba = gdk_screen_get_rgba_visual(screen)) {
        gtk_widget_set_visual(window, rgba);
        gtk_widget_set_app_paintable(window, TRUE);
    }
    GtkWidget* empty_titlebar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_size_request(empty_titlebar, -1, 0);
    gtk_window_set_titlebar(GTK_WINDOW(window), empty_titlebar);
    gtk_window_set_decorated(GTK_WINDOW(window), FALSE);

    gtk_window_set_default_size(GTK_WINDOW(window), 1160, 720);
    gtk_widget_set_size_request(window, 960, 620);
    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);

    g_object_set(gtk_settings_get_default(), "gtk-application-prefer-dark-theme", TRUE, nullptr);
    std::error_code ec;
    if (fs::is_regular_file(ctx.icon_png_path, ec)) {
        gtk_window_set_icon_from_file(GTK_WINDOW(window), ctx.icon_png_path.c_str(), nullptr);
    }

    app.bridge->SetFolderPicker([window](const std::string& title) -> std::string {
        GtkWidget* dlg = gtk_file_chooser_dialog_new(
            title.c_str(), GTK_WINDOW(window), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
            "_Cancel", GTK_RESPONSE_CANCEL, "_Select", GTK_RESPONSE_ACCEPT, nullptr);
        std::string chosen;
        if (gtk_dialog_run(GTK_DIALOG(dlg)) == GTK_RESPONSE_ACCEPT) {
            char* fn = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dlg));
            if (fn) {
                chosen = fn;
                g_free(fn);
            }
        }
        gtk_widget_destroy(dlg);
        return chosen;
    });

    WebKitUserContentManager* ucm = webkit_user_content_manager_new();
    webkit_user_content_manager_register_script_message_handler(ucm, "bbHost");
    g_signal_connect(ucm, "script-message-received::bbHost", G_CALLBACK(OnScriptMessage), &app);

    const char* init_js = R"JS(
        window.__bbSeq = 1;
        window.__bbPending = new Map();
        window.__bbResolve = function(id, data) {
            const cb = window.__bbPending.get(id);
            if (cb) { window.__bbPending.delete(id); cb(data); }
        };
        window.bbInvoke = function(action, payload) {
            return new Promise((resolve) => {
                const id = window.__bbSeq++;
                window.__bbPending.set(id, resolve);
                window.webkit.messageHandlers.bbHost.postMessage(
                    JSON.stringify({ id: id, action: action, payload: payload || {} })
                );
            });
        };
    )JS";
    WebKitUserScript* user_script = webkit_user_script_new(
        init_js, WEBKIT_USER_CONTENT_INJECT_TOP_FRAME,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, nullptr, nullptr);
    webkit_user_content_manager_add_script(ucm, user_script);

    GtkWidget* webview_widget = webkit_web_view_new_with_user_content_manager(ucm);
    app.webview = WEBKIT_WEB_VIEW(webview_widget);

    GdkRGBA bg = {0.0, 0.0, 0.0, 0.0};
    webkit_web_view_set_background_color(app.webview, &bg);

    WebKitSettings* settings = webkit_web_view_get_settings(app.webview);
    webkit_settings_set_enable_javascript(settings, TRUE);
    webkit_settings_set_allow_file_access_from_file_urls(settings, TRUE);
    webkit_settings_set_allow_universal_access_from_file_urls(settings, TRUE);
    webkit_settings_set_enable_developer_extras(settings, TRUE);
    webkit_settings_set_media_playback_requires_user_gesture(settings, FALSE);

    g_signal_connect(app.webview, "context-menu", G_CALLBACK(+[](WebKitWebView*, WebKitContextMenu*, GdkEvent*, WebKitHitTestResult*, gpointer) -> gboolean {
        return TRUE;
    }), nullptr);

    gtk_container_add(GTK_CONTAINER(window), webview_widget);
    g_signal_connect(window, "draw", G_CALLBACK(+[](GtkWidget*, cairo_t* cr, gpointer) -> gboolean {
        cairo_save(cr);
        cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
        cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.0);
        cairo_paint(cr);
        cairo_restore(cr);
        return FALSE;
    }), nullptr);
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), nullptr);

    const fs::path html_path = fs::weakly_canonical(ctx.ui_dir / "index.html", ec);
    const std::string uri = "file://" + html_path.string();
    webkit_web_view_load_uri(app.webview, uri.c_str());

    gtk_widget_show_all(window);
    gtk_main();
    return 0;
}

#endif

}  // namespace

int main(int argc, char** argv) {
    auto ctx = ResolveContext(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test-sfx") == 0) {
            bb_launcher::LauncherBridge bridge(ctx);
            const std::string state = bridge.HandleCommand("getState", "{}");
            std::cout << "STATE: " << state << "\n";
            const std::string sfx = bridge.HandleCommand("getUiSfx", "{}");
            std::cout << "SFX_JSON_LEN: " << sfx.size()
                      << " authentic=" << (sfx.find("\"authentic\":true") != std::string::npos ? "true" : "false")
                      << "\n";
            return 0;
        }
    }

#ifdef _WIN32
    return RunWindowsWebView(ctx);
#else
    return RunLinuxWebView(ctx, argc, argv);
#endif
}
