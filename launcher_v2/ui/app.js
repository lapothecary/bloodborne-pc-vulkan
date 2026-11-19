(() => {
  const I18N = {
    en: {
      nav_home: "Home",
      nav_mods: "Mods & Patches",
      nav_online: "Online",
      nav_online_title: "Online (bbnet20)",
      nav_system: "System, Cache & Files",
      hero_ready: "Ready to hunt",
      hero_setup: "Welcome to Yharnam",
      btn_play: "Play",
      pick_game: "Game Folder (eboot.bin)",
      open_mods_dir: "Open mods/ folder",
      mods_desc: "Mods are staged before launch. Toggle mod folders inside mods/ without deleting files.",
      mods_empty: "No mods found in mods/ yet. Extract mods into mods/ to toggle them here.",
      online_desc: "Online layer inherited from bbport 0.7 (Seamless Hunt / The Hunter's Dream). Configured before launch.",
      online_pool: "Online Pool",
      online_pool_sub: "Choose a community server or play offline",
      online_upnp_sub: "Automatic UPnP port forwarding for co-op",
      shader_cache_title: "Vulkan Shader Cache (5,761)",
      btn_prebake: "Pre-Bake",
      btn_cancel_bake: "Cancel",
      btn_clear_cache: "Clear Cache",
      updates_title: "Port Updates (GitHub SHA-256)",
      updates_idle: "Check for new releases without losing saves or mods",
      updates_checking: "Checking GitHub Releases...",
      updates_up_to_date: "Up to date",
      updates_avail: "Update available:",
      updates_done: "Update installed — restart the launcher to apply.",
      btn_check_update: "Check Updates",
      btn_update_now: "Update",
      safe_reset_title: "Safe-Mode Graphics Reset",
      safe_reset_sub: "Reset resolution, upscaler & remaster shaders if the game fails to start (saves stay untouched)",
      btn_safe_reset: "Reset",
      sfx_title: "Bloodborne Menu SFX",
      sfx_sub: "Decoded locally from your sprj_main_*.fsb with gothic chamber reverb",
      folders_title: "Quick Folders",
      folders_sub: "Open save data, crash logs, or change your game folder",
      btn_saves: "Saves",
      btn_logs: "Logs",
      btn_change_game: "Game Folder",
      wiz_eyebrow: "FIRST-TIME SETUP",
      wiz_title: "Welcome to Yharnam",
      wiz_desc: "Point the launcher to your decrypted Bloodborne v1.09 game folder containing eboot.bin and dvdroot_ps4/. No game files are bundled with this port.",
      wiz_no_folder: "No folder selected yet",
      wiz_choose: "Choose Game Folder",
      wiz_skip: "Skip for now",
      toast_launching: "Launching Bloodborne...",
      toast_cache_cleared: "Shader cache cleared",
      toast_safe_reset: "Graphics settings reset to safe defaults",
      toast_need_game: "Please select your Bloodborne v1.09 folder (eboot.bin) first",
      toast_bake_done: "All shaders pre-compiled!",
    },
    ru: {
      nav_home: "Главная",
      nav_mods: "Моды и патчи",
      nav_online: "Онлайн",
      nav_online_title: "Онлайн (bbnet20)",
      nav_system: "Система, кэш и файлы",
      hero_ready: "Готов к охоте",
      hero_setup: "Добро пожаловать в Ярнам",
      btn_play: "Начать охоту",
      pick_game: "Папка игры (eboot.bin)",
      open_mods_dir: "Открыть папку mods/",
      mods_desc: "Моды монтируются до запуска игры. Переключайте папки внутри mods/ без удаления файлов.",
      mods_empty: "Папка mods/ пока пуста. Положите распакованные моды в mods/ и обновите список.",
      online_desc: "Сетевой модуль из базового bbport 0.7 (Seamless Hunt / The Hunter's Dream). Настраивается до старта игры.",
      online_pool: "Сетевой режим",
      online_pool_sub: "Выберите сервер или автономную игру",
      online_upnp_sub: "Автоматический проброс P2P-портов для кооператива",
      shader_cache_title: "Кэш шейдеров Vulkan (5,761)",
      btn_prebake: "Прекомпиляция",
      btn_cancel_bake: "Отмена",
      btn_clear_cache: "Очистить кэш",
      updates_title: "Обновления порта (GitHub SHA-256)",
      updates_idle: "Проверка свежих релизов (сейвы, моды и bbport.ini сохраняются)",
      updates_checking: "Проверяем релизы на GitHub...",
      updates_up_to_date: "Установлена актуальная версия",
      updates_avail: "Доступно обновление:",
      updates_done: "Обновление установлено — перезапустите лаунчер.",
      btn_check_update: "Проверить",
      btn_update_now: "Обновить",
      safe_reset_title: "Безопасный сброс графики",
      safe_reset_sub: "Сбросить разрешение, апскейлер и ремастер-эффекты, если игра не запускается (сейвы не трогает)",
      btn_safe_reset: "Сбросить",
      sfx_title: "Звуки меню Bloodborne",
      sfx_sub: "Декодируются локально из sprj_main_*.fsb с реверберацией каменного зала",
      folders_title: "Служебные папки",
      folders_sub: "Быстрый доступ к сохранениям, логам крашей и папке игры",
      btn_saves: "Сейвы",
      btn_logs: "Логи",
      btn_change_game: "Папка игры",
      wiz_eyebrow: "ПЕРВЫЙ ЗАПУСК",
      wiz_title: "Добро пожаловать в Ярнам",
      wiz_desc: "Укажите папку с вашим дампом Bloodborne v1.09, где лежат eboot.bin и dvdroot_ps4/. Файлы самой игры в порт не включены.",
      wiz_no_folder: "Папка ещё не выбрана",
      wiz_choose: "Выбрать папку игры",
      wiz_skip: "Пропустить",
      toast_launching: "Запуск Bloodborne...",
      toast_cache_cleared: "Кэш шейдеров очищен",
      toast_safe_reset: "Настройки графики сброшены в безопасный режим",
      toast_need_game: "Сначала укажите папку с дампом Bloodborne v1.09 (eboot.bin)",
      toast_bake_done: "Все шейдеры скомпилированы!",
    }
  };

  const POOL_LABELS = {
    offline: "Offline",
    seamless: "Seamless Hunt (BBSC)",
    dream: "The Hunter's Dream",
  };

  let state = { lang: "en", sfxEnabled: true, sfxVolume: 0.30, gameValid: false };
  const sfxBuffers = {};
  let lastHoverAt = 0;
  let wizardSkipped = false;
  let bakePollTimer = null;
  let updPollTimer = null;

  // Web Audio API + Gothic Stone-Chamber Stereo Reverb (emulating Bloodborne's FMOD SFX Reverb)
  let audioCtx = null;
  let reverbConvolver = null;
  let reverbWetGain = null;
  let reverbDryGain = null;

  function initAudioGraph(ctx) {
    if (reverbConvolver) return;
    const sr = ctx.sampleRate || 48000;
    const decaySec = 1.35;
    const len = Math.floor(sr * decaySec);
    const ir = ctx.createBuffer(2, len, sr);
    const preDelay = Math.floor(sr * 0.014);
    for (let ch = 0; ch < 2; ++ch) {
      const data = ir.getChannelData(ch);
      let lp = 0;
      for (let i = preDelay; i < len; ++i) {
        const t = (i - preDelay) / (len - preDelay);
        const env = Math.pow(1.0 - t, 2.8);
        const white = (Math.random() * 2.0 - 1.0);
        lp = lp * 0.72 + white * 0.28;
        data[i] = lp * env;
      }
      const er = ch === 0 ? [0.019, 0.031, 0.047] : [0.023, 0.037, 0.053];
      er.forEach((sec, idx) => {
        const pos = Math.floor(sr * sec);
        if (pos < len) data[pos] += (0.42 - idx * 0.1) * (idx % 2 === 0 ? 1 : -1);
      });
    }

    reverbConvolver = ctx.createConvolver();
    reverbConvolver.buffer = ir;

    const wetLp = ctx.createBiquadFilter();
    wetLp.type = "lowpass";
    wetLp.frequency.value = 3600;

    reverbWetGain = ctx.createGain();
    reverbWetGain.gain.value = 0.34;
    reverbDryGain = ctx.createGain();
    reverbDryGain.gain.value = 0.88;

    reverbConvolver.connect(wetLp);
    wetLp.connect(reverbWetGain);
    reverbWetGain.connect(ctx.destination);
    reverbDryGain.connect(ctx.destination);
  }

  function ensureAudioCtx() {
    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) {
        audioCtx = new AC();
        initAudioGraph(audioCtx);
      }
    }
    if (audioCtx && audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
    return audioCtx;
  }

  function base64DataUriToArrayBuffer(uri) {
    const comma = uri.indexOf(",");
    const b64 = comma >= 0 ? uri.slice(comma + 1) : uri;
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }

  function playFallbackTone(kind) {
    try {
      const ctx = ensureAudioCtx();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const now = ctx.currentTime;
      const vol = Math.max(0, Math.min(1, state.sfxVolume)) * 0.08;
      osc.type = "triangle";
      osc.frequency.setValueAtTime(kind === "hover" ? 420 : kind === "gamestart" ? 180 : 300, now);
      osc.frequency.exponentialRampToValueAtTime(kind === "hover" ? 280 : 120, now + (kind === "gamestart" ? 0.35 : 0.06));
      gain.gain.setValueAtTime(vol, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + (kind === "gamestart" ? 0.4 : 0.07));
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + (kind === "gamestart" ? 0.42 : 0.08));
    } catch (_) {}
  }

  function playSfx(kind, onDone = null) {
    if (!state.sfxEnabled || state.sfxVolume <= 0.01) {
      if (onDone) onDone();
      return;
    }
    if (kind === "hover") {
      const now = performance.now();
      if (now - lastHoverAt < 50) return;
      lastHoverAt = now;
    }
    const ctx = ensureAudioCtx();
    const buf = sfxBuffers[kind];
    if (ctx && buf) {
      try {
        const src = ctx.createBufferSource();
        const gain = ctx.createGain();
        src.buffer = buf;
        gain.gain.value = Math.max(0, Math.min(1, Number(state.sfxVolume) || 0.30));
        src.connect(gain);
        if (reverbDryGain && reverbConvolver) {
          gain.connect(reverbDryGain);
          gain.connect(reverbConvolver);
        } else {
          gain.connect(ctx.destination);
        }
        if (onDone) {
          let fired = false;
          const finish = () => { if (!fired) { fired = true; onDone(); } };
          const tailExtraMs = kind === "gamestart" ? 550 : 120;
          src.onended = () => setTimeout(finish, tailExtraMs);
          setTimeout(finish, Math.ceil(buf.duration * 1000) + tailExtraMs + 150);
        }
        src.start(0);
        return;
      } catch (_) {}
    }
    playFallbackTone(kind);
    if (onDone) setTimeout(onDone, kind === "gamestart" ? 450 : 100);
  }

  function tr(key) {
    return (I18N[state.lang] && I18N[state.lang][key]) || I18N.en[key] || key;
  }

  let toastTimer = null;
  function showToast(msg) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
  }

  function switchTab(tabId, silent = false) {
    document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tabId));
    document.querySelectorAll(".rail-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
    document.getElementById("hero-backdrop").classList.toggle("dimmed", tabId !== "home");
    document.getElementById("dd-online-pool").classList.remove("open");
    if (!silent) playSfx("tab");
  }

  function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = tr(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      el.title = tr(el.dataset.i18nTitle);
    });
    document.getElementById("btn-lang").textContent = state.lang.toUpperCase();
  }

  function renderState() {
    applyTranslations();
    document.getElementById("ver-badge").textContent = state.version || "v1.1.6";
    document.getElementById("hero-title").textContent = state.gameValid ? tr("hero_ready") : tr("hero_setup");

    // Show First-Time Setup Wizard automatically if game folder is not valid & not skipped
    const wiz = document.getElementById("setup-wizard");
    wiz.classList.toggle("hidden", state.gameValid || wizardSkipped);
    document.getElementById("wiz-path").textContent = state.gameDir || tr("wiz_no_folder");

    const enabledMods = (state.mods || []).filter((m) => m.enabled).length;
    document.getElementById("home-mods-val").textContent =
      enabledMods > 0 ? `${enabledMods} / ${state.mods.length}` : "0";

    const poolKey = state.onlinePool || "offline";
    document.getElementById("home-online-val").textContent =
      poolKey === "seamless" ? "Seamless Hunt" : poolKey === "dream" ? "Hunter's Dream" : "Offline";
    document.getElementById("dd-online-label").textContent = POOL_LABELS[poolKey] || "Offline";
    document.querySelectorAll("#dd-online-menu .bb-dd-item").forEach((item) => {
      item.classList.toggle("selected", item.dataset.val === poolKey);
    });

    const pct = Math.min(100, Math.round((state.shaderCount / Math.max(1, state.shaderTotal)) * 100));
    document.getElementById("home-shader-val").textContent = `${pct}% (${state.shaderCount})`;
    document.getElementById("shader-cache-sub").textContent =
      `${state.shaderCount} / ${state.shaderTotal} (${pct}%)`;
    document.getElementById("home-path-val").textContent = state.gameDir || "—";

    document.getElementById("chk-online-upnp").checked = !!state.onlineUpnp;
    document.getElementById("chk-sfx-on").checked = !!state.sfxEnabled;
    document.getElementById("rng-sfx-vol").value = Math.round((state.sfxVolume ?? 0.30) * 100);
    document.getElementById("btn-sfx").classList.toggle("muted", !state.sfxEnabled);

    const modsList = document.getElementById("mods-list");
    modsList.innerHTML = "";
    if (!state.mods || state.mods.length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = tr("mods_empty");
      modsList.appendChild(empty);
    } else {
      state.mods.forEach((mod) => {
        const row = document.createElement("div");
        row.className = "bb-setting-row";
        row.innerHTML = `
          <div>
            <div class="row-title"></div>
            <div class="row-sub">mods/${mod.name}</div>
          </div>
          <label class="bb-switch">
            <input type="checkbox" ${mod.enabled ? "checked" : ""} />
            <span class="slider"></span>
          </label>`;
        row.querySelector(".row-title").textContent = mod.name;
        const inp = row.querySelector("input");
        inp.addEventListener("change", async () => {
          playSfx("select");
          state = await window.bbInvoke("toggleMod", { name: mod.name, enabled: inp.checked });
          renderState();
        });
        modsList.appendChild(row);
      });
    }
  }

  function renderBake(b) {
    const btn = document.getElementById("btn-bake");
    const wrap = document.getElementById("bake-progress-wrap");
    const bar = document.getElementById("bake-progress-bar");
    const lbl = document.getElementById("bake-progress-label");

    if (b.running) {
      btn.textContent = tr("btn_cancel_bake");
      btn.dataset.baking = "1";
      wrap.classList.remove("hidden");
      bar.style.width = `${Math.max(2, b.pct)}%`;
      lbl.textContent = `${b.pct}% — ${b.done} / ${b.total} (${b.phase || "compiling"})`;
    } else {
      btn.textContent = tr("btn_prebake");
      btn.dataset.baking = "0";
      if (b.finished) {
        wrap.classList.remove("hidden");
        bar.style.width = "100%";
        lbl.textContent = `100%${b.elapsed ? " (" + b.elapsed + "s)" : ""}`;
        showToast(tr("toast_bake_done"));
        window.bbInvoke("getState", {}).then((s) => { state = s; renderState(); });
      } else if (b.error) {
        wrap.classList.remove("hidden");
        lbl.textContent = b.error;
      } else {
        wrap.classList.add("hidden");
      }
      if (bakePollTimer) { clearInterval(bakePollTimer); bakePollTimer = null; }
    }
  }

  function renderUpdate(u) {
    const homeBanner = document.getElementById("home-update-banner");
    const homeText = document.getElementById("home-update-text");
    const homeBarWrap = document.getElementById("home-update-bar-wrap");
    const homeBar = document.getElementById("home-update-bar");
    const sysSub = document.getElementById("sys-update-sub");
    const sysApplyBtn = document.getElementById("btn-sys-apply-update");
    const sysProgWrap = document.getElementById("sys-update-progress-wrap");
    const sysBar = document.getElementById("sys-update-bar");
    const sysLbl = document.getElementById("sys-update-progress-label");

    if (u.checking) {
      sysSub.textContent = tr("updates_checking");
      return;
    }
    if (u.applying) {
      homeBanner.classList.remove("hidden");
      homeBarWrap.classList.remove("hidden");
      homeBar.style.width = `${Math.max(4, u.pct)}%`;
      homeText.textContent = `${u.pct}% — ${u.detail || u.stage}`;
      sysProgWrap.classList.remove("hidden");
      sysBar.style.width = `${Math.max(4, u.pct)}%`;
      sysLbl.textContent = `${u.pct}% — ${u.detail || u.stage}`;
      return;
    }

    if (updPollTimer) { clearInterval(updPollTimer); updPollTimer = null; }
    homeBarWrap.classList.add("hidden");
    sysProgWrap.classList.add("hidden");

    if (u.restartNeeded) {
      homeBanner.classList.remove("hidden");
      homeText.textContent = tr("updates_done");
      sysSub.textContent = tr("updates_done");
      sysApplyBtn.classList.add("hidden");
      showToast(tr("updates_done"));
      // Exit cleanly so the detached Windows/Linux updater finishes swapping binaries and relaunches Launcher
      setTimeout(() => window.bbInvoke("quit", {}), 700);
      return;
    }

    if (u.error) {
      homeBanner.classList.remove("hidden");
      homeText.textContent = u.error;
      sysSub.textContent = u.error;
      showToast(u.error);
      return;
    }

    if (u.checked) {
      if (!u.ok) {
        sysSub.textContent = u.error || "Check failed";
      } else if (u.upToDate) {
        homeBanner.classList.add("hidden");
        sysApplyBtn.classList.add("hidden");
        sysSub.textContent = `${tr("updates_up_to_date")} (${u.localTag || state.version})`;
      } else {
        homeBanner.classList.remove("hidden");
        homeText.textContent = `${tr("updates_avail")} ${u.remoteTag}`;
        sysSub.textContent = `${tr("updates_avail")} ${u.remoteTag} (current: ${u.localTag || state.version})`;
        sysApplyBtn.classList.remove("hidden");
      }
    }
  }

  function startUpdatePoll() {
    if (updPollTimer) clearInterval(updPollTimer);
    updPollTimer = setInterval(async () => {
      const u = await window.bbInvoke("pollUpdate", {});
      renderUpdate(u);
    }, 350);
  }

  async function loadSfx() {
    const res = await window.bbInvoke("getUiSfx", {});
    const clips = res.clips || {};
    const ctx = ensureAudioCtx();
    if (!ctx) return;
    for (const [key, uri] of Object.entries(clips)) {
      try {
        sfxBuffers[key] = await ctx.decodeAudioData(base64DataUriToArrayBuffer(uri));
      } catch (_) {}
    }
  }

  async function init() {
    document.getElementById("titlebar-drag").addEventListener("mousedown", (e) => {
      if (e.button === 0) window.bbInvoke("startDrag", {});
    });
    document.getElementById("btn-win-min").addEventListener("click", () => window.bbInvoke("minimizeWindow", {}));
    document.getElementById("btn-win-close").addEventListener("click", () => window.bbInvoke("closeWindow", {}));

    document.addEventListener("pointerenter", (e) => {
      if (e.target && e.target.closest && e.target.closest("button:not(.win-btn), .bb-menu-row")) {
        playSfx("hover");
      }
    }, true);

    document.querySelectorAll("[data-tab]").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));
    document.querySelectorAll("[data-goto]").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.goto)));

    const dd = document.getElementById("dd-online-pool");
    document.getElementById("dd-online-trigger").addEventListener("click", (e) => {
      e.stopPropagation();
      playSfx("click");
      dd.classList.toggle("open");
    });
    document.querySelectorAll("#dd-online-menu .bb-dd-item").forEach((item) => {
      item.addEventListener("click", async () => {
        dd.classList.remove("open");
        playSfx("select");
        state = await window.bbInvoke("setOnline", { onlinePool: item.dataset.val });
        renderState();
      });
    });
    document.addEventListener("click", () => dd.classList.remove("open"));

    let launching = false;
    document.getElementById("btn-play").addEventListener("click", () => {
      if (launching) return;
      if (!state.gameValid) {
        playSfx("cancel");
        showToast(tr("toast_need_game"));
        return;
      }
      launching = true;
      showToast(tr("toast_launching"));
      window.bbInvoke("launchGame", {});
      setTimeout(() => window.bbInvoke("hideWindow", {}), 160);
      playSfx("gamestart", () => {
        setTimeout(() => window.bbInvoke("quit", {}), 120);
      });
    });

    const pickGame = async () => {
      playSfx("click");
      state = await window.bbInvoke("pickGameFolder", {});
      renderState();
      await loadSfx();
    };
    document.getElementById("btn-pick-game-home").addEventListener("click", pickGame);
    document.getElementById("btn-pick-game-sys").addEventListener("click", pickGame);
    document.getElementById("btn-wiz-pick").addEventListener("click", pickGame);
    document.getElementById("btn-wiz-skip").addEventListener("click", () => {
      playSfx("cancel");
      wizardSkipped = true;
      renderState();
    });

    document.getElementById("btn-lang").addEventListener("click", async () => {
      playSfx("select");
      const next = state.lang === "en" ? "ru" : "en";
      state = await window.bbInvoke("setUiPref", { lang: next });
      renderState();
    });

    const toggleSfx = async (enabled) => {
      state = await window.bbInvoke("setUiPref", { sfxEnabled: enabled });
      renderState();
      if (enabled) playSfx("select");
    };
    document.getElementById("btn-sfx").addEventListener("click", () => toggleSfx(!state.sfxEnabled));
    document.getElementById("chk-sfx-on").addEventListener("change", (e) => toggleSfx(e.target.checked));
    document.getElementById("rng-sfx-vol").addEventListener("input", (e) => {
      state.sfxVolume = Number(e.target.value) / 100;
    });
    document.getElementById("rng-sfx-vol").addEventListener("change", async (e) => {
      const v = (Number(e.target.value) / 100).toFixed(2);
      state = await window.bbInvoke("setUiPref", { sfxVolume: v });
      renderState();
      playSfx("click");
    });

    document.getElementById("chk-online-upnp").addEventListener("change", async (e) => {
      playSfx("select");
      state = await window.bbInvoke("setOnline", { onlineUpnp: e.target.checked });
      renderState();
    });

    // Live Shader Pre-Baker
    document.getElementById("btn-bake").addEventListener("click", async (e) => {
      if (e.currentTarget.dataset.baking === "1") {
        playSfx("cancel");
        const b = await window.bbInvoke("cancelBake", {});
        renderBake(b);
        return;
      }
      playSfx("click");
      const b = await window.bbInvoke("startBake", {});
      renderBake(b);
      if (bakePollTimer) clearInterval(bakePollTimer);
      bakePollTimer = setInterval(async () => {
        const cur = await window.bbInvoke("pollBake", {});
        renderBake(cur);
      }, 450);
    });

    document.getElementById("btn-clear-cache").addEventListener("click", async () => {
      playSfx("cancel");
      state = await window.bbInvoke("clearShaderCache", {});
      renderState();
      showToast(tr("toast_cache_cleared"));
    });

    document.getElementById("btn-safe-reset").addEventListener("click", async () => {
      playSfx("cancel");
      state = await window.bbInvoke("resetSafeMode", {});
      renderState();
      showToast(tr("toast_safe_reset"));
    });

    // 1-Click SHA-256 Updater
    const triggerCheck = async () => {
      playSfx("click");
      const u = await window.bbInvoke("checkUpdate", {});
      renderUpdate(u);
      startUpdatePoll();
    };
    const triggerApply = async () => {
      playSfx("click");
      const u = await window.bbInvoke("applyUpdate", {});
      renderUpdate(u);
      startUpdatePoll();
    };
    document.getElementById("btn-check-update").addEventListener("click", triggerCheck);
    document.getElementById("btn-home-update").addEventListener("click", triggerApply);
    document.getElementById("btn-sys-apply-update").addEventListener("click", triggerApply);

    document.getElementById("btn-open-mods").addEventListener("click", () => { playSfx("click"); window.bbInvoke("openFolder", { which: "mods" }); });
    document.getElementById("btn-open-saves").addEventListener("click", () => { playSfx("click"); window.bbInvoke("openFolder", { which: "saves" }); });
    document.getElementById("btn-open-logs").addEventListener("click", () => { playSfx("click"); window.bbInvoke("openFolder", { which: "logs" }); });

    state = await window.bbInvoke("getState", {});
    renderState();
    await loadSfx();

    // Quiet background update check 1.2s after launch
    setTimeout(async () => {
      const u = await window.bbInvoke("checkUpdate", {});
      renderUpdate(u);
      startUpdatePoll();
    }, 1200);
  }

  window.addEventListener("DOMContentLoaded", init);
})();
