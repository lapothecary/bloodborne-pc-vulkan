#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

export BB_PREBUILT="${BB_PREBUILT:-1}"
export BB_DATA_DIR="${BB_DATA_DIR:-$PWD}"

# Read launcher-persisted startup flags from bbport.ini if present (handles both key=val and key = val)
INI_FILE="$BB_DATA_DIR/bbport.ini"
ini_val() {
    local key="$1" def="$2"
    if [[ -f "$INI_FILE" ]]; then
        local v
        v="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$INI_FILE" 2>/dev/null | tail -n1 | cut -d= -f2- | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' || true)"
        if [[ -n "$v" ]]; then
            echo "$v"
            return
        fi
    fi
    echo "$def"
}

# Normalize a candidate path (direct CUSA*, inside dvdroot_ps4, or parent folder containing CUSA*)
_bb_resolve_dir() {
    local raw="$1"
    [[ -n "$raw" ]] || return 1
    if [[ -f "$raw/eboot.bin" ]]; then
        realpath "$raw"
        return 0
    fi
    if [[ -f "$raw/../eboot.bin" && "$(basename -- "$raw")" == "dvdroot_ps4" ]]; then
        realpath "$raw/.."
        return 0
    fi
    local sub
    for sub in CUSA03173 CUSA00900 CUSA00207 CUSA01363 game; do
        if [[ -f "$raw/$sub/eboot.bin" ]]; then
            realpath "$raw/$sub"
            return 0
        fi
    done
    return 1
}

# Auto-detect unpacked PS4 Bloodborne dump: bbport.ini -> up to 3 parent levels + siblings -> native picker
if [[ -z ${BB_GAME_DIR:-} ]]; then
    saved_game_dir="$(ini_val game_dir "")"
    if resolved="$(_bb_resolve_dir "$saved_game_dir")"; then
        export BB_GAME_DIR="$resolved"
    fi
fi
if [[ -z ${BB_GAME_DIR:-} ]]; then
    _bb_cur="$PWD"
    for _lvl in 0 1 2 3; do
        for _name in CUSA03173 CUSA00900 CUSA00207 CUSA01363 game Bloodborne/CUSA03173 Bloodborne/CUSA00900 Bloodborne/game; do
            if [[ -f "$_bb_cur/$_name/eboot.bin" ]]; then
                export BB_GAME_DIR="$(realpath "$_bb_cur/$_name")"
                break 2
            fi
        done
        for _sub in "$_bb_cur"/*; do
            [[ -d "$_sub" ]] || continue
            _bn="$(basename -- "$_sub")"
            case "$_bn" in
                .*|out|bin|logs|user|patches|scripts|gpu|mods|launcher|launcher_v2|data|fsr4_shaders|dantelion_gx|third_party|build_gpu|build_win32_gpu) continue ;;
            esac
            if [[ -f "$_sub/eboot.bin" ]]; then
                export BB_GAME_DIR="$(realpath "$_sub")"
                break 2
            fi
            for _name in CUSA03173 CUSA00900 CUSA00207 CUSA01363 game; do
                if [[ -f "$_sub/$_name/eboot.bin" ]]; then
                    export BB_GAME_DIR="$(realpath "$_sub/$_name")"
                    break 3
                fi
            done
        done
        _bb_next="$(dirname -- "$_bb_cur")"
        [[ "$_bb_next" == "$_bb_cur" ]] && break
        _bb_cur="$_bb_next"
    done
fi
if [[ -z ${BB_GAME_DIR:-} && -n ${DISPLAY:-}${WAYLAND_DISPLAY:-} && -x ./out/bb-launcher ]]; then
    _picked="$(./out/bb-launcher --pick-game-dir 2>/dev/null || true)"
    if [[ -n "$_picked" && -f "$_picked/eboot.bin" ]]; then
        export BB_GAME_DIR="$_picked"
    fi
fi

export BB_SAVE_LOG="${BB_SAVE_LOG:-1}"
export BB_GX_MODULES="${BB_GX_MODULES:-all}"
export BB_FPS="${BB_FPS:-uncap}"
export BB_FPS_LIMIT="${BB_FPS_LIMIT:-0}"
export BB_PC_MODEL="${BB_PC_MODEL:-0}"
export BB_PREUPLOAD="${BB_PREUPLOAD:-2}"
export BB_READBACKS="${BB_READBACKS:-1}"
export BB_BREADCRUMBS="${BB_BREADCRUMBS:-0}"
export BB_INDIRECT_GUARD="${BB_INDIRECT_GUARD:-0}"
export BB_OBJECT_MOTION="${BB_OBJECT_MOTION:-0}"
export BB_LIVE_RES="${BB_LIVE_RES:-0}"
export BB_WINDOW_MODE="${BB_WINDOW_MODE:-$(ini_val window_mode "")}"
export BB_FULLSCREEN="${BB_FULLSCREEN:-$(ini_val fullscreen 1)}"
export BB_PRESENT_MODE="${BB_PRESENT_MODE:-$(ini_val present_mode Mailbox)}"
export BB_SKIP_NETWORK_CHOICE="${BB_SKIP_NETWORK_CHOICE:-$(ini_val skip_network_choice 1)}"
export BB_LANGUAGE="${BB_LANGUAGE:-$(ini_val language 1)}"

exec bash ./run.sh "$@"
# Release v1.1.1
