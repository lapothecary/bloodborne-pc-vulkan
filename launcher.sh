#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if [[ -z ${BB_GAME_DIR:-} ]]; then
    for cand in ../CUSA03173 ../CUSA00900 ../CUSA00207 ../CUSA01363 ./CUSA03173 ./CUSA00900; do
        if [[ -f "$cand/eboot.bin" ]]; then
            export BB_GAME_DIR="$(realpath "$cand")"
            break
        fi
    done
fi

export BB_PREBUILT="${BB_PREBUILT:-1}"
export BB_DATA_DIR="${BB_DATA_DIR:-$PWD}"
export BB_GX_MODULES="${BB_GX_MODULES:-all}"

if [[ -x ./bin/bb-launcher ]]; then
    exec ./bin/bb-launcher "$@"
elif [[ -x ./launcher_v2/build/bb-launcher ]]; then
    exec ./launcher_v2/build/bb-launcher "$@"
else
    echo "Error: bb-launcher binary not found." >&2
    exit 1
fi
