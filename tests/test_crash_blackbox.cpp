// SPDX-License-Identifier: GPL-2.0-or-later
// Unit test for Universal Crash Black-Box Reporter (`gpu/shim/bbport_crash_blackbox.cpp`):
// Verifies symbol resolution (`dantelion_gx` + `eboot.bin`), `DLRegularHeap` corrupted chunk-header
// detection, Flight Recorder ring-buffer output, and creation of `logs/crash-YYYYMMDD_HHMMSS.log`
// via plain in-memory data structures and normal C++ function calls.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "gpu/bbgpu.h"
#include "gpu/shim/bb_golden_engine.h"
#include "gpu/shim/bbport_crash_blackbox.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <string>
#include <ucontext.h>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    const fs::path temp_data = fs::temp_directory_path() / "bbport_crash_blackbox_test";
    fs::remove_all(temp_data);
    fs::create_directories(temp_data);
    setenv("BB_DATA_DIR", temp_data.string().c_str(), 1);

    // 1. Initialize & record realistic Flight Recorder events
    BbCrashBlackBox::Initialize();
    BbCrashBlackBox::UpdatePlayerPosition(-142.50f, -18.25f, 312.75f);
    bbgpu_flight_record("FILE_OPEN", "/app0/chr/c2200.chrbnd.dcx");
    bbgpu_flight_record("DCX", "inflated 4120 KiB -> 12840 KiB");
    bbgpu_flight_record("FILE_OPEN", "/app0/menu/menu_dlc2.tpf.dcx");
    bbgpu_flight_record("SAVE_WR", "/savedata0/userdata0000");
    bbgpu_flight_record("SAVE_SYNC", "/savedata0/userdata0000");

    // 2. Verify host C++20 `dantelion_gx` symbol resolution without .pdb
    BbGoldenEngine::HostSymbolMatch gx_lookup{};
    assert(BbGoldenEngine::ResolveGuestVma(0x21cda20ULL, &gx_lookup));
    const auto gx_fn_addr = gx_lookup.host_func_addr + 0x10u;
    char resolved_sym[256]{};
    const int ok_sym = bbgpu_resolve_code_address(gx_fn_addr, resolved_sym, sizeof(resolved_sym));
    assert(ok_sym == 1);
    std::printf("Resolved dantelion_gx symbol: %s\n", resolved_sym);
    assert(std::strstr(resolved_sym, "21cda20") != nullptr);

    // 3. Build a synthetic guest image + DLHeapManager + "GFX" DLRegularHeap arena in plain RAM
    std::vector<std::uint8_t> fake_image(0x05800000u, 0);
    std::vector<std::uint8_t> fake_arena(0x4000u, 0);
    const auto arena_base = reinterpret_cast<std::uint64_t>(fake_arena.data());
    const auto arena_end  = arena_base + fake_arena.size();

    // Heap impl + strategy layout (`*(strat + 0x08) == impl`, `name` at `impl + 0x460`, `DLRegularHeap` at `impl + 0x28`)
    alignas(16) std::uint8_t fake_heap_obj[0x500]{};
    const char* gfx_name = "GFX";
    auto* const impl_ptr = fake_heap_obj + 0x08;
    auto* const strat_ptr = fake_heap_obj + 0x20;
    const std::uint64_t impl_addr = reinterpret_cast<std::uint64_t>(impl_ptr);
    const std::uint64_t strat_addr = reinterpret_cast<std::uint64_t>(strat_ptr);
    std::memcpy(strat_ptr + 0x08, &impl_addr, 8);
    const std::uint32_t heap_id = 3u;
    std::memcpy(impl_ptr + BbCrashBlackBox::DantelionHeap::kImplHeapIdOffset, &heap_id, 4);
    std::memcpy(impl_ptr + BbCrashBlackBox::DantelionHeap::kImplHeapNamePtrOffset, &gfx_name, 8);

    auto* reg = reinterpret_cast<BbCrashBlackBox::DantelionHeap::DLRegularHeapState*>(
        impl_ptr + BbCrashBlackBox::DantelionHeap::kImplRegularHeapOffset);
    reg->arena_first_chunk  = arena_base + 0x100u;
    reg->arena_end_sentinel = arena_end - 0x100u;
    reg->used_bytes         = 48ULL * 1024ULL * 1024ULL;
    reg->alloc_count        = 1420u;

    // Place an allocated 184-byte chunk at `arena_base + 0x200` (hdr at `+0x1f0`, next_hdr at `+0x2b0`)
    // with a CORRUPTED `[next_hdr + 0x08] = 0x0000000800000000ULL` (matching the real v1.1.5 crash!)
    const std::uint64_t user_chunk = arena_base + 0x200u;
    const std::uint64_t next_hdr   = arena_base + 0x2b0u;
    auto* cur_hdr = reinterpret_cast<BbCrashBlackBox::DantelionHeap::DLRegularChunkHeader*>(fake_arena.data() + 0x1f0u);
    cur_hdr->prev_phys_hdr  = 0;
    cur_hdr->next_and_flags = next_hdr | 0x3ULL; // curr_in_use=1, prev_in_use=1
    std::memcpy(fake_arena.data() + 0x200u, "FLVER2_OVERFLOW_PAYLOAD_MARKER", 30);

    const std::uint64_t bad_next_hdr_val = 0x0000000800000000ULL;
    std::memcpy(fake_arena.data() + 0x2b0u + 0x08u, &bad_next_hdr_val, 8);

    // Wire DLHeapManager at `fake_image + 0x56a1c10`
    BbCrashBlackBox::DantelionHeap::DLHeapArenaEntry entry{
        .allocator   = strat_addr,
        .arena_start = arena_base,
        .arena_end   = arena_end,
    };
    BbCrashBlackBox::DantelionHeap::DLHeapArenaVector vec{
        .allocator_or_vtable = 0,
        .begin_ptr           = reinterpret_cast<std::uint64_t>(&entry),
        .end_ptr             = reinterpret_cast<std::uint64_t>(&entry + 1),
        .cap_ptr             = reinterpret_cast<std::uint64_t>(&entry + 1),
    };
    alignas(16) std::uint8_t fake_mgr[0x480]{};
    const std::uint64_t vec_addr = reinterpret_cast<std::uint64_t>(&vec);
    std::memcpy(fake_mgr + BbCrashBlackBox::DantelionHeap::kDLHeapManagerVecOffset, &vec_addr, 8);
    const std::uint64_t mgr_addr = reinterpret_cast<std::uint64_t>(fake_mgr);
    std::memcpy(fake_image.data() + BbCrashBlackBox::DantelionHeap::kDLHeapManagerGlobalVma, &mgr_addr, 8);
    std::memcpy(fake_image.data() + BbCrashBlackBox::DantelionHeap::kGXMainAllocatorGlobalVma, &strat_addr, 8);

    // Build synthetic 2-frame RBP chain (`DLRegularHeap::free` <- `dantelion_gx!sub_21cda20` <- `eboot.bin+0x21c6500`)
    std::uintptr_t frame_outer[2]{0, reinterpret_cast<std::uintptr_t>(fake_image.data() + 0x21c6500u)};
    std::uintptr_t frame_inner[2]{reinterpret_cast<std::uintptr_t>(frame_outer), gx_fn_addr};

    ucontext_t uc{};
    uc.uc_mcontext.gregs[REG_RIP] = static_cast<greg_t>(reinterpret_cast<std::uintptr_t>(fake_image.data() + 0x20859c7u));
    uc.uc_mcontext.gregs[REG_RBP] = static_cast<greg_t>(reinterpret_cast<std::uintptr_t>(frame_inner));
    uc.uc_mcontext.gregs[REG_RSP] = static_cast<greg_t>(reinterpret_cast<std::uintptr_t>(frame_inner));
    uc.uc_mcontext.gregs[REG_RDI] = static_cast<greg_t>(reinterpret_cast<std::uintptr_t>(reg));
    uc.uc_mcontext.gregs[REG_RBX] = static_cast<greg_t>(user_chunk);
    uc.uc_mcontext.gregs[REG_RDX] = static_cast<greg_t>(bad_next_hdr_val);
    uc.uc_mcontext.gregs[REG_ERR] = 2; // WRITE fault

    // 4. Write the full Crash Black-Box report
    bbgpu_write_crash_blackbox(SIGSEGV, reinterpret_cast<const void*>(0x7fffffffeULL), &uc,
                               fake_image.data(), fake_image.size());

    // 5. Verify `logs/crash-*.log` was created and contains all key sections
    const fs::path logs_dir = temp_data / "logs";
    assert(fs::is_directory(logs_dir));
    fs::path crash_log;
    for (const auto& de : fs::directory_iterator(logs_dir)) {
        if (de.path().filename().string().starts_with("crash-")) {
            crash_log = de.path();
            break;
        }
    }
    assert(!crash_log.empty());
    std::ifstream ifs(crash_log);
    const std::string report((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    assert(report.find("BLOODBORNE PC — ENGINE CRASH BLACK BOX") != std::string::npos);
    assert(report.find("DLRegularHeap::free") != std::string::npos);
    assert(report.find("dantelion_gx!") != std::string::npos);
    assert(report.find("CORRUPTED HEAP HEADER DETECTED") != std::string::npos);
    assert(report.find("CORRUPTED NEXT CHUNK HEADER") != std::string::npos);
    assert(report.find("FLVER2_OVERFLOW_") != std::string::npos);
    assert(report.find("PAYLOAD_MARKER") != std::string::npos);
    assert(report.find("/app0/chr/c2200.chrbnd.dcx") != std::string::npos);
    assert(report.find("/savedata0/userdata0000") != std::string::npos);

    fs::remove_all(temp_data);
    std::puts("PASS: Crash Black-Box verified (symbolizer + DLRegularHeap corruption detector + Flight Recorder + logs/crash-*.log)");
    return 0;
}
