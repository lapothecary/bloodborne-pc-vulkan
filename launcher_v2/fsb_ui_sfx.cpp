#include "fsb_ui_sfx.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

extern "C" {
#include "libatrac9.h"
}

namespace bb_launcher {
namespace {

struct TargetSfx {
    const char* key;
    const char* fsb_name;
};

constexpr std::array<TargetSfx, 6> kTargetSfx = {{
    {"hover", "CURSOL_MOVE"},
    {"click", "CURSOL_SELECT"},
    {"select", "CURSOL_SELECT"},
    {"cancel", "CURSOL_CANCEL"},
    {"tab", "menu_open"},
    {"gamestart", "CURSOL_OK"},
}};

inline uint32_t ReadU32(const uint8_t* p) {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint64_t ReadU64(const uint8_t* p) {
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

std::string Base64Encode(const std::vector<uint8_t>& in) {
    static constexpr char kTab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= in.size()) {
        uint32_t n = (uint32_t(in[i]) << 16) | (uint32_t(in[i + 1]) << 8) | uint32_t(in[i + 2]);
        out.push_back(kTab[(n >> 18) & 63]);
        out.push_back(kTab[(n >> 12) & 63]);
        out.push_back(kTab[(n >> 6) & 63]);
        out.push_back(kTab[n & 63]);
        i += 3;
    }
    if (i < in.size()) {
        uint32_t n = uint32_t(in[i]) << 16;
        if (i + 1 < in.size()) n |= uint32_t(in[i + 1]) << 8;
        out.push_back(kTab[(n >> 18) & 63]);
        out.push_back(kTab[(n >> 12) & 63]);
        out.push_back(i + 1 < in.size() ? kTab[(n >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto sz = f.tellg();
    if (sz <= 0) return {};
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

bool WriteWavFile(const std::filesystem::path& out_path,
                  const std::vector<int16_t>& pcm,
                  int sample_rate,
                  int channels) {
    if (pcm.empty() || sample_rate <= 0 || channels <= 0) return false;
    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    const uint32_t riff_size = 36u + data_bytes;
    const uint16_t audio_fmt = 1;
    const uint16_t ch16 = static_cast<uint16_t>(channels);
    const uint32_t sr32 = static_cast<uint32_t>(sample_rate);
    const uint32_t byte_rate = sr32 * ch16 * 2u;
    const uint16_t block_align = static_cast<uint16_t>(ch16 * 2u);
    const uint16_t bits_per_sample = 16;
    const uint32_t fmt_size = 16;

    std::ofstream f(out_path, std::ios::binary);
    if (!f) return false;
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&riff_size), 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    f.write(reinterpret_cast<const char*>(&audio_fmt), 2);
    f.write(reinterpret_cast<const char*>(&ch16), 2);
    f.write(reinterpret_cast<const char*>(&sr32), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_bytes), 4);
    f.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(data_bytes));
    return f.good();
}

std::filesystem::path FindMainFsb(const std::filesystem::path& game_dir) {
    namespace fs = std::filesystem;
    if (game_dir.empty()) return {};
    const fs::path snd_dir = game_dir / "dvdroot_ps4" / "sound";
    std::error_code ec;
    if (!fs::is_directory(snd_dir, ec)) return {};

    const fs::path eng = snd_dir / "sprj_main_eng.fsb";
    if (fs::is_regular_file(eng, ec)) return eng;

    for (const auto& entry : fs::directory_iterator(snd_dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        const std::string fn = entry.path().filename().string();
        if (fn.rfind("sprj_main_", 0) == 0 && fn.size() > 4 &&
            fn.substr(fn.size() - 4) == ".fsb") {
            return entry.path();
        }
    }
    return {};
}

bool DecodeAt9ToWav(const uint8_t* at9_data,
                    size_t at9_size,
                    uint32_t target_samples,
                    const uint8_t config_data[4],
                    const std::filesystem::path& out_wav) {
    void* handle = Atrac9GetHandle();
    if (!handle) return false;

    unsigned char cfg[4] = {config_data[0], config_data[1], config_data[2], config_data[3]};
    if (Atrac9InitDecoder(handle, cfg) != 0) {
        Atrac9ReleaseHandle(handle);
        return false;
    }

    Atrac9CodecInfo info{};
    if (Atrac9GetCodecInfo(handle, &info) != 0 || info.superframeSize <= 0 ||
        info.frameSamples <= 0 || info.channels <= 0) {
        Atrac9ReleaseHandle(handle);
        return false;
    }

    std::vector<int16_t> pcm;
    pcm.reserve(static_cast<size_t>(target_samples + info.frameSamples) *
                static_cast<size_t>(info.channels));
    std::vector<int16_t> frame_buf(static_cast<size_t>(info.frameSamples) *
                                   static_cast<size_t>(info.channels));

    size_t sf_off = 0;
    while (sf_off + static_cast<size_t>(info.superframeSize) <= at9_size) {
        size_t pos = sf_off;
        for (int f = 0; f < info.framesInSuperframe; ++f) {
            int used = 0;
            int rc = Atrac9Decode(handle, at9_data + pos, frame_buf.data(), &used);
            if (rc != 0 || used <= 0) break;
            pos += static_cast<size_t>(used);
            pcm.insert(pcm.end(), frame_buf.begin(), frame_buf.end());
            if (target_samples > 0 &&
                pcm.size() >= static_cast<size_t>(target_samples) * static_cast<size_t>(info.channels)) {
                break;
            }
        }
        sf_off += static_cast<size_t>(info.superframeSize);
        if (target_samples > 0 &&
            pcm.size() >= static_cast<size_t>(target_samples) * static_cast<size_t>(info.channels)) {
            break;
        }
    }

    Atrac9ReleaseHandle(handle);

    const size_t max_elems = static_cast<size_t>(target_samples) * static_cast<size_t>(info.channels);
    if (max_elems > 0 && pcm.size() > max_elems) {
        pcm.resize(max_elems);
    }

    // Preserve 100% of the natural reverb/decay tail with zero truncation.
    return WriteWavFile(out_wav, pcm, info.samplingRate, info.channels);
}

void ExtractMissingFromFsb(const std::filesystem::path& fsb_path,
                           const std::filesystem::path& sfx_dir) {
    namespace fs = std::filesystem;
    std::ifstream f(fsb_path, std::ios::binary);
    if (!f) return;

    uint8_t header[60]{};
    if (!f.read(reinterpret_cast<char*>(header), sizeof(header))) return;
    if (std::memcmp(header, "FSB5", 4) != 0) return;

    const uint32_t num_samples = ReadU32(header + 8);
    const uint32_t shdr_size = ReadU32(header + 12);
    const uint32_t name_size = ReadU32(header + 16);
    const uint32_t data_size = ReadU32(header + 20);
    const uint32_t mode = ReadU32(header + 24);
    if (mode != 13 || num_samples == 0 || name_size == 0) return;  // 13 == AT9

    std::vector<uint8_t> meta(static_cast<size_t>(shdr_size) + static_cast<size_t>(name_size));
    if (!f.read(reinterpret_cast<char*>(meta.data()), static_cast<std::streamsize>(meta.size()))) {
        return;
    }

    struct SampleMeta {
        uint64_t data_off = 0;
        uint32_t num_pcm_samples = 0;
        uint8_t at9_cfg[4] = {};
        bool has_at9_cfg = false;
        std::string name;
    };

    std::vector<SampleMeta> samples(num_samples);
    size_t pos = 0;
    for (uint32_t i = 0; i < num_samples && pos + 8 <= shdr_size; ++i) {
        const uint64_t sh = ReadU64(meta.data() + pos);
        pos += 8;
        bool has_chunks = (sh & 1ull) != 0;
        samples[i].data_off = ((sh >> 6) & 0x0FFFFFFFull) * 16ull;
        samples[i].num_pcm_samples = static_cast<uint32_t>((sh >> 34) & 0x3FFFFFFFull);

        while (has_chunks && pos + 4 <= shdr_size) {
            const uint32_t ch = ReadU32(meta.data() + pos);
            pos += 4;
            has_chunks = (ch & 1u) != 0;
            const uint32_t ch_size = (ch >> 1) & 0xFFFFFFu;
            const uint32_t ch_type = (ch >> 25) & 0x7Fu;
            if (pos + ch_size > shdr_size) break;
            if (ch_type == 9 && ch_size >= 8) {  // AT9DATA chunk
                std::memcpy(samples[i].at9_cfg, meta.data() + pos + 4, 4);
                samples[i].has_at9_cfg = true;
            }
            pos += ch_size;
        }
    }

    const uint8_t* nt = meta.data() + shdr_size;
    for (uint32_t i = 0; i < num_samples && (i + 1) * 4 <= name_size; ++i) {
        const uint32_t rel = ReadU32(nt + i * 4);
        if (rel < name_size) {
            const char* s = reinterpret_cast<const char*>(nt + rel);
            size_t max_len = name_size - rel;
            size_t len = strnlen(s, max_len);
            samples[i].name.assign(s, len);
        }
    }

    const uint64_t data_base = 60ull + shdr_size + name_size;
    std::error_code ec;
    fs::create_directories(sfx_dir, ec);

    for (const auto& t : kTargetSfx) {
        const fs::path wav_path = sfx_dir / (std::string(t.key) + ".wav");
        if (fs::is_regular_file(wav_path, ec) && fs::file_size(wav_path, ec) > 64) {
            continue;
        }
        for (uint32_t i = 0; i < num_samples; ++i) {
            if (samples[i].name != t.fsb_name || !samples[i].has_at9_cfg) continue;
            const uint64_t next_off =
                (i + 1 < num_samples) ? samples[i + 1].data_off : static_cast<uint64_t>(data_size);
            if (next_off <= samples[i].data_off) break;
            const size_t raw_len = static_cast<size_t>(next_off - samples[i].data_off);
            std::vector<uint8_t> raw(raw_len);
            f.seekg(static_cast<std::streamoff>(data_base + samples[i].data_off));
            if (f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw_len))) {
                DecodeAt9ToWav(raw.data(), raw.size(), samples[i].num_pcm_samples,
                               samples[i].at9_cfg, wav_path);
            }
            break;
        }
    }
}

}  // namespace

std::unordered_map<std::string, std::string> LoadOrExtractUiSfx(
    const std::filesystem::path& game_dir,
    const std::filesystem::path& cache_dir) {
    namespace fs = std::filesystem;
    std::unordered_map<std::string, std::string> result;
    const fs::path sfx_dir = cache_dir / "ui_sfx";

    std::error_code ec;
    const fs::path stamp = sfx_dir / "mapping_v3_fulltail.ok";
    bool all_cached = fs::is_regular_file(stamp, ec);
    for (const auto& t : kTargetSfx) {
        const fs::path wp = sfx_dir / (std::string(t.key) + ".wav");
        if (!fs::is_regular_file(wp, ec) || fs::file_size(wp, ec) <= 64) {
            all_cached = false;
            break;
        }
    }

    if (!all_cached) {
        fs::remove_all(sfx_dir, ec);
        const fs::path fsb = FindMainFsb(game_dir);
        if (!fsb.empty()) {
            ExtractMissingFromFsb(fsb, sfx_dir);
            std::ofstream sf(stamp);
            sf << "v3_fulltail\n";
        }
    }

    for (const auto& t : kTargetSfx) {
        const fs::path wp = sfx_dir / (std::string(t.key) + ".wav");
        auto bytes = ReadFileBytes(wp);
        if (bytes.size() > 64) {
            result[t.key] = "data:audio/wav;base64," + Base64Encode(bytes);
        }
    }
    return result;
}

}  // namespace bb_launcher
