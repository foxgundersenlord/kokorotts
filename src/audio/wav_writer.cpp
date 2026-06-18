#include "wav_writer.h"
#include <fstream>
#include <cstring>

namespace audio {

namespace {

// Helpers to write little-endian integers
void write_u16(std::ostream& os, uint16_t v) {
    os.put(static_cast<char>(v & 0xFF));
    os.put(static_cast<char>((v >> 8) & 0xFF));
}
void write_u32(std::ostream& os, uint32_t v) {
    os.put(static_cast<char>(v & 0xFF));
    os.put(static_cast<char>((v >>  8) & 0xFF));
    os.put(static_cast<char>((v >> 16) & 0xFF));
    os.put(static_cast<char>((v >> 24) & 0xFF));
}

bool write_impl(const std::string& path, const float* samples, size_t count, uint32_t sr) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;

    const uint32_t data_bytes  = static_cast<uint32_t>(count * sizeof(float));
    const uint32_t fmt_chunk   = 18; // PCM fmt chunk is 16 bytes + extra 2 for IEEE float
    const uint32_t riff_size   = 4 + (8 + fmt_chunk) + (8 + data_bytes);
    const uint16_t block_align = WavWriter::CHANNELS * (WavWriter::BITS / 8);
    const uint32_t byte_rate   = sr * WavWriter::CHANNELS * (WavWriter::BITS / 8);

    // RIFF header
    f.write("RIFF", 4);
    write_u32(f, riff_size);
    f.write("WAVE", 4);

    // fmt chunk — format 3 = IEEE float
    f.write("fmt ", 4);
    write_u32(f, fmt_chunk);
    write_u16(f, 3);                     // WAVE_FORMAT_IEEE_FLOAT
    write_u16(f, WavWriter::CHANNELS);
    write_u32(f, sr);
    write_u32(f, byte_rate);
    write_u16(f, block_align);
    write_u16(f, WavWriter::BITS);
    write_u16(f, 0);                     // cbSize (extra bytes = 0)

    // fact chunk (required for non-PCM formats)
    f.write("fact", 4);
    write_u32(f, 4);
    write_u32(f, static_cast<uint32_t>(count)); // number of samples per channel

    // data chunk
    f.write("data", 4);
    write_u32(f, data_bytes);
    f.write(reinterpret_cast<const char*>(samples), data_bytes);

    return f.good();
}

} // namespace

bool WavWriter::write(const std::string& path, const std::vector<float>& samples, uint32_t sr) {
    return write_impl(path, samples.data(), samples.size(), sr);
}

bool WavWriter::write(const std::string& path, const float* samples, size_t count, uint32_t sr) {
    return write_impl(path, samples, count, sr);
}

} // namespace audio
