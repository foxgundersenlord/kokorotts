#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace audio {

// Writes a mono 32-bit IEEE float PCM WAV file.
// The Kokoro model outputs 24 kHz float32 samples.
class WavWriter {
public:
    static constexpr uint32_t SAMPLE_RATE = 24000;
    static constexpr uint16_t CHANNELS    = 1;
    static constexpr uint16_t BITS        = 32;

    // Write samples to a WAV file. Returns true on success.
    static bool write(const std::string& path,
                      const std::vector<float>& samples,
                      uint32_t sample_rate = SAMPLE_RATE);

    // Write raw float pointer.
    static bool write(const std::string& path,
                      const float* samples, size_t count,
                      uint32_t sample_rate = SAMPLE_RATE);
};

} // namespace audio
