#pragma once
#include <string>
#include <vector>
#include <memory>
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/cuda.h>
#include "misaki/g2p.h"
#include "audio/wav_writer.h"

namespace kokoro {

struct WordTimestamp {
    std::string word;
    double start_sec;
    double end_sec;
};

// High-level Kokoro-82M TTS interface.
//
// Usage:
//   KokoroTTS::Config cfg;
//   cfg.model_path   = "models/kokoro_traced.pt";
//   cfg.voices_dir   = "models/voices";
//   cfg.cmudict_path = "data/cmudict.dict";
//   cfg.voice        = "af_heart";
//
//   KokoroTTS tts(cfg);
//   tts.synthesize_to_wav("Hello world.", "output.wav");
//   // Also writes output.wav.json with per-word timestamps
class KokoroTTS {
public:
    struct Config {
        std::string model_path;           // TorchScript .pt model
        std::string voices_dir;           // Directory containing voice .pt tensors
        std::string cmudict_path;         // CMU pronouncing dict (optional)
        std::string voice = "af_heart";   // Default voice name (filename without .pt)
        std::string espeak_voice = "en-us";
        float speed = 1.0f;
        bool use_gpu = false;
    };

    struct SynthResult {
        std::vector<float>         audio;
        std::vector<WordTimestamp> words;
    };

    explicit KokoroTTS(const Config& cfg);

    // Synthesize text → audio samples (float32, 24 kHz mono) + word timestamps.
    SynthResult synthesize(const std::string& text) const;

    // Synthesize text and write to a WAV file.
    // Also writes <output_path>.json containing per-word timestamps.
    void synthesize_to_wav(const std::string& text,
                           const std::string& output_path) const;

    // Change the active voice at runtime. voice_name is the stem (no extension).
    void set_voice(const std::string& voice_name);

    // List voice .bin files found in voices_dir.
    std::vector<std::string> list_voices() const;

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    mutable torch::jit::Module model_; // mutable: jit::Module::forward is non-const
    torch::Tensor voice_pack_; // shape: (N, 1, 256) — the loaded voice tensor
    std::unique_ptr<misaki::G2P> g2p_;
    torch::Device device_;

    void load_voice(const std::string& voice_name);

    // Select style vector from voice_pack based on sequence length.
    // Returns tensor of shape (1, 256).
    torch::Tensor select_style(size_t token_count) const;

    // Build word-level timestamps by mapping token timestamps back to words.
    // token_ids: the full padded token sequence fed to the model.
    // ts_tensor:  shape [N_tokens, 4] — [token_id, start_sec, end_sec, dur_frames].
    // words:      the space-split word list in order (from G2P normalization).
    static std::vector<WordTimestamp> build_word_timestamps(
        const std::vector<int64_t>&   token_ids,
        const torch::Tensor&          ts_tensor,
        const std::vector<std::string>& words,
        double                         audio_duration_sec);   // ← add this

    // Serialize a list of WordTimestamps to a JSON string (no external deps).
    static std::string timestamps_to_json(
        const std::vector<WordTimestamp>& words);
};

} // namespace kokoro