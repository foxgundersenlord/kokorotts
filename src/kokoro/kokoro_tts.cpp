#include "kokoro_tts.h"
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

namespace fs = std::filesystem;

namespace kokoro {

// Split text into sentence-sized chunks at ./?/! boundaries.
// Each chunk is short enough to stay within the model's token limit.
static std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < text.size(); ++i) {
        cur += text[i];
        char c = text[i];
        // Sentence boundary: .  !  ? — possibly followed by closing quote/paren
        if (c == '.' || c == '!' || c == '?') {
            // Consume any trailing closing punctuation on the same boundary
            while (i + 1 < text.size() &&
                   (text[i+1] == '"' || text[i+1] == '\'' ||
                    text[i+1] == ')' || text[i+1] == ']' ||
                    text[i+1] == '\xe2')) { // start of UTF-8 curly quote
                // For UTF-8 two/three-byte sequences swallow the full codepoint
                unsigned char nc = static_cast<unsigned char>(text[i+1]);
                int seq = (nc < 0x80) ? 1 : (nc < 0xE0) ? 2 : (nc < 0xF0) ? 3 : 4;
                for (int s = 0; s < seq && i + 1 < text.size(); ++s)
                    cur += text[++i];
            }
            // Only split if the next non-space character is uppercase or end-of-string
            size_t j = i + 1;
            while (j < text.size() && text[j] == ' ') ++j;
            if (j >= text.size() || std::isupper(static_cast<unsigned char>(text[j]))) {
                // Trim
                size_t s = cur.find_first_not_of(" \t\r\n");
                if (s != std::string::npos) out.push_back(cur.substr(s));
                cur.clear();
            }
        }
        // Also split on explicit newlines
        else if (c == '\n') {
            size_t s = cur.find_first_not_of(" \t\r\n");
            if (s != std::string::npos) out.push_back(cur.substr(s));
            cur.clear();
        }
    }
    // Remainder
    size_t s = cur.find_first_not_of(" \t\r\n");
    if (s != std::string::npos) out.push_back(cur.substr(s));
    return out;
}

KokoroTTS::KokoroTTS(const Config& cfg)
    : cfg_(cfg),
      device_(cfg.use_gpu && torch::cuda::is_available()
              ? torch::kCUDA : torch::kCPU)
{
    // Load TorchScript model
    try {
        model_ = torch::jit::load(cfg_.model_path, device_);
        model_.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load Kokoro model from '" +
                                 cfg_.model_path + "': " + e.what());
    }

    // Build G2P engine
    misaki::G2P::Config g2p_cfg;
    g2p_cfg.cmudict_path  = cfg_.cmudict_path;
    g2p_cfg.espeak_voice  = cfg_.espeak_voice;
    g2p_cfg.normalize_text = true;
    g2p_ = std::make_unique<misaki::G2P>(g2p_cfg);

    if (!g2p_->cmudict_loaded() && cfg_.cmudict_path.empty())
        std::cerr << "[kokoro] Warning: no CMU dict configured. "
                     "OOV words will use espeak-ng only.\n";
    if (!g2p_->espeak_available())
        std::cerr << "[kokoro] Warning: espeak-ng unavailable. "
                     "Only CMU dict words will be phonemized.\n";

    // Load default voice
    load_voice(cfg_.voice);
}

void KokoroTTS::load_voice(const std::string& voice_name) {
    // Voice tensors are stored as raw float32 binary files (shape: N × 1 × 256).
    // Generate them from .pt files using: scripts/export_model.py
    std::string path = (fs::path(cfg_.voices_dir) / (voice_name + ".bin")).string();
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open voice file: " + path +
                                 "\nRun scripts/export_model.py to generate .bin voice files.");

    std::streamsize nbytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (nbytes % sizeof(float) != 0)
        throw std::runtime_error("Voice file size not divisible by 4: " + path);

    std::vector<float> buf(nbytes / sizeof(float));
    f.read(reinterpret_cast<char*>(buf.data()), nbytes);
    if (!f)
        throw std::runtime_error("Failed to read voice file: " + path);

    // Shape: (N, 1, 256) — N is number of supported sequence lengths.
    int64_t N = static_cast<int64_t>(buf.size()) / 256;
    if (N * 256 != static_cast<int64_t>(buf.size()))
        throw std::runtime_error("Voice file size not a multiple of 256 floats: " + path);

    voice_pack_ = torch::from_blob(buf.data(), {N, 1, 256}, torch::kFloat32)
                      .clone()
                      .to(device_);
    cfg_.voice = voice_name;
}

void KokoroTTS::set_voice(const std::string& voice_name) {
    load_voice(voice_name);
}

std::vector<std::string> KokoroTTS::list_voices() const {
    std::vector<std::string> voices;
    if (!fs::exists(cfg_.voices_dir)) return voices;
    for (auto& entry : fs::directory_iterator(cfg_.voices_dir)) {
        if (entry.path().extension() == ".bin")
            voices.push_back(entry.path().stem().string());
    }
    std::sort(voices.begin(), voices.end());
    return voices;
}

torch::Tensor KokoroTTS::select_style(size_t token_count) const {
    // Kokoro indexes the voice pack at (len(tokens) - 1), clamped to valid range.
    int64_t N = voice_pack_.size(0);
    int64_t idx = static_cast<int64_t>(token_count) - 1;
    idx = std::max<int64_t>(0, std::min(idx, N - 1));

    // voice_pack_ shape: (N, 1, 256)  → slice at idx → (1, 256) → squeeze → (256,)
    return voice_pack_[idx].squeeze(0);
}

std::vector<float> KokoroTTS::synthesize(const std::string& text) const {
    // ── 1. G2P ──────────────────────────────────────────────────────────────
    std::vector<int64_t> token_ids = g2p_->text_to_tokens(text);
    if (token_ids.size() <= 2) {
        std::cerr << "[kokoro] Warning: empty token sequence for: " << text << "\n";
        return {};
    }

    // ── 2. Build input tensors ───────────────────────────────────────────────
    const int64_t seq_len = static_cast<int64_t>(token_ids.size());

    torch::Tensor input_ids = torch::from_blob(
        token_ids.data(),
        {1, seq_len},
        torch::TensorOptions().dtype(torch::kInt64)
    ).clone().to(device_);

    // Style vector: shape (1, 256)
    torch::Tensor ref_s = select_style(token_ids.size())
                              .unsqueeze(0)   // → (1, 256)
                              .to(device_);

    // ── 3. Inference ─────────────────────────────────────────────────────────
    // The traced KokoroWrapper takes (input_ids, ref_s) and returns audio directly.
    // Speed is baked into the trace at export time (default 1.0).
    torch::Tensor audio;
    torch::Tensor ts;
    {
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {input_ids, ref_s};
        auto outputs = model_.forward(inputs).toTuple();

        audio = outputs->elements()[0].toTensor();
        ts = outputs->elements()[0].toTensor();

    }

    // audio shape: (1, num_samples) — squeeze batch dim
    audio = audio.squeeze(0).to(torch::kFloat32).cpu().contiguous();

    const float* ptr = audio.data_ptr<float>();
    return std::vector<float>(ptr, ptr + audio.numel());
}

void KokoroTTS::synthesize_to_wav(const std::string& text,
                                   const std::string& output_path) const {
    auto sentences = split_sentences(text);
    if (sentences.empty()) sentences.push_back(text); // fallback: treat as one chunk

    // 0.25 s of silence between sentences (24 kHz)
    static const int SILENCE_SAMPLES = 6000;

    std::vector<float> all_samples;
    bool first = true;
    for (const auto& sent : sentences) {
        auto chunk = synthesize(sent);
        if (chunk.empty()) {
            std::cerr << "[kokoro] No audio for chunk: " << sent << "\n";
            continue;
        }
        if (!first)
            all_samples.insert(all_samples.end(), SILENCE_SAMPLES, 0.0f);
        all_samples.insert(all_samples.end(), chunk.begin(), chunk.end());
        first = false;
    }

    if (all_samples.empty()) {
        std::cerr << "[kokoro] No audio generated for: " << text << "\n";
        return;
    }
    if (!audio::WavWriter::write(output_path, all_samples))
        throw std::runtime_error("Failed to write WAV file: " + output_path);
}

} // namespace kokoro
