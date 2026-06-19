#include "kokoro_tts.h"
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace kokoro {

// ---------------------------------------------------------------------------
// Sentence splitting
// ---------------------------------------------------------------------------

static std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < text.size(); ++i) {
        cur += text[i];
        char c = text[i];
        if (c == '.' || c == '!' || c == '?') {
            while (i + 1 < text.size() &&
                   (text[i+1] == '"' || text[i+1] == '\'' ||
                    text[i+1] == ')' || text[i+1] == ']' ||
                    text[i+1] == '\xe2')) {
                unsigned char nc = static_cast<unsigned char>(text[i+1]);
                int seq = (nc < 0x80) ? 1 : (nc < 0xE0) ? 2 : (nc < 0xF0) ? 3 : 4;
                for (int s = 0; s < seq && i + 1 < text.size(); ++s)
                    cur += text[++i];
            }
            size_t j = i + 1;
            while (j < text.size() && text[j] == ' ') ++j;
            if (j >= text.size() || std::isupper(static_cast<unsigned char>(text[j]))) {
                size_t s = cur.find_first_not_of(" \t\r\n");
                if (s != std::string::npos) out.push_back(cur.substr(s));
                cur.clear();
            }
        } else if (c == '\n') {
            size_t s = cur.find_first_not_of(" \t\r\n");
            if (s != std::string::npos) out.push_back(cur.substr(s));
            cur.clear();
        }
    }
    size_t s = cur.find_first_not_of(" \t\r\n");
    if (s != std::string::npos) out.push_back(cur.substr(s));
    return out;
}

// ---------------------------------------------------------------------------
// Word extraction (mirrors G2P word tokenizer — splits on whitespace)
// ---------------------------------------------------------------------------

static std::vector<std::string> extract_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream ss(text);
    std::string w;
    while (ss >> w) {
        // Strip leading/trailing punctuation to get a clean label
        size_t start = 0, end = w.size();
        while (start < end && !std::isalnum(static_cast<unsigned char>(w[start])) && w[start] != '\'')
            ++start;
        while (end > start && !std::isalnum(static_cast<unsigned char>(w[end-1])) && w[end-1] != '\'')
            --end;
        if (start < end)
            words.push_back(w.substr(start, end - start));
        else
            words.push_back(w); // keep punctuation-only tokens as-is
    }
    return words;
}

// ---------------------------------------------------------------------------
// Word timestamp building
// ---------------------------------------------------------------------------

// Space token ID in Kokoro vocab
static constexpr int64_t SPACE_TOKEN = 16;
static constexpr int64_t PAD_TOKEN   = 0;

// The token sequence fed to the model is [0, tok1, tok2, …, 0].
// The timestamp tensor has one row per non-pad token position.
// We group consecutive non-space/non-pad tokens into words by tracking
// how many phoneme tokens each word contributes.
//
// Because we don't have a direct token→word map from G2P (it's embedded in
// g2p.cpp's text_to_tokens loop), we re-derive word boundaries by scanning
// the token sequence for SPACE_TOKEN separators.
std::vector<WordTimestamp> KokoroTTS::build_word_timestamps(
    const std::vector<int64_t>&    token_ids,
    const torch::Tensor&           ts_tensor,
    const std::vector<std::string>& words)
{
    if (ts_tensor.dim() != 2 || ts_tensor.size(1) < 3) return {};

    const int64_t N = ts_tensor.size(0);
    const float* ts = ts_tensor.to(torch::kFloat32).cpu().contiguous().data_ptr<float>();

    // The model receives token_ids as-is (including both pad tokens at index 0
    // and end), and ts_tensor has one row per token position — including pads.
    // PAD tokens have real (but meaningless) duration entries; we must skip them
    // when building word spans, but we still consume their ts_tensor row so that
    // the row cursor stays aligned with the token position.
    struct TokSpan { double start; double end; };
    std::vector<TokSpan> spans;
    spans.reserve(token_ids.size());

    for (size_t i = 0; i < token_ids.size(); ++i) {
        if ((int64_t)i < N) {
            spans.push_back({static_cast<double>(ts[i * 4 + 1]),
                             static_cast<double>(ts[i * 4 + 2])});
        } else {
            spans.push_back({0.0, 0.0});
        }
    }

    std::vector<WordTimestamp> result;
    double word_start = -1.0, word_end = 0.0;
    size_t word_idx = 0;

    auto flush_word = [&]() {
        if (word_start < 0.0 || word_idx >= words.size()) return;
        result.push_back({words[word_idx], word_start, word_end});
        ++word_idx;
        word_start = -1.0;
        word_end   = 0.0;  // BUG FIX: reset word_end so stale end doesn't leak
                           // into the next word if it has no phonemes
    };

    for (size_t i = 0; i < token_ids.size(); ++i) {
        int64_t tok = token_ids[i];

        // Pad tokens are not word boundaries; skip without flushing.
        if (tok == PAD_TOKEN) continue;

        if (tok == SPACE_TOKEN) {
            flush_word();
            continue;
        }

        // ts_tensor's start/end are already cumulative from the start of the
        // audio (built via torch.cumsum over the full padded sequence in
        // export_model.py), so they need no offset correction here.
        double ts_start = spans[i].start;
        double ts_end   = spans[i].end;
        if (word_start < 0.0) word_start = ts_start;
        if (ts_end > word_end) word_end = ts_end;
    }
    
    flush_word(); // last word — no trailing SPACE token in sequence

    // The trailing pad token (and any sentence-final punctuation) can carry
    // real predicted duration that extends past the last word's tokens —
    // this is audio the vocoder actually renders (decay/tail), so the last
    // reported word's end should not be shorter than it.
    if (!result.empty() && N > 0) {
        double full_end = static_cast<double>(ts[(N - 1) * 4 + 2]); // end_sec of last row
        if (full_end > result.back().end_sec) {
            result.back().end_sec = full_end;
        }
    }

    return result;

    return result;
}

// ---------------------------------------------------------------------------
// JSON serialisation (no external library needed)
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

std::string KokoroTTS::timestamps_to_json(const std::vector<WordTimestamp>& words) {
    std::ostringstream os;
    os << "[\n";
    for (size_t i = 0; i < words.size(); ++i) {
        const auto& w = words[i];
        os << "  {";
        os << "\"word\": \"" << json_escape(w.word) << "\", ";
        os << "\"start\": " << w.start_sec << ", ";
        os << "\"end\": "   << w.end_sec;
        os << "}";
        if (i + 1 < words.size()) os << ",";
        os << "\n";
    }
    os << "]\n";
    return os.str();
}

// ---------------------------------------------------------------------------
// Constructor / voice loading
// ---------------------------------------------------------------------------

KokoroTTS::KokoroTTS(const Config& cfg)
    : cfg_(cfg),
      device_(cfg.use_gpu && torch::cuda::is_available()
              ? torch::kCUDA : torch::kCPU)
{
    try {
        model_ = torch::jit::load(cfg_.model_path, device_);
        model_.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load Kokoro model from '" +
                                 cfg_.model_path + "': " + e.what());
    }

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

    load_voice(cfg_.voice);
}

void KokoroTTS::load_voice(const std::string& voice_name) {
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
    int64_t N   = voice_pack_.size(0);
    // token_count includes leading + trailing pad; match Python which indexes
    // by the number of real tokens (no pads)
    int64_t idx = static_cast<int64_t>(token_count) - 2;
    idx = std::max<int64_t>(0, std::min(idx, N - 1));
    return voice_pack_[idx].squeeze(0);
}

// ---------------------------------------------------------------------------
// Core synthesis
// ---------------------------------------------------------------------------

KokoroTTS::SynthResult KokoroTTS::synthesize(const std::string& text) const {
    // ── 1. G2P ──────────────────────────────────────────────────────────────
    auto [token_ids, words] = g2p_->text_to_tokens_ex(text);  // single pass
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

    torch::Tensor ref_s = select_style(token_ids.size())
                              .unsqueeze(0)
                              .to(device_);

    // ── 3. Inference ─────────────────────────────────────────────────────────
    // KokoroWrapper returns (audio, ts_tensor) — see scripts/export_model.py.
    torch::Tensor audio;
    torch::Tensor ts_tensor;
    {
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {input_ids, ref_s};
        auto outputs = model_.forward(inputs).toTuple();
        audio     = outputs->elements()[0].toTensor();  // waveform
        ts_tensor = outputs->elements()[1].toTensor();  // [N, 4] timestamps
    }

    audio = audio.squeeze(0).to(torch::kFloat32).cpu().contiguous();

    SynthResult result;

    const float* ptr = audio.data_ptr<float>();
    result.audio.assign(ptr, ptr + audio.numel());

    // ── 4. Build word timestamps ─────────────────────────────────────────────
    ts_tensor = ts_tensor.to(torch::kFloat32).cpu().contiguous();
    result.words = build_word_timestamps(token_ids, ts_tensor, words);

    return result;
}

// ---------------------------------------------------------------------------
// WAV + JSON output
// ---------------------------------------------------------------------------

void KokoroTTS::synthesize_to_wav(const std::string& text,
                                   const std::string& output_path) const {
    auto sentences = split_sentences(text);
    if (sentences.empty()) sentences.push_back(text);

    static const int SILENCE_SAMPLES = 6000; // 0.25 s at 24 kHz

    std::vector<float>         all_samples;
    std::vector<WordTimestamp> all_words;
    double time_offset = 0.0;
    bool first = true;

    for (const auto& sent : sentences) {
        auto chunk = synthesize(sent);
        if (chunk.audio.empty()) continue;

        if (!first) {
            all_samples.insert(all_samples.end(), SILENCE_SAMPLES, 0.0f);
            time_offset += static_cast<double>(SILENCE_SAMPLES) /
                        audio::WavWriter::SAMPLE_RATE;
        }

        for (auto& w : chunk.words) {
            w.start_sec += time_offset;
            w.end_sec   += time_offset;
            all_words.push_back(w);
        }

        // Advance offset by the ACTUAL audio length written for this chunk,
        // not by where the last word happened to end. Any tail audio after
        // the last word's predicted end (vocoder decay, etc.) must still be
        // accounted for or every subsequent sentence's timestamps drift.
        time_offset += static_cast<double>(chunk.audio.size()) /
                    audio::WavWriter::SAMPLE_RATE;

        all_samples.insert(all_samples.end(),
                        chunk.audio.begin(), chunk.audio.end());
        first = false;
    }

    if (all_samples.empty()) {
        std::cerr << "[kokoro] No audio generated for: " << text << "\n";
        return;
    }

    // Write WAV
    if (!audio::WavWriter::write(output_path, all_samples))
        throw std::runtime_error("Failed to write WAV file: " + output_path);

    // Write JSON timestamps alongside the WAV
    std::string json_path = output_path + ".json";
    std::ofstream jf(json_path);
    if (!jf.is_open())
        throw std::runtime_error("Failed to write timestamp JSON: " + json_path);
    jf << timestamps_to_json(all_words);
    if (!jf)
        throw std::runtime_error("Error while writing timestamp JSON: " + json_path);

    std::cout << "Wrote: " << json_path << "\n";
}

} // namespace kokoro