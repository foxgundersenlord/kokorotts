#include "kokoro/kokoro_tts.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <filesystem>
#include <unistd.h>

namespace fs = std::filesystem;

// Returns the directory containing this executable.
static fs::path self_dir() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) { buf[n] = '\0'; return fs::path(buf).parent_path(); }
    return fs::current_path();
}

// If `p` doesn't exist as given, try relative to the binary dir and its parent.
static std::string resolve(const std::string& p) {
    if (p.empty() || fs::exists(p)) return p;
    fs::path base = self_dir();
    for (int up = 0; up < 2; ++up, base = base.parent_path()) {
        auto candidate = base / p;
        if (fs::exists(candidate)) return candidate.lexically_normal().string();
    }
    return p; // keep original so error messages show what was tried
}

static void usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " [OPTIONS] \"text to synthesize\"\n\n"
        "Options:\n"
        "  --model    PATH   TorchScript model file  (default: models/kokoro_traced.pt)\n"
        "  --voices   DIR    Voice tensors directory  (default: models/voices)\n"
        "  --dict     PATH   CMU pronouncing dict     (default: data/cmudict.dict)\n"
        "  --voice    NAME   Voice name               (default: af_heart)\n"
        "  --speed    FLOAT  Speech speed multiplier  (default: 1.0)\n"
        "  --output   PATH   Output WAV file          (default: output.wav)\n"
        "  --gpu             Use CUDA if available\n"
        "  --list-voices     Print available voices and exit\n"
        "  --help            Show this message\n\n"
        "Example:\n"
        "  " << prog << " --voice af_heart --output hello.wav \"Hello, world!\"\n";
}

int main(int argc, char* argv[]) {
    kokoro::KokoroTTS::Config cfg;
    cfg.model_path   = "models/kokoro_traced.pt";
    cfg.voices_dir   = "models/voices";
    cfg.cmudict_path = "data/cmudict.dict";
    cfg.voice        = "af_heart";
    cfg.speed        = 1.0f;

    std::string output_path = "output.wav";
    std::string text;
    bool list_voices = false;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* flag) { return strcmp(argv[i], flag) == 0; };
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing argument for " << argv[i] << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if      (arg("--help"))        { usage(argv[0]); return 0; }
        else if (arg("--list-voices")) list_voices = true;
        else if (arg("--model"))       cfg.model_path   = next();
        else if (arg("--voices"))      cfg.voices_dir   = next();
        else if (arg("--dict"))        cfg.cmudict_path = next();
        else if (arg("--voice"))       cfg.voice        = next();
        else if (arg("--output"))      output_path      = next();
        else if (arg("--gpu"))         cfg.use_gpu      = true;
        else if (arg("--speed"))       cfg.speed        = std::stof(next());
        else if (argv[i][0] != '-') {
            if (!text.empty()) text += ' ';
            text += argv[i];
        }
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    try {
        cfg.model_path   = resolve(cfg.model_path);
        cfg.voices_dir   = resolve(cfg.voices_dir);
        cfg.cmudict_path = resolve(cfg.cmudict_path);

        kokoro::KokoroTTS tts(cfg);

        if (list_voices) {
            std::cout << "Available voices in '" << cfg.voices_dir << "':\n";
            for (auto& v : tts.list_voices()) std::cout << "  " << v << "\n";
            return 0;
        }

        if (text.empty()) {
            std::cerr << "No text provided.\n";
            usage(argv[0]);
            return 1;
        }

        std::cout << "Synthesizing: \"" << text << "\"\n"
                  << "Voice: " << cfg.voice << "  Speed: " << cfg.speed << "\n";

        tts.synthesize_to_wav(text, output_path);
        std::cout << "Wrote: " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
