# Kokoro TTS (C++ / LibTorch)

A standalone C++ command-line tool for running [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech entirely offline via LibTorch, with a from-scratch reimplementation of Misaki's English grapheme-to-phoneme (G2P) pipeline. No Python required at inference time.

Given a string of text, it synthesizes natural-sounding 24 kHz speech to a WAV file and writes a companion JSON file with per-word timestamps.

```
./kokoro_tts --voice af_heart --output hello.wav "Hello, world!"
```

## Features

- **Self-contained C++ inference** — loads a traced TorchScript model and voice tensors directly via LibTorch; no Python at runtime.
- **CPU-only by default** — links against `libtorch_cpu` only, so it builds and runs without a CUDA toolkit installed (see [GPU support](#gpu-support) to enable CUDA).
- **Full G2P pipeline reimplemented in C++**:
  - Text normalization (numbers, currency, ordinals, common abbreviations)
  - CMU Pronouncing Dictionary lookup (ARPABET → Kokoro phoneme tokens)
  - Optional espeak-ng fallback (IPA → tokens) for out-of-vocabulary words
- **Per-word timestamps** — every synthesis call returns word-level start/end times, written alongside the WAV as a `.json` file.
- **Multi-sentence synthesis** — input text is split into sentences, synthesized independently, and concatenated with short silence gaps, with timestamps correctly offset across the whole clip.
- **Voice switching at runtime** via `set_voice()` / `--voice`.

## How it works

```
text
  │
  ▼
TextNormalizer        (numbers, currency, ordinals, abbreviations)
  │
  ▼
sentence splitter      (src/kokoro/kokoro_tts.cpp)
  │
  ▼
G2P                    CMU dict lookup → ARPABET → tokens
  │                    (espeak-ng IPA fallback for OOV words)
  ▼
Tokenizer              pad with leading/trailing token 0, cap at 512
  │
  ▼
KokoroTTS model        TorchScript trace: (input_ids, ref_s) → (audio, ts_tensor)
  │
  ▼
WavWriter + JSON       24kHz float32 WAV + per-word timestamps
```

The voice tensor (`ref_s`) is selected from the loaded voice pack by indexing on token sequence length, matching Kokoro's reference Python implementation.

## Requirements

- CMake ≥ 3.18
- A C++17 compiler
- [LibTorch](https://pytorch.org/) (CPU build is sufficient; see below)
- *(optional)* `libespeak-ng-dev` — enables phonemization of words not found in the CMU dictionary
- A traced Kokoro model and voice files (see [Exporting the model](#exporting-the-model))

## Building

1. **Get LibTorch.** Either download the CPU-only distribution and extract it as `libtorch/` in the project root:

   ```bash
   python scripts/download_assets.py   # also fetches the CMU dictionary
   ```

   This prints the exact LibTorch download URL and version. Alternatively, point CMake at an existing PyTorch install:

   ```bash
   cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
   ```

2. **(Optional) Install espeak-ng** for OOV fallback phonemization:

   ```bash
   sudo apt install libespeak-ng-dev
   ```

   The build will warn and continue without it if missing — words outside the CMU dictionary will simply be skipped during synthesis.

3. **Configure and build:**

   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

   If `libtorch/` exists in the project root, CMake auto-detects it as `CMAKE_PREFIX_PATH`.

## Exporting the model

The C++ binary expects a TorchScript trace, not the raw HuggingFace checkpoint. Export it once from Python:

```bash
pip install "transformers<5.0.0" spacy torch huggingface_hub loguru
python scripts/export_model.py --output-dir models --speed 1.0
```

This:
1. Downloads `kokoro-v1_0.pth` from `hexgrad/Kokoro-82M` on HuggingFace
2. Wraps the model so speed is baked in at trace time and timestamps are computed as tensor ops (so the trace generalizes to any input length)
3. Traces and saves `models/kokoro_traced.pt`
4. Downloads voice packs and converts each `voices/*.pt` to a raw little-endian float32 `.bin` (shape `N×1×256`) for fast loading in C++

Re-run with a different `--speed` to bake in a different default rate (the C++ side also exposes `--speed` as a runtime multiplier on top of this).

## Running

```bash
./kokoro_tts [OPTIONS] "text to synthesize"
```

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | `models/kokoro_traced.pt` | TorchScript model file |
| `--voices DIR` | `models/voices` | Directory of voice `.bin` tensors |
| `--dict PATH` | `data/cmudict.dict` | CMU pronouncing dictionary |
| `--voice NAME` | `af_heart` | Voice name (filename stem, no extension) |
| `--speed FLOAT` | `1.0` | Speech speed multiplier |
| `--output PATH` | `output.wav` | Output WAV path |
| `--gpu` | off | Use CUDA if available |
| `--list-voices` | — | Print available voices and exit |
| `--help` | — | Show usage |

Relative paths for `--model`, `--voices`, and `--dict` are resolved against the executable's own directory (and its parent) if they don't exist relative to the current working directory — so the binary can be run from anywhere as long as assets sit alongside it.

**Example:**

```bash
./kokoro_tts --voice af_heart --output hello.wav "Hello, world!"
```

This produces:
- `hello.wav` — 24 kHz mono float32 PCM audio
- `hello.wav.json` — per-word timestamps:

  ```json
  [
    {"word": "Hello", "start": 0.0125, "end": 0.4125},
    {"word": "world", "start": 0.4625, "end": 0.9125}
  ]
  ```

**List available voices:**

```bash
./kokoro_tts --list-voices --voices models/voices
```

## GPU support

The default build links only `torch_cpu` and `c10`, deliberately avoiding `libtorch_cuda.so` and its CUDA toolkit dependencies — this lets the project build cleanly on machines without CUDA installed, even when using a CUDA-enabled LibTorch distribution. (`CMakeLists.txt` pre-declares a stub `torch::cudart` target so `find_package(Torch)` doesn't immediately fail in CUDA-enabled libtorch builds.)

To enable real GPU inference:
1. Install the CUDA toolkit version matching your LibTorch build.
2. In `CMakeLists.txt`, link `${TORCH_LIBRARIES}` instead of `torch_cpu` directly.
3. Build and run with `--gpu`.

## Project layout

```
CMakeLists.txt
scripts/
  export_model.py        Export traced model + convert voice packs
  download_assets.py     Fetch CMU dict / print LibTorch setup instructions
  modules.py              StyleTTS2-derived model modules (encoder, predictor)
  istftnet.py              Vocoder (iSTFTNet/HiFi-GAN-style generator)
  custom_stft.py           Conv-based STFT/iSTFT (ONNX-friendly alternative)
src/
  main.cpp                 CLI entry point
  kokoro/
    kokoro_tts.{h,cpp}      High-level synthesis API, sentence splitting,
                            word-timestamp construction, WAV/JSON output
  misaki/
    text_normalizer.{h,cpp} Number/currency/ordinal/abbreviation expansion
    tokenizer.{h,cpp}       IPA string → token ID encoding
    cmudict.{h,cpp}         CMU dict loading + ARPABET → token mapping
    g2p.{h,cpp}              Full G2P pipeline orchestration + espeak-ng fallback
    vocab.h                  Kokoro phoneme vocabulary (token ID table)
  audio/
    wav_writer.{h,cpp}       Minimal float32 PCM WAV writer
```

## Using `KokoroTTS` as a library

```cpp
#include "kokoro/kokoro_tts.h"

kokoro::KokoroTTS::Config cfg;
cfg.model_path   = "models/kokoro_traced.pt";
cfg.voices_dir    = "models/voices";
cfg.cmudict_path = "data/cmudict.dict";
cfg.voice         = "af_heart";

kokoro::KokoroTTS tts(cfg);
tts.synthesize_to_wav("Hello world.", "output.wav");
// → output.wav + output.wav.json

// Or work with raw samples + timestamps directly:
auto result = tts.synthesize("Hello world.");
// result.audio  -> std::vector<float>  (24kHz mono)
// result.words  -> std::vector<WordTimestamp>
```

## Notes

- Out-of-vocabulary words are silently skipped if neither the CMU dictionary nor espeak-ng can resolve them — build with espeak-ng for best coverage.
- The CMU dictionary keeps only the first pronunciation listed per word; alternate-pronunciation markers (`WORD(2)`) are ignored.
- Token sequences are capped at 512 (the model's maximum); longer input is truncated.
