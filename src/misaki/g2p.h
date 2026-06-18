#pragma once
#include "text_normalizer.h"
#include "cmudict.h"
#include "tokenizer.h"
#include <string>
#include <vector>
#include <cstdint>

namespace misaki {

// Grapheme-to-Phoneme engine — reimplements Misaki's English G2P pipeline.
//
// Pipeline:
//   1. TextNormalizer  — expand numbers, currency, abbreviations
//   2. Word tokenization — split on whitespace and punctuation
//   3. CmuDict lookup — ARPABET → Kokoro token IDs
//   4. espeak-ng fallback — IPA → Kokoro token IDs (if KOKORO_HAS_ESPEAK)
//   5. Tokenizer::encode — add padding tokens [0, ..., 0]
class G2P {
public:
    struct Config {
        std::string cmudict_path;   // path to cmudict.dict (empty = skip)
        std::string espeak_voice = "en-us"; // espeak-ng voice identifier
        bool normalize_text = true;
    };

    G2P();
    explicit G2P(const Config& cfg);
    ~G2P();

    // Full pipeline: raw text → padded Kokoro token IDs.
    std::vector<int64_t> text_to_tokens(const std::string& text) const;

    // Raw IPA string → token IDs (without padding).
    std::vector<int64_t> ipa_to_tokens(const std::string& ipa) const;

    bool cmudict_loaded() const;
    bool espeak_available() const;

    // Full pipeline returning both tokens and the word list in order.
    struct TokenizeResult {
        std::vector<int64_t>     token_ids;
        std::vector<std::string> words;     // one entry per SPACE-delimited group
    };
    TokenizeResult text_to_tokens_ex(const std::string& text) const;

private:
    Config cfg_;
    TextNormalizer normalizer_;
    CmuDict cmudict_;
    Tokenizer tokenizer_;
    bool espeak_ok_ = false;

    void init_espeak();

    // Convert a single word to tokens using CMU dict (primary) or espeak (fallback).
    std::vector<int64_t> word_to_tokens(const std::string& word) const;

    // Get IPA for a word via espeak-ng (returns empty string if unavailable).
    std::string espeak_ipa(const std::string& word) const;

    // Map a single IPA codepoint string to token IDs (handles substitutions).
    std::vector<int64_t> ipa_char_to_tokens(const std::string& cp) const;
};

} // namespace misaki
