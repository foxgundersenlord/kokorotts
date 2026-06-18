#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace misaki {

// Loads the CMU Pronouncing Dictionary and converts ARPABET phonemes to
// Kokoro token IDs.
//
// ARPABET stress digits: 0 = unstressed, 1 = primary, 2 = secondary.
// Stress markers (ˈ/ˌ) are inserted as separate tokens before the vowel.
class CmuDict {
public:
    // Load from a cmudict.dict file (one entry per line: WORD  P1 P2 ...).
    bool load(const std::string& path);

    bool loaded() const { return loaded_; }

    // Returns true if the word is in the dictionary.
    bool contains(const std::string& word) const;

    // Looks up a word (case-insensitive) and returns Kokoro token IDs.
    // Returns empty vector if not found.
    std::vector<int64_t> lookup(const std::string& word) const;

    // Converts a sequence of ARPABET phones (e.g. ["HH","EH1","L","OW0"])
    // directly to token IDs.
    static std::vector<int64_t> arpabet_to_tokens(const std::vector<std::string>& phones);

    // Converts a single ARPABET phone to token IDs (may return 1–3 tokens).
    static std::vector<int64_t> phone_to_tokens(const std::string& phone);

private:
    // key: uppercase word, value: flat token sequence
    std::unordered_map<std::string, std::vector<int64_t>> dict_;
    bool loaded_ = false;

    static std::string to_upper(const std::string& s);
};

} // namespace misaki
