#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace misaki {

// Converts an IPA phoneme string into Kokoro token IDs.
// Pads with token 0 at both ends: [0, tok1, tok2, ..., 0].
// Capped at 512 tokens total (model maximum).
class Tokenizer {
public:
    // Returns padded token sequence. Unknown codepoints are skipped.
    std::vector<int64_t> encode(const std::string& phonemes) const;

    // Decode token IDs back to UTF-8 string (for debugging).
    std::string decode(const std::vector<int64_t>& tokens) const;

private:
    // Extracts one UTF-8 encoded codepoint starting at s[pos], advancing pos.
    static std::string next_codepoint(const std::string& s, size_t& pos);
};

} // namespace misaki
