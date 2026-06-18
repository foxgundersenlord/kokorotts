#include "tokenizer.h"
#include "vocab.h"
#include <stdexcept>

namespace misaki {

static constexpr size_t MAX_TOKENS = 512;

std::string Tokenizer::next_codepoint(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return {};
    unsigned char c = static_cast<unsigned char>(s[pos]);
    int len;
    if      (c < 0x80) len = 1;
    else if (c < 0xE0) len = 2;
    else if (c < 0xF0) len = 3;
    else                len = 4;
    if (pos + len > s.size()) { pos = s.size(); return {}; }
    std::string cp = s.substr(pos, len);
    pos += len;
    return cp;
}

std::vector<int64_t> Tokenizer::encode(const std::string& phonemes) const {
    const auto& V = vocab();
    std::vector<int64_t> tokens;
    tokens.reserve(phonemes.size() + 2);
    tokens.push_back(0); // leading pad

    size_t pos = 0;
    while (pos < phonemes.size() && tokens.size() < MAX_TOKENS - 1) {
        std::string cp = next_codepoint(phonemes, pos);
        if (cp.empty()) break;
        auto it = V.find(cp);
        if (it != V.end()) tokens.push_back(it->second);
        // Unknown codepoints are silently skipped (e.g. language-specific chars)
    }

    tokens.push_back(0); // trailing pad
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int64_t>& tokens) const {
    const auto& RV = vocab_reverse();
    std::string result;
    for (int64_t id : tokens) {
        if (id == 0) continue;
        auto it = RV.find(id);
        if (it != RV.end()) result += it->second;
    }
    return result;
}

} // namespace misaki
