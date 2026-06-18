#include "g2p.h"
#include "vocab.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <unistd.h>

#ifdef KOKORO_HAS_ESPEAK
#include <espeak-ng/speak_lib.h>
#endif

namespace misaki {

// Punctuation characters that map directly to token IDs
static const std::unordered_map<std::string, int64_t> PUNCT_TOKENS = {
    {",", 3}, {".", 4}, {"!", 5}, {"?", 6}, {";", 1}, {":", 2},
    {"\"", 11}, {"(", 12}, {")", 13}, {"/", 7},
    {"\xe2\x80\x94", 9},   // em dash —
    {"\xe2\x80\xa6", 10},  // ellipsis …
    {"\xe2\x80\x9c", 14},  // "
    {"\xe2\x80\x9d", 15},  // "
};

// IPA codepoints from espeak-ng that aren't directly in the VOCAB,
// mapped to approximate substitutes. ɚ (U+025A) IS token 85 now.
static const std::unordered_map<std::string, std::vector<int64_t>> IPA_SUBSTITUTIONS = {
    // ɝ (stressed rhotacized schwa, U+025D) → ɜ ː  (ɝ not in vocab)
    {"\xc9\x9d", {87, 158}},
    // ɫ (velarized l, U+026B) → l
    {"\xc9\xab", {54}},
    // ʷ (U+02B7) → w
    {"\xca\xb7", {65}},
    // ASCII g (U+0067) espeak occasionally emits → ɡ (IPA U+0261, token 92)
    {"g", {92}},
    // ɘ (U+0258) close-mid central unrounded → ə
    {"\xc9\x98", {83}},
    // ɵ (U+0275) close-mid central rounded → ə
    {"\xc9\xb5", {83}},
};

G2P::G2P() : G2P(Config{}) {}

G2P::G2P(const Config& cfg) : cfg_(cfg) {
    if (!cfg_.cmudict_path.empty())
        cmudict_.load(cfg_.cmudict_path);
    init_espeak();
}

G2P::~G2P() {
#ifdef KOKORO_HAS_ESPEAK
    if (espeak_ok_) espeak_Terminate();
#endif
}

void G2P::init_espeak() {
#ifdef KOKORO_HAS_ESPEAK
    int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr,
                                  espeakINITIALIZE_DONT_EXIT);
    if (rate < 0) { espeak_ok_ = false; return; }
    espeak_SetVoiceByName(cfg_.espeak_voice.c_str());
    espeak_ok_ = true;
#endif
}

bool G2P::cmudict_loaded() const { return cmudict_.loaded(); }
bool G2P::espeak_available() const { return espeak_ok_; }

// ── IPA character → token IDs ──────────────────────────────────────────────

std::vector<int64_t> G2P::ipa_char_to_tokens(const std::string& cp) const {
    // Direct VOCAB lookup first
    const auto& V = vocab();
    auto it = V.find(cp);
    if (it != V.end()) return {it->second};

    // Substitution table for chars espeak emits that aren't in VOCAB
    auto sit = IPA_SUBSTITUTIONS.find(cp);
    if (sit != IPA_SUBSTITUTIONS.end()) return sit->second;

    return {}; // Unknown — silently skip
}

std::vector<int64_t> G2P::ipa_to_tokens(const std::string& ipa) const {
    std::vector<int64_t> result;
    size_t pos = 0;
    while (pos < ipa.size()) {
        unsigned char c = static_cast<unsigned char>(ipa[pos]);
        int len;
        if      (c < 0x80) len = 1;
        else if (c < 0xE0) len = 2;
        else if (c < 0xF0) len = 3;
        else                len = 4;
        if (pos + len > ipa.size()) break;
        std::string cp = ipa.substr(pos, len);
        pos += len;
        auto toks = ipa_char_to_tokens(cp);
        result.insert(result.end(), toks.begin(), toks.end());
    }
    return result;
}

// ── espeak-ng IPA fallback ─────────────────────────────────────────────────

std::string G2P::espeak_ipa(const std::string& word) const {
    if (!espeak_ok_) return {};

#ifdef KOKORO_HAS_ESPEAK
    // Capture IPA output to a temp file via espeak_SetPhonemeTrace
    char outname[] = "/tmp/kokoro_ipa_XXXXXX";
    int ofd = mkstemp(outname);
    if (ofd < 0) return {};
    close(ofd);

    FILE* out = fopen(outname, "w");
    if (!out) { unlink(outname); return {}; }
    espeak_SetPhonemeTrace(espeakPHONEMES_IPA, out);

    std::string synth_text = word + "\n";
    espeak_Synth(synth_text.c_str(), synth_text.size(),
                 0, POS_CHARACTER, 0,
                 espeakCHARS_UTF8 | espeakPHONEMES,
                 nullptr, nullptr);
    espeak_Synchronize();
    fclose(out);
    espeak_SetPhonemeTrace(0, nullptr);

    FILE* in = fopen(outname, "r");
    std::string result;
    if (in) {
        char buf[1024];
        while (fgets(buf, sizeof(buf), in)) {
            std::string line(buf);
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
                line.pop_back();
            size_t start = line.find_first_not_of(' ');
            if (start != std::string::npos) result += line.substr(start);
        }
        fclose(in);
    }

    unlink(outname);
    return result;
#else
    return {};
#endif
}

// ── Word → token IDs ──────────────────────────────────────────────────────

static std::string strip_punct(const std::string& word) {
    std::string r;
    for (unsigned char c : word)
        if (std::isalpha(c) || c == '\'') r += std::tolower(c);
    return r;
}

std::vector<int64_t> G2P::word_to_tokens(const std::string& word) const {
    std::string clean = strip_punct(word);
    if (clean.empty()) return {};

    // Primary: CMU dict
    if (cmudict_.loaded()) {
        auto toks = cmudict_.lookup(clean);
        if (!toks.empty()) return toks;
    }

    // Fallback: espeak-ng IPA
    std::string ipa = espeak_ipa(clean);
    if (!ipa.empty()) return ipa_to_tokens(ipa);

    return {}; // OOV and no espeak — skip silently
}

// ── Main pipeline ─────────────────────────────────────────────────────────

std::vector<int64_t> G2P::text_to_tokens(const std::string& text) const {
    std::string normalized = cfg_.normalize_text ? normalizer_.normalize(text) : text;

    std::vector<int64_t> all_tokens;
    all_tokens.push_back(0); // leading pad

    // Tokenize into words and punctuation using a simple state machine.
    // We iterate over UTF-8 characters; ASCII punctuation is handled inline.
    size_t pos = 0;
    bool need_space = false;

    while (pos < normalized.size()) {
        unsigned char c = static_cast<unsigned char>(normalized[pos]);

        // Detect multi-byte UTF-8 (likely IPA or Unicode punct)
        if (c >= 0x80) {
            int len = (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            if (pos + len > normalized.size()) break;
            std::string cp = normalized.substr(pos, len);
            pos += len;

            // Check if this is a Unicode punctuation token
            auto pit = PUNCT_TOKENS.find(cp);
            if (pit != PUNCT_TOKENS.end()) {
                if (!all_tokens.empty() && all_tokens.back() == 16 /*space*/)
                    all_tokens.pop_back();
                all_tokens.push_back(pit->second);
                need_space = false;
            }
            // Otherwise treat as part of word (shouldn't happen after normalization)
            continue;
        }

        // Space / whitespace
        if (std::isspace(c)) {
            if (!all_tokens.empty() && all_tokens.back() != 0 && all_tokens.back() != 16)
                need_space = true;
            pos++;
            continue;
        }

        // ASCII punctuation
        {
            std::string cp(1, static_cast<char>(c));
            auto pit = PUNCT_TOKENS.find(cp);
            if (pit != PUNCT_TOKENS.end()) {
                if (!all_tokens.empty() && all_tokens.back() == 16)
                    all_tokens.pop_back();
                all_tokens.push_back(pit->second);
                need_space = false;
                pos++;
                continue;
            }
        }

        // Word: collect until next whitespace or ASCII punctuation
        size_t word_start = pos;
        while (pos < normalized.size()) {
            unsigned char wc = static_cast<unsigned char>(normalized[pos]);
            if (wc >= 0x80) { int l = (wc<0xE0)?2:(wc<0xF0)?3:4; pos += l; continue; }
            if (std::isspace(wc) || PUNCT_TOKENS.count(std::string(1, static_cast<char>(wc)))) break;
            pos++;
        }
        std::string word = normalized.substr(word_start, pos - word_start);

        auto toks = word_to_tokens(word);
        if (!toks.empty()) {
            if (need_space && !all_tokens.empty() && all_tokens.back() != 0)
                all_tokens.push_back(16); // space token
            all_tokens.insert(all_tokens.end(), toks.begin(), toks.end());
            need_space = false;
        }
    }

    // Cap at 512 tokens (including padding)
    if (all_tokens.size() >= 512) all_tokens.resize(511);
    all_tokens.push_back(0); // trailing pad
    return all_tokens;
}

} // namespace misaki
