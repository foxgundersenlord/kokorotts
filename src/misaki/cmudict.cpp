#include "cmudict.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace misaki {

// ── Token ID constants (from vocab.h) ─────────────────────────────────────
static constexpr int64_t T_STRESS_PRIMARY   = 156; // ˈ
static constexpr int64_t T_STRESS_SECONDARY = 157; // ˌ
static constexpr int64_t T_SCHWA            = 83;  // ə
static constexpr int64_t T_WEDGE            = 138; // ʌ
static constexpr int64_t T_OPEN_BACK        = 69;  // ɑ
static constexpr int64_t T_ASH              = 72;  // æ
static constexpr int64_t T_OPEN_O           = 76;  // ɔ
static constexpr int64_t T_NEAR_OPEN        = 70;  // ɐ
static constexpr int64_t T_SMALL_CAP_I      = 102; // ɪ
static constexpr int64_t T_CLOSE_FRONT      = 51;  // i
static constexpr int64_t T_NEAR_CLOSE_U     = 135; // ʊ
static constexpr int64_t T_CLOSE_BACK       = 63;  // u
static constexpr int64_t T_OPEN_E           = 86;  // ɛ
static constexpr int64_t T_OPEN_MID_CENTRAL = 87;  // ɜ
static constexpr int64_t T_A               = 43;  // a
static constexpr int64_t T_E               = 47;  // e
static constexpr int64_t T_O               = 57;  // o
static constexpr int64_t T_APPROX_R        = 123; // ɹ
static constexpr int64_t T_B               = 44;
static constexpr int64_t T_D               = 46;
static constexpr int64_t T_F               = 48;
static constexpr int64_t T_G               = 92;  // ɡ (IPA, NOT ASCII g)
static constexpr int64_t T_H               = 50;
static constexpr int64_t T_J               = 52;  // j (palatal approx)
static constexpr int64_t T_K               = 53;
static constexpr int64_t T_L               = 54;
static constexpr int64_t T_M               = 55;
static constexpr int64_t T_N               = 56;
static constexpr int64_t T_NG              = 112; // ŋ
static constexpr int64_t T_P               = 58;
static constexpr int64_t T_S               = 61;
static constexpr int64_t T_SH              = 131; // ʃ
static constexpr int64_t T_T               = 62;
static constexpr int64_t T_TH_VOICED       = 81;  // ð
static constexpr int64_t T_TH_VOICELESS    = 119; // θ
static constexpr int64_t T_V               = 64;
static constexpr int64_t T_W               = 65;
static constexpr int64_t T_Z               = 68;
static constexpr int64_t T_ZH              = 147; // ʒ
static constexpr int64_t T_AFFRICATE_DZ    = 82;  // ʤ
static constexpr int64_t T_AFFRICATE_TS    = 133; // ʧ

std::vector<int64_t> CmuDict::phone_to_tokens(const std::string& phone) {
    // Parse stress digit from end
    int stress = 0;
    std::string base = phone;
    if (!phone.empty() && std::isdigit(static_cast<unsigned char>(phone.back()))) {
        stress = phone.back() - '0';
        base = phone.substr(0, phone.size() - 1);
    }

    std::vector<int64_t> toks;
    // Prepend stress marker before the vowel token(s)
    if (stress == 1) toks.push_back(T_STRESS_PRIMARY);
    else if (stress == 2) toks.push_back(T_STRESS_SECONDARY);

    if      (base == "AA") toks.push_back(T_OPEN_BACK);
    else if (base == "AE") toks.push_back(T_ASH);
    else if (base == "AH") toks.push_back(stress == 0 ? T_SCHWA : T_WEDGE);
    else if (base == "AO") toks.push_back(T_OPEN_O);
    else if (base == "AW") { toks.push_back(T_A);  toks.push_back(T_NEAR_CLOSE_U); }
    else if (base == "AY") { toks.push_back(T_A);  toks.push_back(T_SMALL_CAP_I); }
    else if (base == "B")  toks.push_back(T_B);
    else if (base == "CH") toks.push_back(T_AFFRICATE_TS);
    else if (base == "D")  toks.push_back(T_D);
    else if (base == "DH") toks.push_back(T_TH_VOICED);
    else if (base == "EH") toks.push_back(T_OPEN_E);
    else if (base == "ER") {
        if (stress == 0) {
            // Unstressed rhotacized schwa ɚ (token 85) — in vocab directly
            toks.push_back(85);
        } else {
            // Stressed: approximate ɝ as ɜ + length mark (ɝ not in vocab)
            toks.push_back(T_OPEN_MID_CENTRAL); // ɜ
            toks.push_back(158);                 // ː length mark
        }
    }
    else if (base == "EY") { toks.push_back(T_E);  toks.push_back(T_SMALL_CAP_I); }
    else if (base == "F")  toks.push_back(T_F);
    else if (base == "G")  toks.push_back(T_G);
    else if (base == "HH") toks.push_back(T_H);
    else if (base == "IH") toks.push_back(T_SMALL_CAP_I);
    else if (base == "IY") toks.push_back(T_CLOSE_FRONT);
    else if (base == "JH") toks.push_back(T_AFFRICATE_DZ);
    else if (base == "K")  toks.push_back(T_K);
    else if (base == "L")  toks.push_back(T_L);
    else if (base == "M")  toks.push_back(T_M);
    else if (base == "N")  toks.push_back(T_N);
    else if (base == "NG") toks.push_back(T_NG);
    else if (base == "OW") { toks.push_back(T_O);  toks.push_back(T_NEAR_CLOSE_U); }
    else if (base == "OY") { toks.push_back(T_OPEN_O); toks.push_back(T_SMALL_CAP_I); }
    else if (base == "P")  toks.push_back(T_P);
    else if (base == "R")  toks.push_back(T_APPROX_R);
    else if (base == "S")  toks.push_back(T_S);
    else if (base == "SH") toks.push_back(T_SH);
    else if (base == "T")  toks.push_back(T_T);
    else if (base == "TH") toks.push_back(T_TH_VOICELESS);
    else if (base == "UH") toks.push_back(T_NEAR_CLOSE_U);
    else if (base == "UW") toks.push_back(T_CLOSE_BACK);
    else if (base == "V")  toks.push_back(T_V);
    else if (base == "W")  toks.push_back(T_W);
    else if (base == "Y")  toks.push_back(T_J);
    else if (base == "Z")  toks.push_back(T_Z);
    else if (base == "ZH") toks.push_back(T_ZH);
    // Unknown phone: emit nothing (shouldn't happen with well-formed CMU dict)

    return toks;
}

std::vector<int64_t> CmuDict::arpabet_to_tokens(const std::vector<std::string>& phones) {
    std::vector<int64_t> result;
    for (auto& ph : phones) {
        auto toks = phone_to_tokens(ph);
        result.insert(result.end(), toks.begin(), toks.end());
    }
    return result;
}

std::string CmuDict::to_upper(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(),
        [](unsigned char c){ return std::toupper(c); });
    return r;
}

bool CmuDict::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';') continue; // skip comments
        std::istringstream ss(line);
        std::string word;
        ss >> word;
        if (word.empty()) continue;

        // Strip alternate-pronunciation markers like WORD(2)
        auto paren = word.find('(');
        if (paren != std::string::npos) word = word.substr(0, paren);

        std::vector<std::string> phones;
        std::string ph;
        while (ss >> ph) phones.push_back(ph);
        if (phones.empty()) continue;

        std::string key = to_upper(word);
        // Keep only the first pronunciation per word
        if (dict_.find(key) == dict_.end())
            dict_[key] = arpabet_to_tokens(phones);
    }

    loaded_ = !dict_.empty();
    return loaded_;
}

bool CmuDict::contains(const std::string& word) const {
    return dict_.count(to_upper(word)) > 0;
}

std::vector<int64_t> CmuDict::lookup(const std::string& word) const {
    auto it = dict_.find(to_upper(word));
    if (it == dict_.end()) return {};
    return it->second;
}

} // namespace misaki
