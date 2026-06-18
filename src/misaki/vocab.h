#pragma once
#include <string>
#include <unordered_map>
#include <cstdint>

// Kokoro-82M v1.0 phoneme vocabulary — sourced directly from
// Kokoro-82M/config.json. Keys are UTF-8 encoded Unicode codepoints.
namespace misaki {

inline const std::unordered_map<std::string, int64_t>& vocab() {
    static const std::unordered_map<std::string, int64_t> V = {
        // Punctuation & symbols
        {";",        1},
        {":",        2},
        {",",        3},
        {".",        4},
        {"!",        5},
        {"?",        6},
        // 7, 8 unassigned
        {"\xe2\x80\x94", 9},   // — EM DASH U+2014
        {"\xe2\x80\xa6", 10},  // … ELLIPSIS U+2026
        {"\"",       11},
        {"(",        12},
        {")",        13},
        {"\xe2\x80\x9c", 14},  // " U+201C
        {"\xe2\x80\x9d", 15},  // " U+201D
        {" ",        16},
        {"\xcc\x83", 17},      // ̃ COMBINING TILDE U+0303

        // Affricates and ligatures
        {"\xca\xa3", 18},  // ʣ U+02A3
        {"\xca\xa5", 19},  // ʥ U+02A5
        {"\xca\xa6", 20},  // ʦ U+02A6
        {"\xca\xa8", 21},  // ʨ U+02A8
        {"\xe1\xb5\x9d", 22},  // ᵝ U+1D5D
        {"\xea\xad\xa7", 23},  // ꭧ U+AB67

        // Capital letter shortcuts (used in Misaki's multilingual output)
        {"A", 24},
        {"I", 25},
        {"O", 31},
        {"Q", 33},
        {"S", 35},
        {"T", 36},
        {"W", 39},
        {"Y", 41},
        {"\xe1\xb5\x8a", 42},  // ᵊ U+1D4A

        // Lower-case Latin (used directly as IPA in some positions)
        {"a", 43}, {"b", 44}, {"c", 45}, {"d", 46}, {"e", 47}, {"f", 48},
        {"h", 50}, {"i", 51}, {"j", 52}, {"k", 53}, {"l", 54}, {"m", 55},
        {"n", 56}, {"o", 57}, {"p", 58}, {"q", 59}, {"r", 60}, {"s", 61},
        {"t", 62}, {"u", 63}, {"v", 64}, {"w", 65}, {"x", 66}, {"y", 67},
        {"z", 68},

        // IPA vowels
        {"\xc9\x91", 69},  // ɑ open back unrounded       U+0251
        {"\xc9\x90", 70},  // ɐ near-open central         U+0250
        {"\xc9\x92", 71},  // ɒ open back rounded         U+0252
        {"\xc3\xa6", 72},  // æ near-open front           U+00E6
        // 73, 74 unassigned
        {"\xce\xb2", 75},  // β voiced bilabial fricative U+03B2
        {"\xc9\x94", 76},  // ɔ open-mid back rounded     U+0254
        {"\xc9\x95", 77},  // ɕ alveolo-palatal sibilant  U+0255
        {"\xc3\xa7", 78},  // ç voiceless palatal frict.  U+00E7
        // 79 unassigned
        {"\xc9\x96", 80},  // ɖ retroflex stop            U+0256
        {"\xc3\xb0", 81},  // ð voiced dental fricative   U+00F0
        {"\xca\xa4", 82},  // ʤ voiced palato-alv. aff.   U+02A4
        {"\xc9\x99", 83},  // ə schwa                     U+0259
        // 84 unassigned
        {"\xc9\x9a", 85},  // ɚ rhotacized schwa          U+025A
        {"\xc9\x9b", 86},  // ɛ open-mid front unrounded  U+025B
        {"\xc9\x9c", 87},  // ɜ open-mid central          U+025C
        // 88, 89 unassigned
        {"\xc9\x9f", 90},  // ɟ voiced palatal stop       U+025F
        // 91 unassigned
        {"\xc9\xa1", 92},  // ɡ voiced velar stop (IPA)   U+0261  NOTE: not ASCII 'g'
        // 93-98 unassigned
        {"\xc9\xa5", 99},  // ɥ voiced labio-palatal app. U+0265
        // 100 unassigned
        {"\xc9\xa8", 101}, // ɨ close central unrounded   U+0268
        {"\xc9\xaa", 102}, // ɪ near-close near-front     U+026A
        {"\xca\x9d", 103}, // ʝ voiced palatal fricative  U+029D
        // 104-109 unassigned
        {"\xc9\xaf", 110}, // ɯ close back unrounded      U+026F
        {"\xc9\xb0", 111}, // ɰ velar approximant         U+0270
        {"\xc5\x8b", 112}, // ŋ velar nasal               U+014B
        {"\xc9\xb3", 113}, // ɳ retroflex nasal           U+0273
        {"\xc9\xb2", 114}, // ɲ palatal nasal             U+0272
        {"\xc9\xb4", 115}, // ɴ uvular nasal              U+0274
        {"\xc3\xb8", 116}, // ø close-mid front rounded   U+00F8
        // 117 unassigned
        {"\xc9\xb8", 118}, // ɸ voiceless bilabial frict. U+0278
        {"\xce\xb8", 119}, // θ voiceless dental frict.   U+03B8
        {"\xc5\x93", 120}, // œ open-mid front rounded    U+0153
        // 121, 122 unassigned
        {"\xc9\xb9", 123}, // ɹ alveolar approximant      U+0279
        // 124 unassigned
        {"\xc9\xbe", 125}, // ɾ alveolar tap              U+027E
        {"\xc9\xbb", 126}, // ɻ retroflex approximant     U+027B
        // 127 unassigned
        {"\xca\x81", 128}, // ʁ voiced uvular fricative   U+0281
        {"\xc9\xbd", 129}, // ɽ retroflex flap            U+027D
        {"\xca\x82", 130}, // ʂ retroflex sibilant frict. U+0282
        {"\xca\x83", 131}, // ʃ voiceless post-alv. sibl. U+0283
        {"\xca\x88", 132}, // ʈ voiceless retroflex stop  U+0288
        {"\xca\xa7", 133}, // ʧ voiceless palato-alv. aff.U+02A7
        // 134 unassigned
        {"\xca\x8a", 135}, // ʊ near-close near-back      U+028A
        {"\xca\x8b", 136}, // ʋ labiodental approximant   U+028B
        // 137 unassigned
        {"\xca\x8c", 138}, // ʌ open-mid back unrounded   U+028C
        {"\xc9\xa3", 139}, // ɣ voiced velar fricative    U+0263
        {"\xc9\xa4", 140}, // ɤ close-mid back unrounded  U+0264
        // 141 unassigned
        {"\xcf\x87", 142}, // χ voiceless uvular frict.   U+03C7
        {"\xca\x8e", 143}, // ʎ palatal lateral app.      U+028E
        // 144-146 unassigned
        {"\xca\x92", 147}, // ʒ voiced post-alv. sibilant U+0292
        {"\xca\x94", 148}, // ʔ glottal stop              U+0294
        // 149-155 unassigned
        {"\xcb\x88", 156}, // ˈ PRIMARY STRESS             U+02C8
        {"\xcb\x8c", 157}, // ˌ SECONDARY STRESS           U+02CC
        {"\xcb\x90", 158}, // ː LENGTH MARK                U+02D0
        // 159-161 unassigned
        {"\xca\xb0", 162}, // ʰ aspirated                  U+02B0
        // 163 unassigned
        {"\xca\xb2", 164}, // ʲ palatalized                U+02B2
        // 165-168 unassigned

        // Tone/prosody arrows (primarily for Chinese tones)
        {"\xe2\x86\x93", 169},  // ↓ U+2193
        {"\xe2\x86\x92", 171},  // → U+2192
        {"\xe2\x86\x97", 172},  // ↗ U+2197
        {"\xe2\x86\x98", 173},  // ↘ U+2198
        // 174 unassigned
        // 175, 176 unassigned
        {"\xe1\xb5\xbb", 177}, // ᵻ U+1D7B
    };
    return V;
}

inline const std::unordered_map<int64_t, std::string>& vocab_reverse() {
    static std::unordered_map<int64_t, std::string> RV;
    if (RV.empty())
        for (auto& [k, v] : vocab()) RV[v] = k;
    return RV;
}

} // namespace misaki
