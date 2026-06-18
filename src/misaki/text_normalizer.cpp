#include "text_normalizer.h"
#include <regex>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace misaki {

static const char* const ONES[] = {
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen"
};
static const char* const TENS[] = {
    "","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"
};
static const char* const ORDINAL_SUFFIX[] = {
    "th","st","nd","rd","th","th","th","th","th","th"
};

std::string TextNormalizer::integer_to_words(long long n) const {
    if (n < 0)   return "negative " + integer_to_words(-n);
    if (n < 20)  return ONES[n];
    if (n < 100) {
        std::string r = TENS[n / 10];
        if (n % 10) r += " " + std::string(ONES[n % 10]);
        return r;
    }
    if (n < 1000) {
        std::string r = std::string(ONES[n / 100]) + " hundred";
        if (n % 100) r += " " + integer_to_words(n % 100);
        return r;
    }
    if (n < 1'000'000) {
        std::string r = integer_to_words(n / 1000) + " thousand";
        if (n % 1000) r += " " + integer_to_words(n % 1000);
        return r;
    }
    if (n < 1'000'000'000) {
        std::string r = integer_to_words(n / 1'000'000) + " million";
        if (n % 1'000'000) r += " " + integer_to_words(n % 1'000'000);
        return r;
    }
    std::string r = integer_to_words(n / 1'000'000'000) + " billion";
    if (n % 1'000'000'000) r += " " + integer_to_words(n % 1'000'000'000);
    return r;
}

std::string TextNormalizer::decimal_to_words(const std::string& int_part,
                                              const std::string& dec_part) const {
    std::string result = integer_to_words(std::stoll(int_part)) + " point";
    for (char c : dec_part) result += " " + std::string(ONES[c - '0']);
    return result;
}

std::string TextNormalizer::number_to_words(long long n) const {
    return integer_to_words(n);
}

std::string TextNormalizer::expand_currency(const std::string& text) const {
    // $1,234.56 → "one thousand two hundred thirty four dollars and fifty six cents"
    std::regex dollar_re(R"(\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?))", std::regex::ECMAScript);
    std::string result;
    std::sregex_iterator it(text.begin(), text.end(), dollar_re);
    std::sregex_iterator end;
    size_t last = 0;
    for (; it != end; ++it) {
        result += text.substr(last, it->position() - last);
        std::string raw = (*it)[1].str();
        // Remove commas
        std::string clean;
        for (char c : raw) if (c != ',') clean += c;
        auto dot = clean.find('.');
        std::string int_part = dot == std::string::npos ? clean : clean.substr(0, dot);
        std::string dec_part = dot == std::string::npos ? "" : clean.substr(dot + 1);
        long long dollars = std::stoll(int_part);
        result += integer_to_words(dollars) + (dollars == 1 ? " dollar" : " dollars");
        if (!dec_part.empty()) {
            while (dec_part.size() < 2) dec_part += "0";
            long long cents = std::stoll(dec_part.substr(0, 2));
            result += " and " + integer_to_words(cents) + (cents == 1 ? " cent" : " cents");
        }
        last = it->position() + it->length();
    }
    result += text.substr(last);
    return result;
}

std::string TextNormalizer::expand_numbers(const std::string& text) const {
    // Handles integers and decimals.
    // std::regex does not support lookbehind; use \b word boundaries instead.
    std::regex num_re(R"(\b(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\b)",
                      std::regex::ECMAScript);
    std::string result;
    std::sregex_iterator it(text.begin(), text.end(), num_re);
    std::sregex_iterator end;
    size_t last = 0;
    for (; it != end; ++it) {
        result += text.substr(last, it->position() - last);
        std::string raw = (*it)[1].str();
        std::string clean;
        for (char c : raw) if (c != ',') clean += c;
        auto dot = clean.find('.');
        if (dot == std::string::npos) {
            try { result += integer_to_words(std::stoll(clean)); }
            catch (...) { result += raw; }
        } else {
            result += decimal_to_words(clean.substr(0, dot), clean.substr(dot + 1));
        }
        last = it->position() + it->length();
    }
    result += text.substr(last);
    return result;
}

std::string TextNormalizer::expand_ordinals(const std::string& text) const {
    // 1st → first, 2nd → second, etc.
    static const std::unordered_map<std::string, std::string> ORDINALS = {
        {"1st","first"},{"2nd","second"},{"3rd","third"},{"4th","fourth"},
        {"5th","fifth"},{"6th","sixth"},{"7th","seventh"},{"8th","eighth"},
        {"9th","ninth"},{"10th","tenth"},{"11th","eleventh"},{"12th","twelfth"},
    };
    std::regex ord_re(R"(\b(\d+)(st|nd|rd|th)\b)", std::regex::icase);
    std::string result;
    std::sregex_iterator it(text.begin(), text.end(), ord_re);
    std::sregex_iterator end;
    size_t last = 0;
    for (; it != end; ++it) {
        result += text.substr(last, it->position() - last);
        std::string full = (*it)[0].str();
        auto oit = ORDINALS.find(full);
        if (oit != ORDINALS.end()) {
            result += oit->second;
        } else {
            try {
                long long n = std::stoll((*it)[1].str());
                result += integer_to_words(n) + "th";
            } catch (...) { result += full; }
        }
        last = it->position() + it->length();
    }
    result += text.substr(last);
    return result;
}

std::string TextNormalizer::expand_abbreviations(const std::string& text) const {
    static const std::vector<std::pair<std::regex, std::string>> ABBREVS = {
        {std::regex(R"(\bMr\.)"),   "mister"},
        {std::regex(R"(\bMrs\.)"),  "missus"},
        {std::regex(R"(\bMs\.)"),   "miss"},
        {std::regex(R"(\bDr\.)"),   "doctor"},
        {std::regex(R"(\bProf\.)"), "professor"},
        {std::regex(R"(\bSt\.)"),   "saint"},
        {std::regex(R"(\bvs\.)"),   "versus"},
        {std::regex(R"(\betc\.)"),  "et cetera"},
        {std::regex(R"(\bi\.e\.)"), "that is"},
        {std::regex(R"(\be\.g\.)"), "for example"},
    };
    std::string result = text;
    for (auto& [re, repl] : ABBREVS)
        result = std::regex_replace(result, re, repl);
    return result;
}

std::string TextNormalizer::normalize_whitespace(const std::string& text) const {
    std::string result;
    bool prev_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space && !result.empty()) result += ' ';
            prev_space = true;
        } else {
            result += c;
            prev_space = false;
        }
    }
    // Trim trailing space
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

std::string TextNormalizer::normalize(const std::string& text) const {
    std::string result = text;
    result = expand_abbreviations(result);
    result = expand_currency(result);
    result = expand_ordinals(result);
    result = expand_numbers(result);
    result = normalize_whitespace(result);
    return result;
}

} // namespace misaki
