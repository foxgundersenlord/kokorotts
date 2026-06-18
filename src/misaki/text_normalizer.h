#pragma once
#include <string>

namespace misaki {

// Normalizes English text before phonemization:
//   - Expands cardinal/ordinal numbers to words
//   - Expands common currency expressions
//   - Expands common abbreviations
//   - Normalizes whitespace and punctuation
class TextNormalizer {
public:
    std::string normalize(const std::string& text) const;

private:
    std::string expand_currency(const std::string& text) const;
    std::string expand_numbers(const std::string& text) const;
    std::string expand_ordinals(const std::string& text) const;
    std::string expand_abbreviations(const std::string& text) const;
    std::string normalize_whitespace(const std::string& text) const;

    std::string number_to_words(long long n) const;
    std::string integer_to_words(long long n) const;
    std::string decimal_to_words(const std::string& integer_part,
                                  const std::string& decimal_part) const;
};

} // namespace misaki
