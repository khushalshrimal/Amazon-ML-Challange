#include <iostream>
#include <string>
#include <cctype>

inline std::string normalize_name(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '&') {
            out += " and ";
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            out += c;
        } else if (c >= 'A' && c <= 'Z') {
            out += (char)(c + 32);
        } else {
            out += ' ';
        }
    }
    std::string clean;
    clean.reserve(out.size());
    bool in_space = false;
    for (char c : out) {
        if (c == ' ') {
            if (!in_space) {
                clean += ' ';
                in_space = true;
            }
        } else {
            clean += c;
            in_space = false;
        }
    }
    size_t start = 0;
    while (start < clean.size() && clean[start] == ' ') start++;
    size_t end = clean.size();
    while (end > start && clean[end - 1] == ' ') end--;
    return clean.substr(start, end - start);
}

int main() {
    std::string s1 = "Zephay Labs & Inc.";
    std::cout << "Original: " << s1 << "\nNormalized: '" << normalize_name(s1) << "'" << std::endl;
    return 0;
}
