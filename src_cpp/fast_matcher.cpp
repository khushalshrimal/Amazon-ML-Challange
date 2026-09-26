#define _WIN32_WINNT 0x0600
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <windows.h>
#include <psapi.h>

struct CompactTarget {
    uint32_t target_numeric_id;
    uint8_t source_type;        // 2 for S2, 3 for S3
    uint8_t norm_name_len;      // length of normalized name
    uint16_t country_id;        // 16-bit country ID
    uint32_t name_prefix4;      // packed 4-char prefix of normalized name

    uint32_t name_token_offset;
    uint16_t name_token_len;

    uint32_t addr_token_offset;
    uint16_t addr_token_len;

    uint32_t digit_token_offset;
    uint16_t digit_token_len;
};

struct S1Record {
    std::string s1_id;
    std::string raw_name;
    std::string raw_addr;
    std::string raw_country;
    std::string candidate_ids_str;
};

// Global Memory Store
static std::vector<CompactTarget> g_targets;
static std::vector<uint32_t> g_token_pool;
static std::unordered_map<uint64_t, uint32_t> g_target_id_map;
static std::unordered_map<std::string, uint32_t> g_token_map;
static std::unordered_map<std::string, uint16_t> g_country_map;

// Atomic stats
static std::atomic<uint64_t> g_s1_processed(0);
static std::atomic<uint64_t> g_cands_processed(0);
static std::atomic<uint64_t> g_matches_found(0);
static std::atomic<uint64_t> g_targets_loaded(0);
static std::atomic<bool> g_is_done(false);

static ULONGLONG SubtractTimes(const FILETIME& ftA, const FILETIME& ftB) {
    LARGE_INTEGER a, b;
    a.LowPart = ftA.dwLowDateTime; a.HighPart = ftA.dwHighDateTime;
    b.LowPart = ftB.dwLowDateTime; b.HighPart = ftB.dwHighDateTime;
    return a.QuadPart - b.QuadPart;
}

static double GetCpuUsagePct() {
    static FILETIME prevSysIdle{}, prevSysKernel{}, prevSysUser{};
    FILETIME sysIdle, sysKernel, sysUser;
    if (GetSystemTimes(&sysIdle, &sysKernel, &sysUser)) {
        ULONGLONG idle = SubtractTimes(sysIdle, prevSysIdle);
        ULONGLONG kernel = SubtractTimes(sysKernel, prevSysKernel);
        ULONGLONG user = SubtractTimes(sysUser, prevSysUser);
        ULONGLONG total = kernel + user;
        prevSysIdle = sysIdle; prevSysKernel = sysKernel; prevSysUser = sysUser;
        if (total > 0) {
            return (double)(total - idle) * 100.0 / (double)total;
        }
    }
    return 0.0;
}

static double GetRamUsageGb() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    }
    return 0.0;
}

inline uint32_t GetOrInsertTokenId(const std::string& token) {
    auto it = g_token_map.find(token);
    if (it != g_token_map.end()) return it->second;
    uint32_t new_id = (uint32_t)g_token_map.size();
    g_token_map[token] = new_id;
    return new_id;
}

inline uint32_t LookupTokenId(const std::string& token) {
    auto it = g_token_map.find(token);
    if (it != g_token_map.end()) return it->second;
    return UINT32_MAX;
}

inline uint16_t GetCountryId(const std::string& country) {
    if (country.empty()) return 0;
    auto it = g_country_map.find(country);
    if (it != g_country_map.end()) return it->second;
    uint16_t new_id = (uint16_t)(g_country_map.size() + 1);
    g_country_map[country] = new_id;
    return new_id;
}

inline void trim_line_cr(std::string& line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
    }
}

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

inline std::string normalize_address(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
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

inline std::string normalize_country(const std::string& input) {
    std::string out;
    for (char c : input) {
        if (c != ' ' && c != '\r' && c != '\n' && c != '\t') {
            out += (char)toupper(c);
        }
    }
    if (out.empty() || out == "UNKNOWN") return "";
    return out;
}

inline uint32_t encode_prefix4(const std::string& str) {
    uint32_t val = 0;
    size_t len = (std::min)((size_t)4, str.size());
    for (size_t i = 0; i < len; ++i) {
        val = (val << 8) | (uint8_t)str[i];
    }
    return val;
}

inline std::vector<uint32_t> extract_target_tokens(const std::string& text, bool only_digits = false) {
    std::vector<uint32_t> tokens;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && text[i] == ' ') i++;
        if (i >= text.size()) break;
        size_t start = i;
        while (i < text.size() && text[i] != ' ') i++;
        std::string token_str = text.substr(start, i - start);

        if (only_digits) {
            bool all_dig = true;
            for (char c : token_str) {
                if (c < '0' || c > '9') { all_dig = false; break; }
            }
            if (!all_dig) continue;
        }

        tokens.push_back(GetOrInsertTokenId(token_str));
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

inline std::vector<uint32_t> extract_s1_tokens(const std::string& text, bool only_digits = false) {
    std::vector<uint32_t> tokens;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && text[i] == ' ') i++;
        if (i >= text.size()) break;
        size_t start = i;
        while (i < text.size() && text[i] != ' ') i++;
        std::string token_str = text.substr(start, i - start);

        if (only_digits) {
            bool all_dig = true;
            for (char c : token_str) {
                if (c < '0' || c > '9') { all_dig = false; break; }
            }
            if (!all_dig) continue;
        }

        uint32_t tid = LookupTokenId(token_str);
        if (tid != UINT32_MAX) {
            tokens.push_back(tid);
        }
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

inline int set_intersection_size_slice(const std::vector<uint32_t>& v1, const uint32_t* v2, uint16_t v2_len) {
    int count = 0;
    size_t i = 0;
    uint16_t j = 0;
    while (i < v1.size() && j < v2_len) {
        if (v1[i] < v2[j]) i++;
        else if (v2[j] < v1[i]) j++;
        else { count++; i++; j++; }
    }
    return count;
}

inline uint64_t parse_candidate_key(const char* str, size_t len) {
    if (len < 4) return 0;
    uint8_t src = 0;
    if (str[0] == 'S' || str[0] == 's') {
        if (str[1] == '2') src = 2;
        else if (str[1] == '3') src = 3;
    }
    if (src == 0) return 0;

    uint64_t num = 0;
    size_t i = 3; // Skip "S2-" or "S3-"
    while (i < len && str[i] >= '0' && str[i] <= '9') {
        num = num * 10 + (str[i] - '0');
        i++;
    }
    return ((uint64_t)src << 32) | num;
}

inline void append_target_id(std::string& out, uint8_t source_type, uint32_t numeric_id) {
    out += "S";
    out += (char)('0' + source_type);
    out += "-";
    out += std::to_string(numeric_id);
}

// Progress Reporter Thread
DWORD WINAPI ProgressThreadFunc(LPVOID lpParam) {
    auto start_time = std::chrono::steady_clock::now();
    const uint64_t TOTAL_S1 = 1732544;

    while (!g_is_done.load()) {
        Sleep(5000);
        if (g_is_done.load()) break;

        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
        uint64_t processed = g_s1_processed.load();
        uint64_t cands = g_cands_processed.load();
        uint64_t matches = g_matches_found.load();
        uint64_t targets = g_targets_loaded.load();

        if (processed == 0) {
            std::cout << "\nStatus: ACTIVE (Loading target index... " << targets << " records)" << std::endl;
            std::cout << "RAM: " << std::fixed << std::setprecision(2) << GetRamUsageGb() << " GB" << std::endl;
            std::cout.flush();
            continue;
        }

        double pct = (processed * 100.0) / TOTAL_S1;
        double rate = (elapsed_sec > 0.1) ? (processed / elapsed_sec) : 0.0;
        double eta_sec = (rate > 1.0) ? ((TOTAL_S1 - processed) / rate) : 0.0;

        int eta_m = (int)(eta_sec / 60.0);
        int eta_s = (int)fmod(eta_sec, 60.0);

        std::cout << "\nS1 processed: " << processed << " / " << TOTAL_S1 << std::endl;
        std::cout << "Progress: " << std::fixed << std::setprecision(2) << pct << "%" << std::endl;
        std::cout << "Candidates processed: " << cands << std::endl;
        std::cout << "Matches: " << matches << std::endl;
        std::cout << "Rate: " << (uint64_t)rate << " S1/sec" << std::endl;
        std::cout << "ETA: " << eta_m << "m " << eta_s << "s" << std::endl;
        std::cout << "CPU: " << std::fixed << std::setprecision(1) << GetCpuUsagePct() << "%" << std::endl;
        std::cout << "RAM: " << std::fixed << std::setprecision(2) << GetRamUsageGb() << " GB" << std::endl;
        std::cout << "Status: ACTIVE" << std::endl;
        std::cout.flush();
    }
    return 0;
}

// Sub-batch matching payload for Win32 worker thread
struct WorkerTask {
    const std::vector<S1Record>* s1_records;
    size_t start_idx;
    size_t end_idx;
    std::string out_buffer;
    uint64_t local_cands;
    uint64_t local_matches;
};

DWORD WINAPI WorkerThreadFunc(LPVOID lpParam) {
    WorkerTask* task = (WorkerTask*)lpParam;
    const auto& records = *(task->s1_records);
    task->out_buffer.reserve((task->end_idx - task->start_idx) * 32);
    task->local_cands = 0;
    task->local_matches = 0;

    for (size_t i = task->start_idx; i < task->end_idx; ++i) {
        const auto& rec = records[i];
        if (rec.candidate_ids_str.empty()) {
            task->out_buffer += rec.s1_id;
            task->out_buffer += "\t\n";
            continue;
        }

        // Tokenize and normalize S1 record
        std::string norm_n1 = normalize_name(rec.raw_name);
        std::string norm_a1 = normalize_address(rec.raw_addr);
        std::string country_str = normalize_country(rec.raw_country);

        uint16_t c1_id = GetCountryId(country_str);
        uint32_t prefix4_n1 = encode_prefix4(norm_n1);
        uint8_t len_n1 = (uint8_t)(std::min)((size_t)255, norm_n1.length());

        std::vector<uint32_t> toks_n1 = extract_s1_tokens(norm_n1);
        std::vector<uint32_t> toks_a1 = extract_s1_tokens(norm_a1);
        std::vector<uint32_t> digits_a1 = extract_s1_tokens(norm_a1, true);

        // Parse candidates
        const char* str = rec.candidate_ids_str.c_str();
        size_t len = rec.candidate_ids_str.size();
        size_t pos = 0;
        int best_cand_idx = -1;
        double best_score = -1.0;
        double second_best_score = -1.0;

        while (pos < len) {
            size_t start = pos;
            while (pos < len && str[pos] != ',') pos++;
            size_t cand_len = pos - start;
            if (pos < len && str[pos] == ',') pos++;

            if (cand_len == 0) continue;
            task->local_cands++;

            uint64_t key = parse_candidate_key(str + start, cand_len);
            if (key == 0) continue;

            auto it = g_target_id_map.find(key);
            if (it == g_target_id_map.end()) continue;

            uint32_t tid = it->second;
            const auto& t = g_targets[tid];

            double score = 0.0;
            const uint32_t* t_n_tokens = &g_token_pool[t.name_token_offset];
            const uint32_t* t_a_tokens = &g_token_pool[t.addr_token_offset];
            const uint32_t* t_d_tokens = &g_token_pool[t.digit_token_offset];

            bool n_exact = (prefix4_n1 != 0 && prefix4_n1 == t.name_prefix4 && len_n1 == t.norm_name_len &&
                            toks_n1.size() == t.name_token_len &&
                            set_intersection_size_slice(toks_n1, t_n_tokens, t.name_token_len) == (int)toks_n1.size());

            if (n_exact && len_n1 >= 3) {
                score += 55.0;
            } else if (!toks_n1.empty() && t.name_token_len > 0) {
                int n_ov = set_intersection_size_slice(toks_n1, t_n_tokens, t.name_token_len);
                double n_jac = (double)n_ov / (toks_n1.size() + t.name_token_len - n_ov);
                score += n_jac * 45.0;
            }

            if (!toks_a1.empty() && t.addr_token_len > 0) {
                int a_ov = set_intersection_size_slice(toks_a1, t_a_tokens, t.addr_token_len);
                double a_jac = (double)a_ov / (toks_a1.size() + t.addr_token_len - a_ov);
                score += a_jac * 25.0;
            }

            if (!digits_a1.empty() && t.digit_token_len > 0) {
                int d_ov = set_intersection_size_slice(digits_a1, t_d_tokens, t.digit_token_len);
                double d_jac = (double)d_ov / (digits_a1.size() + t.digit_token_len - d_ov);
                score += d_jac * 15.0;
            }

            if (c1_id != 0 && t.country_id != 0 && c1_id == t.country_id) {
                score += 5.0;
            }

            if (len_n1 >= 4 && t.norm_name_len >= 4 && prefix4_n1 != 0 && prefix4_n1 == t.name_prefix4) {
                score += 5.0;
            }

            int len_diff_n = std::abs((int)len_n1 - (int)t.norm_name_len);
            if (len_diff_n > 10) {
                score -= std::min(15.0, (len_diff_n - 10) * 0.5);
            }

            if (score > best_score) {
                second_best_score = best_score;
                best_score = score;
                best_cand_idx = (int)tid;
            } else if (score > second_best_score) {
                second_best_score = score;
            }
        }

        // Decision rule
        bool accept = false;
        if (best_cand_idx != -1 && best_score >= 48.0) {
            if (best_score - second_best_score < 2.0 && best_score < 60.0) {
                const auto& top_target = g_targets[best_cand_idx];
                if (prefix4_n1 != 0 && prefix4_n1 == top_target.name_prefix4 && len_n1 == top_target.norm_name_len) {
                    accept = true;
                }
            } else {
                accept = true;
            }
        }

        if (accept) {
            task->out_buffer += rec.s1_id;
            task->out_buffer += "\t";
            append_target_id(task->out_buffer, g_targets[best_cand_idx].source_type, g_targets[best_cand_idx].target_numeric_id);
            task->out_buffer += "\n";
            task->local_matches++;
        } else {
            task->out_buffer += rec.s1_id;
            task->out_buffer += "\t\n";
        }
    }
    return 0;
}

void LoadTargetSourceFile(const std::string& filepath, uint8_t default_source_type) {
    std::ifstream infile(filepath, std::ios::in | std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "ERROR: Cannot open " << filepath << std::endl;
        return;
    }

    std::string line;
    if (std::getline(infile, line)) {
        // Skip header
    }

    while (std::getline(infile, line)) {
        trim_line_cr(line);
        if (line.empty()) continue;
        size_t p1 = line.find('\t');
        if (p1 == std::string::npos) continue;
        size_t p2 = line.find('\t', p1 + 1);
        if (p2 == std::string::npos) continue;
        size_t p3 = line.find('\t', p2 + 1);

        std::string entity_id = line.substr(0, p1);
        std::string name = line.substr(p1 + 1, p2 - p1 - 1);
        std::string addr = (p3 == std::string::npos) ? line.substr(p2 + 1) : line.substr(p2 + 1, p3 - p2 - 1);
        std::string country = (p3 == std::string::npos) ? "" : line.substr(p3 + 1);

        uint64_t key = parse_candidate_key(entity_id.c_str(), entity_id.size());
        if (key == 0) continue;

        std::string norm_n = normalize_name(name);
        std::string norm_a = normalize_address(addr);
        std::string country_norm = normalize_country(country);

        std::vector<uint32_t> n_toks = extract_target_tokens(norm_n);
        std::vector<uint32_t> a_toks = extract_target_tokens(norm_a);
        std::vector<uint32_t> d_toks = extract_target_tokens(norm_a, true);

        CompactTarget target;
        target.target_numeric_id = (uint32_t)(key & 0xFFFFFFFF);
        target.source_type = (uint8_t)(key >> 32);
        target.norm_name_len = (uint8_t)(std::min)((size_t)255, norm_n.length());
        target.country_id = GetCountryId(country_norm);
        target.name_prefix4 = encode_prefix4(norm_n);

        // Append tokens to flat pool
        target.name_token_offset = (uint32_t)g_token_pool.size();
        target.name_token_len = (uint16_t)n_toks.size();
        g_token_pool.insert(g_token_pool.end(), n_toks.begin(), n_toks.end());

        target.addr_token_offset = (uint32_t)g_token_pool.size();
        target.addr_token_len = (uint16_t)a_toks.size();
        g_token_pool.insert(g_token_pool.end(), a_toks.begin(), a_toks.end());

        target.digit_token_offset = (uint32_t)g_token_pool.size();
        target.digit_token_len = (uint16_t)d_toks.size();
        g_token_pool.insert(g_token_pool.end(), d_toks.begin(), d_toks.end());

        uint32_t tid = (uint32_t)g_targets.size();
        g_targets.push_back(target);
        g_target_id_map[key] = tid;
        g_targets_loaded.fetch_add(1, std::memory_order_relaxed);
    }
}

int main() {
    auto t0 = std::chrono::steady_clock::now();
    std::cout << "==========================================================" << std::endl;
    std::cout << "=== FAST DETERMINISTIC C++ ENTITY MATCHING PIPELINE ===" << std::endl;
    std::cout << "==========================================================" << std::endl;

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int num_threads = sysinfo.dwNumberOfProcessors;
    if (num_threads <= 0) num_threads = 4;
    std::cout << "Hardware Threads: " << num_threads << std::endl;

    // Reserve vector space
    g_targets.reserve(10000000);
    g_token_pool.reserve(80000000);
    g_target_id_map.reserve(10000000);

    // Launch progress thread
    HANDLE hProgress = CreateThread(NULL, 0, ProgressThreadFunc, NULL, 0, NULL);

    // Step 1: Load Target Data (S2 & S3)
    std::cout << "Loading Target Source-2 and Source-3 datasets..." << std::endl;
    LoadTargetSourceFile("dataset/test/test_source2.tsv", 2);
    LoadTargetSourceFile("dataset/test/test_source3.tsv", 3);
    std::cout << "Target dataset loaded: " << g_targets.size() << " records, "
              << g_token_pool.size() << " total tokens in pool, "
              << g_token_map.size() << " unique tokens." << std::endl;
    std::cout << "RAM after target load: " << std::fixed << std::setprecision(2) << GetRamUsageGb() << " GB" << std::endl;

    // Step 2: Open S1 input file and Candidate pairs file in binary mode
    std::ifstream s1_file("dataset/test/test_source1.tsv", std::ios::in | std::ios::binary);
    std::ifstream cand_file("output/candidate_pairs.tsv", std::ios::in | std::ios::binary);

    if (!s1_file.is_open() || !cand_file.is_open()) {
        std::cerr << "ERROR: Failed to open test_source1.tsv or candidate_pairs.tsv!" << std::endl;
        return 1;
    }

    std::ofstream out_file("output/matching_results.tsv", std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out_file.is_open()) {
        std::cerr << "ERROR: Failed to open output/matching_results.tsv for writing!" << std::endl;
        return 1;
    }

    // Write exact header
    out_file << "source1_entity_id\tmatched_entity_ids\n";

    std::string s1_line, cand_line;
    std::getline(s1_file, s1_line);   // Skip headers
    std::getline(cand_file, cand_line);

    const size_t CHUNK_SIZE = 40000;
    std::vector<S1Record> chunk_records;
    chunk_records.reserve(CHUNK_SIZE);

    while (true) {
        chunk_records.clear();
        while (chunk_records.size() < CHUNK_SIZE && std::getline(s1_file, s1_line)) {
            if (!std::getline(cand_file, cand_line)) break;
            trim_line_cr(s1_line);
            trim_line_cr(cand_line);
            if (s1_line.empty()) continue;

            size_t p1 = s1_line.find('\t');
            if (p1 == std::string::npos) continue;
            size_t p2 = s1_line.find('\t', p1 + 1);
            if (p2 == std::string::npos) continue;
            size_t p3 = s1_line.find('\t', p2 + 1);

            S1Record rec;
            rec.s1_id = s1_line.substr(0, p1);
            rec.raw_name = s1_line.substr(p1 + 1, p2 - p1 - 1);
            rec.raw_addr = (p3 == std::string::npos) ? s1_line.substr(p2 + 1) : s1_line.substr(p2 + 1, p3 - p2 - 1);
            rec.raw_country = (p3 == std::string::npos) ? "" : s1_line.substr(p3 + 1);

            size_t cp1 = cand_line.find('\t');
            if (cp1 != std::string::npos) {
                rec.candidate_ids_str = cand_line.substr(cp1 + 1);
            }

            chunk_records.push_back(std::move(rec));
        }

        if (chunk_records.empty()) break;

        // Process chunk with Win32 Worker Threads
        size_t n_records = chunk_records.size();
        size_t batch_size = (n_records + num_threads - 1) / num_threads;

        std::vector<WorkerTask> tasks(num_threads);
        std::vector<HANDLE> hThreads(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            tasks[t].s1_records = &chunk_records;
            tasks[t].start_idx = t * batch_size;
            tasks[t].end_idx = (std::min)((t + 1) * batch_size, n_records);
            hThreads[t] = CreateThread(NULL, 0, WorkerThreadFunc, &tasks[t], 0, NULL);
        }

        WaitForMultipleObjects(num_threads, hThreads.data(), TRUE, INFINITE);

        for (int t = 0; t < num_threads; ++t) {
            CloseHandle(hThreads[t]);
            out_file.write(tasks[t].out_buffer.data(), tasks[t].out_buffer.size());
            g_cands_processed.fetch_add(tasks[t].local_cands, std::memory_order_relaxed);
            g_matches_found.fetch_add(tasks[t].local_matches, std::memory_order_relaxed);
        }

        g_s1_processed.fetch_add(n_records, std::memory_order_relaxed);
    }

    out_file.close();
    g_is_done = true;
    WaitForSingleObject(hProgress, 1000);
    CloseHandle(hProgress);

    auto t1 = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "MATCHING COMPLETED IN " << std::fixed << std::setprecision(2) << total_sec << " SECONDS!" << std::endl;
    std::cout << "S1 processed: " << g_s1_processed.load() << std::endl;
    std::cout << "Candidates processed: " << g_cands_processed.load() << std::endl;
    std::cout << "Matches found: " << g_matches_found.load() << std::endl;
    std::cout << "Output written to output/matching_results.tsv" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
