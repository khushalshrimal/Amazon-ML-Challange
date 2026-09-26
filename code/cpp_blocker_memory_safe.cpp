/*
 * 2-PASS Memory-Safe C++ Blocker & Candidate Generator for Amazon ML Challenge
 * Peak RAM < 400 MB (Fully 32-bit & 64-bit safe).
 * Implements Strategy 3 Multi-Key Union Blocker with Posting Limit = 100.
 * Exact match to Python normalize.py & blocking.py semantics.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

using namespace std;

// --- RAM MEASUREMENT HELPER ---
size_t get_peak_ram_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
#endif
    return 0;
}

// --- TIME FORMATTING HELPER ---
string format_time(double seconds) {
    if (seconds < 0 || seconds != seconds) return "00:00";
    int total_s = static_cast<int>(seconds);
    int m = (total_s / 60) % 60;
    int h = total_s / 3600;
    int s = total_s % 60;
    ostringstream oss;
    if (h > 0) {
        oss << setfill('0') << setw(2) << h << ":" << setw(2) << m << ":" << setw(2) << s;
    } else {
        oss << setfill('0') << setw(2) << m << ":" << setw(2) << s;
    }
    return oss.str();
}

// --- NORMALIZATION ---
string normalize_name(const string& src) {
    if (src.empty()) return "";
    string res;
    res.reserve(src.size() + 10);
    
    for (size_t i = 0; i < src.size(); ++i) {
        char c = src[i];
        if (c == '&') {
            res += " and ";
        } else if (isalnum(static_cast<unsigned char>(c))) {
            res += static_cast<char>(tolower(static_cast<unsigned char>(c)));
        } else {
            res += ' ';
        }
    }
    
    string clean;
    clean.reserve(res.size());
    bool in_space = false;
    for (char c : res) {
        if (c == ' ') {
            if (!in_space && !clean.empty()) {
                clean += ' ';
                in_space = true;
            }
        } else {
            clean += c;
            in_space = false;
        }
    }
    if (!clean.empty() && clean.back() == ' ') {
        clean.pop_back();
    }
    return clean;
}

string normalize_address(const string& src) {
    if (src.empty()) return "";
    string res;
    res.reserve(src.size());
    
    for (char c : src) {
        if (isalnum(static_cast<unsigned char>(c))) {
            res += static_cast<char>(tolower(static_cast<unsigned char>(c)));
        } else {
            res += ' ';
        }
    }
    
    string clean;
    clean.reserve(res.size());
    bool in_space = false;
    for (char c : res) {
        if (c == ' ') {
            if (!in_space && !clean.empty()) {
                clean += ' ';
                in_space = true;
            }
        } else {
            clean += c;
            in_space = false;
        }
    }
    if (!clean.empty() && clean.back() == ' ') {
        clean.pop_back();
    }
    return clean;
}

string normalize_country(const string& src) {
    if (src.empty()) return "UNKNOWN";
    string res;
    for (char c : src) {
        if (!isspace(static_cast<unsigned char>(c))) {
            res += static_cast<char>(toupper(static_cast<unsigned char>(c)));
        }
    }
    return res.empty() ? "UNKNOWN" : res;
}

vector<string> extract_tokens(const string& str) {
    vector<string> toks;
    if (str.empty()) return toks;
    stringstream ss(str);
    string tok;
    while (ss >> tok) {
        toks.push_back(tok);
    }
    return toks;
}

vector<string> extract_3grams(const string& str) {
    vector<string> ngs;
    if (str.empty()) return ngs;
    if (str.size() < 3) {
        ngs.push_back(str);
        return ngs;
    }
    for (size_t i = 0; i <= str.size() - 3; ++i) {
        ngs.push_back(str.substr(i, 3));
    }
    return ngs;
}

bool safe_getline(istream& is, string& t) {
    t.clear();
    char c;
    while (is.get(c)) {
        if (c == '\r') continue;
        if (c == '\n') return true;
        t += c;
    }
    return !t.empty();
}

int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    string base_dir_str = (argc > 1) ? argv[1] : ".";
    size_t max_records_limit = (argc > 2) ? stoull(argv[2]) : 0; // 0 = no limit

    string s2_path = base_dir_str + "/dataset/test/test_source2.tsv";
    string s3_path = base_dir_str + "/dataset/test/test_source3.tsv";
    string s1_path = base_dir_str + "/dataset/test/test_source1.tsv";
    string output_cand_path = base_dir_str + "/output/candidate_pairs.tsv";

    cout << "================================================================================" << endl;
    cout << "=== 2-PASS MEMORY-SAFE C++ HYBRID BLOCKER (POSTING LIMIT = 100) ===" << endl;
    cout << "================================================================================" << endl;

    auto t0_total = chrono::steady_clock::now();
    size_t total_expected_targets = (max_records_limit > 0) ? max_records_limit : 9969589;
    vector<string> target_files = {s2_path, s3_path};

    // --- PASS 1: FREQUENCY ANALYSIS FOR STOP WORDS ---
    cout << "\n[PASS 1/2] Streaming Target Pool to Compute Token Frequencies..." << endl;
    auto t0_p1 = chrono::steady_clock::now();
    auto last_log = chrono::steady_clock::now();
    
    unordered_map<string, int> name_token_freq;
    unordered_map<string, int> addr_token_freq;
    unordered_map<string, int> name_ngram_freq;
    
    size_t pass1_target_cnt = 0;
    
    for (const auto& fpath : target_files) {
        if (max_records_limit > 0 && pass1_target_cnt >= max_records_limit) break;
        ifstream infile(fpath, ios::in | ios::binary);
        if (!infile.is_open()) {
            cerr << "ERROR: Could not open file " << fpath << endl;
            return 1;
        }
        
        string line;
        safe_getline(infile, line); // header
        
        while (safe_getline(infile, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            string mid, name, addr, country;
            getline(ss, mid, '\t');
            getline(ss, name, '\t');
            getline(ss, addr, '\t');
            getline(ss, country, '\t');
            
            string norm_n = normalize_name(name);
            string norm_a = normalize_address(addr);
            
            vector<string> n_toks = extract_tokens(norm_n);
            vector<string> a_toks = extract_tokens(norm_a);
            vector<string> ngs = extract_3grams(norm_n);
            
            for (const auto& t : n_toks) name_token_freq[t]++;
            for (const auto& t : a_toks) addr_token_freq[t]++;
            for (const auto& ng : ngs) name_ngram_freq[ng]++;
            
            pass1_target_cnt++;
            
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - t0_p1).count();
            double since_last = chrono::duration<double>(now - last_log).count();
            
            if (since_last >= 2.0 || pass1_target_cnt == total_expected_targets) {
                last_log = now;
                double pct = (static_cast<double>(pass1_target_cnt) / total_expected_targets) * 100.0;
                double rate = elapsed > 0 ? (pass1_target_cnt / elapsed) : 0;
                double eta = rate > 0 ? ((total_expected_targets - pass1_target_cnt) / rate) : 0;
                
                cout << "[PASS 1] records=" << pass1_target_cnt << "/" << total_expected_targets
                     << " | " << fixed << setprecision(1) << pct << "%"
                     << " | rate=" << static_cast<long long>(rate) << " rec/s"
                     << " | elapsed=" << format_time(elapsed)
                     << " | ETA=" << format_time(eta)
                     << " | RAM=" << get_peak_ram_mb() << " MB" << endl << flush;
            }
            if (max_records_limit > 0 && pass1_target_cnt >= max_records_limit) break;
        }
    }
    
    double N = static_cast<double>(pass1_target_cnt);
    double max_n_freq = N * 0.03;
    double max_a_freq = N * 0.01;
    double max_ng_freq = N * 0.05;
    
    unordered_set<string> stop_name_tokens;
    unordered_set<string> stop_addr_tokens;
    unordered_set<string> stop_ngrams;
    
    for (const auto& kv : name_token_freq) if (kv.second > max_n_freq) stop_name_tokens.insert(kv.first);
    for (const auto& kv : addr_token_freq) if (kv.second > max_a_freq) stop_addr_tokens.insert(kv.first);
    for (const auto& kv : name_ngram_freq) if (kv.second > max_ng_freq) stop_ngrams.insert(kv.first);
    
    name_token_freq.clear();
    addr_token_freq.clear();
    name_ngram_freq.clear();
    
    cout << "\n[PASS 1 FINISHED] " << pass1_target_cnt << " records scanned in "
         << fixed << setprecision(2) << chrono::duration<double>(chrono::steady_clock::now() - t0_p1).count() << "s." << endl;
    cout << "Stop words computed: " << stop_name_tokens.size() << " name, "
         << stop_addr_tokens.size() << " addr, " << stop_ngrams.size() << " ngrams (RAM: " << get_peak_ram_mb() << " MB).\n" << endl;

    // --- PASS 2: STREAM & BUILD CAPPED INVERTED INDEX ---
    cout << "[PASS 2/2] Building Capped Inverted Index (Posting Cap = 100)..." << endl;
    auto t0_p2 = chrono::steady_clock::now();
    last_log = chrono::steady_clock::now();
    
    vector<string> target_mids;
    target_mids.reserve(pass1_target_cnt);
    
    unordered_map<string, vector<uint32_t>> blocker_index;
    blocker_index.reserve(2000000);
    
    const size_t MAX_LIMIT = 100;
    
    auto add_to_index_capped = [&](const string& key, uint32_t idx) {
        auto& vec = blocker_index[key];
        if (vec.size() < MAX_LIMIT) {
            vec.push_back(idx);
        }
    };
    
    uint32_t pass2_target_cnt = 0;
    for (const auto& fpath : target_files) {
        if (max_records_limit > 0 && pass2_target_cnt >= max_records_limit) break;
        ifstream infile(fpath, ios::in | ios::binary);
        if (!infile.is_open()) {
            cerr << "ERROR: Could not open file " << fpath << endl;
            return 1;
        }
        
        string line;
        safe_getline(infile, line); // header
        
        while (safe_getline(infile, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            string mid, name, addr, country;
            getline(ss, mid, '\t');
            getline(ss, name, '\t');
            getline(ss, addr, '\t');
            getline(ss, country, '\t');
            
            uint32_t idx = pass2_target_cnt;
            target_mids.push_back(mid);
            
            string norm_n = normalize_name(name);
            string norm_a = normalize_address(addr);
            string c = normalize_country(country);
            
            vector<string> n_toks = extract_tokens(norm_n);
            vector<string> a_toks = extract_tokens(norm_a);
            vector<string> ngs = extract_3grams(norm_n);
            
            // Channel 1: Name 2-pref
            if (n_toks.size() >= 2) {
                add_to_index_capped("N2:" + n_toks[0] + " " + n_toks[1] + "|" + c, idx);
            } else if (!n_toks.empty()) {
                add_to_index_capped("N2:" + n_toks[0] + "|" + c, idx);
            }
            
            // Channel 2: Addr 2-pref
            if (a_toks.size() >= 2) {
                add_to_index_capped("A2:" + a_toks[0] + " " + a_toks[1] + "|" + c, idx);
            } else if (!a_toks.empty()) {
                add_to_index_capped("A2:" + a_toks[0] + "|" + c, idx);
            }
            
            // Channel 3: Rare name tokens
            int rare_n_cnt = 0;
            for (const auto& tok : n_toks) {
                if (stop_name_tokens.find(tok) == stop_name_tokens.end() && tok.size() > 2) {
                    add_to_index_capped("NR:" + tok + "|" + c, idx);
                    if (++rare_n_cnt >= 2) break;
                }
            }
            
            // Channel 4: Rare addr tokens
            int rare_a_cnt = 0;
            for (const auto& tok : a_toks) {
                bool is_dig = !tok.empty() && all_of(tok.begin(), tok.end(), ::isdigit);
                if (stop_addr_tokens.find(tok) == stop_addr_tokens.end() && (tok.size() > 2 || is_dig)) {
                    add_to_index_capped("AR:" + tok + "|" + c, idx);
                    if (++rare_a_cnt >= 2) break;
                }
            }
            
            // Channel 5: Name 3-gram
            for (const auto& ng : ngs) {
                if (stop_ngrams.find(ng) == stop_ngrams.end()) {
                    add_to_index_capped("NG:" + ng + "|" + c, idx);
                    break;
                }
            }
            
            pass2_target_cnt++;
            
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - t0_p2).count();
            double since_last = chrono::duration<double>(now - last_log).count();
            
            if (since_last >= 2.0 || pass2_target_cnt == total_expected_targets) {
                last_log = now;
                double pct = (static_cast<double>(pass2_target_cnt) / total_expected_targets) * 100.0;
                double rate = elapsed > 0 ? (pass2_target_cnt / elapsed) : 0;
                double eta = rate > 0 ? ((total_expected_targets - pass2_target_cnt) / rate) : 0;
                
                cout << "[PASS 2] records=" << pass2_target_cnt << "/" << total_expected_targets
                     << " | " << fixed << setprecision(1) << pct << "%"
                     << " | rate=" << static_cast<long long>(rate) << " rec/s"
                     << " | elapsed=" << format_time(elapsed)
                     << " | ETA=" << format_time(eta)
                     << " | RAM=" << get_peak_ram_mb() << " MB" << endl << flush;
            }
            if (max_records_limit > 0 && pass2_target_cnt >= max_records_limit) break;
        }
    }

    // Prune entries that reached MAX_LIMIT exactly to match set semantics
    size_t pruned_cnt = 0;
    for (auto it = blocker_index.begin(); it != blocker_index.end(); ) {
        if (it->second.size() >= MAX_LIMIT) {
            it = blocker_index.erase(it);
            pruned_cnt++;
        } else {
            ++it;
        }
    }
    
    cout << "\n[PASS 2 FINISHED] Index built in "
         << fixed << setprecision(2) << chrono::duration<double>(chrono::steady_clock::now() - t0_p2).count() << "s." << endl;
    cout << "Pruned " << pruned_cnt << " keys reaching limit = 100 (Peak RAM: " << get_peak_ram_mb() << " MB).\n" << endl;

    // --- CANDIDATE GENERATION FOR S1 QUERIES ---
    cout << "[CANDIDATES] Generating Candidate Pairs for S1 Queries..." << endl;
    auto t0_cand = chrono::steady_clock::now();
    
    ifstream s1_file(s1_path, ios::in | ios::binary);
    if (!s1_file.is_open()) {
        cerr << "ERROR: Could not open file " << s1_path << endl;
        return 1;
    }
    
    ofstream cand_out(output_cand_path, ios::out | ios::binary);
    if (!cand_out.is_open()) {
        cerr << "ERROR: Could not create output file " << output_cand_path << endl;
        return 1;
    }
    
    cand_out << "source1_entity_id\tcandidate_entity_ids\n";
    
    string line;
    safe_getline(s1_file, line); // header
    
    size_t s1_count = 0;
    size_t total_cand_pairs = 0;
    size_t total_expected_s1 = (max_records_limit > 0) ? min(static_cast<size_t>(1000), static_cast<size_t>(1732544)) : 1732544;
    last_log = chrono::steady_clock::now();
    
    while (safe_getline(s1_file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string s1_id, r_n, r_a, c_str;
        getline(ss, s1_id, '\t');
        getline(ss, r_n, '\t');
        getline(ss, r_a, '\t');
        getline(ss, c_str, '\t');
        
        string norm_n = normalize_name(r_n);
        string norm_a = normalize_address(r_a);
        string c = normalize_country(c_str);
        
        vector<string> n_toks = extract_tokens(norm_n);
        vector<string> a_toks = extract_tokens(norm_a);
        vector<string> ngs = extract_3grams(norm_n);
        
        unordered_set<uint32_t> cand_indices;
        
        // Channel 1
        if (n_toks.size() >= 2) {
            auto it = blocker_index.find("N2:" + n_toks[0] + " " + n_toks[1] + "|" + c);
            if (it != blocker_index.end()) cand_indices.insert(it->second.begin(), it->second.end());
        } else if (!n_toks.empty()) {
            auto it = blocker_index.find("N2:" + n_toks[0] + "|" + c);
            if (it != blocker_index.end()) cand_indices.insert(it->second.begin(), it->second.end());
        }
        
        // Channel 2
        if (a_toks.size() >= 2) {
            auto it = blocker_index.find("A2:" + a_toks[0] + " " + a_toks[1] + "|" + c);
            if (it != blocker_index.end()) cand_indices.insert(it->second.begin(), it->second.end());
        } else if (!a_toks.empty()) {
            auto it = blocker_index.find("A2:" + a_toks[0] + "|" + c);
            if (it != blocker_index.end()) cand_indices.insert(it->second.begin(), it->second.end());
        }
        
        // Channel 3
        int rare_n_cnt = 0;
        for (const auto& tok : n_toks) {
            if (stop_name_tokens.find(tok) == stop_name_tokens.end() && tok.size() > 2) {
                auto it = blocker_index.find("NR:" + tok + "|" + c);
                if (it != blocker_index.end()) cand_indices.insert(it->second.begin(), it->second.end());
                if (++rare_n_cnt >= 2) break;
            }
        }
        
        // Channel 4
        int rare_a_cnt = 0;
        for (const auto& tok : a_toks) {
            bool is_dig = !tok.empty() && all_of(tok.begin(), tok.end(), ::isdigit);
            if (stop_addr_tokens.find(tok) == stop_addr_tokens.end() && (tok.size() > 2 || is_dig)) {
                auto it = blocker_index.find("AR:" + tok + "|" + c);
                if (it != blocker_index.end()) cand_indices.insert(it->second.begin(), it->second.end());
                if (++rare_a_cnt >= 2) break;
            }
        }
        
        // Channel 5
        for (const auto& ng : ngs) {
            if (stop_ngrams.find(ng) == stop_ngrams.end()) {
                auto it = blocker_index.find("NG:" + ng + "|" + c);
                if (it != blocker_index.end()) cand_indices.insert(it->second.begin(), it->second.end());
                break;
            }
        }
        
        total_cand_pairs += cand_indices.size();
        
        cand_out << s1_id << "\t";
        bool first = true;
        for (uint32_t idx : cand_indices) {
            if (!first) cand_out << ",";
            cand_out << target_mids[idx];
            first = false;
        }
        cand_out << "\n";
        
        s1_count++;
        
        auto now = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(now - t0_cand).count();
        double since_last = chrono::duration<double>(now - last_log).count();
        
        if (since_last >= 2.0 || s1_count == total_expected_s1) {
            last_log = now;
            double pct = (static_cast<double>(s1_count) / total_expected_s1) * 100.0;
            double rate = elapsed > 0 ? (s1_count / elapsed) : 0;
            double eta = rate > 0 ? ((total_expected_s1 - s1_count) / rate) : 0;
            
            cout << "[CANDIDATES] S1=" << s1_count << "/" << total_expected_s1
                 << " | " << fixed << setprecision(1) << pct << "%"
                 << " | candidates=" << total_cand_pairs
                 << " | rate=" << static_cast<long long>(rate) << " S1/s"
                 << " | elapsed=" << format_time(elapsed)
                 << " | ETA=" << format_time(eta)
                 << " | RAM=" << get_peak_ram_mb() << " MB" << endl << flush;
        }
        if (max_records_limit > 0 && s1_count >= max_records_limit) break;
    }
    
    cand_out.close();
    
    double t_total_sec = chrono::duration<double>(chrono::steady_clock::now() - t0_total).count();
    cout << "\n================================================================================" << endl;
    cout << "2-PASS C++ CANDIDATE GENERATION FINISHED IN " << fixed << setprecision(2) << t_total_sec << "s!" << endl;
    cout << "Total S1 Queries Processed: " << s1_count << endl;
    cout << "Total Candidate Pairs Generated: " << total_cand_pairs << endl;
    cout << "Peak RAM: " << get_peak_ram_mb() << " MB" << endl;
    cout << "Candidate file written to: " << output_cand_path << endl;
    cout << "================================================================================" << endl;
    
    return 0;
}
