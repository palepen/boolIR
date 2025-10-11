#include "tokenizer/wordpiece_tokenizer.h"
#include <iostream>
#include <algorithm>
#include <cctype>

WordPieceTokenizer::WordPieceTokenizer(
    const std::string& vocab_file,
    int max_input_chars_per_word,
    const std::string& unk_token
) : max_input_chars_per_word_(max_input_chars_per_word),
    unk_token_(unk_token) {
    
    load_vocab(vocab_file);
}

void WordPieceTokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open vocabulary file: " + vocab_file);
    }

    std::string token;
    int64_t idx = 0;
    
    while (std::getline(file, token)) {
        token.erase(token.find_last_not_of(" \n\r\t") + 1);
        if (!token.empty()) {
            vocab_[token] = idx++;
        }
    }
    file.close();

    cls_token_id_ = vocab_.count("[CLS]") ? vocab_["[CLS]"] : 101;
    sep_token_id_ = vocab_.count("[SEP]") ? vocab_["[SEP]"] : 102;
    pad_token_id_ = vocab_.count("[PAD]") ? vocab_["[PAD]"] : 0;
    unk_token_id_ = vocab_.count("[UNK]") ? vocab_["[UNK]"] : 100;

    std::cout << "Loaded vocabulary with " << vocab_.size() << " tokens" << std::endl;
}

// ... (keep all the private helper methods like is_whitespace, clean_text, etc. the same)
bool WordPieceTokenizer::is_whitespace(char c) const { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
bool WordPieceTokenizer::is_punctuation(char c) const { if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) { return true; } return false; }
std::string WordPieceTokenizer::clean_text(const std::string& text) const { std::string output; output.reserve(text.length()); for (unsigned char uc : text) { char c = static_cast<char>(uc); if (uc == 0) continue; if (std::iscntrl(uc)) { if (c == '\n' || c == '\r' || c == '\t') output += ' '; continue; } if (is_whitespace(c)) { output += ' '; } else { output += c; } } return output; }
std::vector<std::string> WordPieceTokenizer::basic_tokenize(const std::string& text) const { std::string cleaned = clean_text(text); std::vector<std::string> tokens; std::string current_token; for (size_t i = 0; i < cleaned.length(); ++i) { char c = cleaned[i]; c = std::tolower(static_cast<unsigned char>(c)); if (is_whitespace(c)) { if (!current_token.empty()) { tokens.push_back(current_token); current_token.clear(); } } else if (is_punctuation(c)) { if (!current_token.empty()) { tokens.push_back(current_token); current_token.clear(); } tokens.push_back(std::string(1, c)); } else { current_token += c; } } if (!current_token.empty()) { tokens.push_back(current_token); } return tokens; }
std::vector<std::string> WordPieceTokenizer::wordpiece_tokenize(const std::string& word) const { if (word.length() > static_cast<size_t>(max_input_chars_per_word_)) { return {unk_token_}; } std::vector<std::string> output_tokens; size_t start = 0; while (start < word.length()) { size_t end = word.length(); std::string cur_substr; bool found = false; while (start < end) { std::string substr = word.substr(start, end - start); if (start > 0) { substr = "##" + substr; } if (vocab_.find(substr) != vocab_.end()) { cur_substr = substr; found = true; break; } end--; } if (!found) { return {unk_token_}; } output_tokens.push_back(cur_substr); start = end; } return output_tokens; }


std::vector<std::string> WordPieceTokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> split_tokens;
    std::vector<std::string> basic_tokens = basic_tokenize(text);
    for (const auto& token : basic_tokens) {
        std::vector<std::string> sub_tokens = wordpiece_tokenize(token);
        split_tokens.insert(split_tokens.end(), sub_tokens.begin(), sub_tokens.end());
    }
    return split_tokens;
}

std::vector<int64_t> WordPieceTokenizer::convert_tokens_to_ids(
    const std::vector<std::string>& tokens
) const {
    std::vector<int64_t> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_token_id_);
        }
    }
    return ids;
}

void WordPieceTokenizer::encode(
    const std::string& text,
    int64_t max_length,
    std::vector<int64_t>& input_ids,
    std::vector<int64_t>& attention_mask
) const {
    std::vector<std::string> tokens = tokenize(text);
    std::vector<int64_t> token_ids = convert_tokens_to_ids(tokens);
    
    input_ids.assign(max_length, pad_token_id_);
    attention_mask.assign(max_length, 0);
    
    input_ids[0] = cls_token_id_;
    attention_mask[0] = 1;
    
    int64_t max_tokens = max_length - 2;
    int64_t num_tokens = std::min(static_cast<int64_t>(token_ids.size()), max_tokens);
    
    for (int64_t i = 0; i < num_tokens; ++i) {
        input_ids[i + 1] = token_ids[i];
        attention_mask[i + 1] = 1;
    }
    
    input_ids[num_tokens + 1] = sep_token_id_;
    attention_mask[num_tokens + 1] = 1;
}

void WordPieceTokenizer::encode_pair(
    const std::string& query,
    const std::string& document,
    int64_t max_length,
    std::vector<int64_t>& input_ids,
    std::vector<int64_t>& attention_mask
) const {
    // Tokenize query and document
    std::vector<std::string> query_tokens = tokenize(query);
    std::vector<std::string> doc_tokens = tokenize(document);

    // Reserve space for special tokens [CLS], [SEP], [SEP]
    int64_t max_content_tokens = max_length - 3;
    
    // Truncate document if necessary, preserving the query
    if (query_tokens.size() + doc_tokens.size() > max_content_tokens) {
        int64_t max_doc_tokens = max_content_tokens - query_tokens.size();
        if (max_doc_tokens > 0) {
            doc_tokens.resize(max_doc_tokens);
        } else {
            doc_tokens.clear(); // Query is too long, no space for doc
        }
    }

    // Convert to IDs
    std::vector<int64_t> query_ids = convert_tokens_to_ids(query_tokens);
    std::vector<int64_t> doc_ids = convert_tokens_to_ids(doc_tokens);

    // Build final sequence
    input_ids.clear();
    attention_mask.clear();
    input_ids.reserve(max_length);
    attention_mask.reserve(max_length);

    // [CLS] Query [SEP] Document [SEP]
    input_ids.push_back(cls_token_id_);
    input_ids.insert(input_ids.end(), query_ids.begin(), query_ids.end());
    input_ids.push_back(sep_token_id_);
    input_ids.insert(input_ids.end(), doc_ids.begin(), doc_ids.end());
    input_ids.push_back(sep_token_id_);

    // Create attention mask and pad
    attention_mask.assign(input_ids.size(), 1);
    while (input_ids.size() < static_cast<size_t>(max_length)) {
        input_ids.push_back(pad_token_id_);
        attention_mask.push_back(0);
    }
}