#ifndef WORDPIECE_TOKENIZER_H
#define WORDPIECE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

class WordPieceTokenizer
{
public:
    explicit WordPieceTokenizer(
        const std::string &vocab_file,
        int max_input_chars_per_word = 200,
        const std::string &unk_token = "[UNK]");

    std::vector<std::string> tokenize(const std::string &text) const;
    std::vector<int64_t> convert_tokens_to_ids(const std::vector<std::string> &tokens) const;

    void encode(
        const std::string &text,
        int64_t max_length,
        std::vector<int64_t> &input_ids,
        std::vector<int64_t> &attention_mask) const;

    // New method for cross-encoder
    void encode_pair(
        const std::string &query,
        const std::string &document,
        int64_t max_length,
        std::vector<int64_t> &input_ids,
        std::vector<int64_t> &attention_mask) const;

    int64_t get_cls_token_id() const { return cls_token_id_; }
    int64_t get_sep_token_id() const { return sep_token_id_; }
    int64_t get_pad_token_id() const { return pad_token_id_; }
    int64_t get_unk_token_id() const { return unk_token_id_; }
    size_t vocab_size() const { return vocab_.size(); }

private:
    std::unordered_map<std::string, int64_t> vocab_;
    int64_t cls_token_id_;
    int64_t sep_token_id_;
    int64_t pad_token_id_;
    int64_t unk_token_id_;
    int max_input_chars_per_word_;
    std::string unk_token_;

    void load_vocab(const std::string &vocab_file);
    std::vector<std::string> basic_tokenize(const std::string &text) const;
    std::vector<std::string> wordpiece_tokenize(const std::string &word) const;
    bool is_whitespace(char c) const;
    bool is_punctuation(char c) const;
    std::string clean_text(const std::string &text) const;
};

#endif 