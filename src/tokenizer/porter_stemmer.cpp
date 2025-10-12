#include "tokenizer/porter_stemmer.h"
#include <libstemmer.h>
#include <algorithm>
#include <cctype>

// Thread-local stemmer instance to avoid repeated allocations
thread_local struct sb_stemmer* stemmer_instance = nullptr;

static struct sb_stemmer* get_stemmer() {
    if (!stemmer_instance) {
        stemmer_instance = sb_stemmer_new("english", NULL);
    }
    return stemmer_instance;
}

std::string PorterStemmer::stem(const std::string& word) {
    if (word.empty()) {
        return word;
    }
    
    // Convert to lowercase
    std::string lower_word = word;
    std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    
    // Get thread-local stemmer instance
    struct sb_stemmer* stemmer = get_stemmer();
    if (!stemmer) {
        return lower_word;
    }
    
    // Perform stemming
    const sb_symbol* stemmed = sb_stemmer_stem(
        stemmer,
        reinterpret_cast<const sb_symbol*>(lower_word.c_str()),
        lower_word.length()
    );
    
    int stemmed_len = sb_stemmer_length(stemmer);
    std::string result(reinterpret_cast<const char*>(stemmed), stemmed_len);
    
    return result;
}