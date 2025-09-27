#ifndef INDEXING_H
#define INDEXING_H

#include <cstdint>
#include <string>
#include <unordered_map>

#ifdef __cplusplus
// In C++, the InvertedIndex struct is a wrapper around a C++ object.
struct InvertedIndex {
    std::unordered_map<std::string, struct PostingList*> postings;
};

extern "C" {
#else
// C-compatible declaration
typedef struct InvertedIndex InvertedIndex;
#endif

typedef struct PostingList {
    uint32_t doc_id;
    uint32_t term_freq;
    struct PostingList *next;
} PostingList;

// Function declarations
InvertedIndex* create_inverted_index(void);
void free_inverted_index(InvertedIndex *index);
void build_index_parallel(InvertedIndex *index, const char *dataset_path);
PostingList* get_posting_list(InvertedIndex *index, const char *term);

#ifdef __cplusplus
}
#endif

#endif // INDEXING_H