#include "indexing/posting_list.h"
#include <algorithm>

void PostingList::add_document(u_int doc_id) {
    // A simple push_back is sufficient for this implementation.
    // In a real system, you might ensure IDs are sorted or unique.
    auto it = std::lower_bound(postings_.begin(), postings_.end(), doc_id);


    if (it == postings_.end() || *it != doc_id) {
        // Insert the doc_id at the correct position. This is an O(N) operation.
        postings_.insert(it, doc_id);
    }
}

const std::vector<u_int>& PostingList::get_postings() const {
    return postings_;
}