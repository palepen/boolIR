#include "indexing/posting_list.h"
#include <algorithm>

void PostingList::add_document(u_int doc_id)
{
    auto it = std::lower_bound(postings_.begin(), postings_.end(), doc_id);

    if (it == postings_.end() || *it != doc_id)
    {

        postings_.insert(it, doc_id);
    }
}

const std::vector<u_int> &PostingList::get_postings() const
{
    return postings_;
}