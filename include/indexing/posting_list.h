#ifndef POSTING_LIST_H
#define POSTING_LIST_H
#include <vector>

class PostingList
{
private:
    std::vector<int> postings_;
public:
    void add_document(int doc_id);

    const std::vector<int>& get_postings() const;
};


#endif