#ifndef POSTING_LIST_H
#define POSTING_LIST_H
#include <vector>
#include <sys/types.h>


class PostingList
{
private:
    std::vector<u_int> postings_;

public:
    void add_document(u_int doc_id);

    const std::vector<u_int> &get_postings() const;
};

#endif