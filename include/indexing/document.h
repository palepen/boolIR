#ifndef DOCUMENT_H
#define DOCUMENT_H
#include <string>
#include <vector>

// Represents a single document with an ID and content.
struct Document {
    int id;
    std::string content;
};

// A collection of documents, used as the input for the indexers.
using DocumentCollection = std::vector<Document>;

#endif