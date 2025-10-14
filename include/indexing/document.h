#ifndef DOCUMENT_H
#define DOCUMENT_H
#include <string>
#include <vector>

// Represents a single document with an ID and content.
struct Document {
    u_int id;
    std::string content;
    Document(u_int id, std::string &content) : id(id), content(content) {}
    Document(u_int id, const std::string &content) : id(id), content(content) {}

    
};

// A collection of documents, used as the input for the indexers.
using DocumentCollection = std::vector<Document>;

#endif