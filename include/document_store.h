#ifndef DOCUMENT_STORE_H
#define DOCUMENT_STORE_H

#include "indexing/document.h"
#include "data_loader.h"  // For IdToDocNameMap
#include <string>
#include <unordered_map>
#include <memory>

class DocumentStore {
public:
    explicit DocumentStore(const std::string& index_path);
    
    const Document* get_document(unsigned int doc_id) const;
    
    const std::string* get_document_name(unsigned int doc_id) const;
    
    size_t size() const { return documents_.size(); }
    
    const std::unordered_map<unsigned int, Document>& get_all() const { 
        return documents_; 
    }

private:
    void load_documents(const std::string& doc_store_path, 
                       const std::string& doc_offset_path);
    
    void load_document_names(const std::string& doc_names_path);
    
    std::unordered_map<unsigned int, Document> documents_;
    
    IdToDocNameMap id_to_doc_name_;
};

#endif