#include "document_store.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdint.h>

DocumentStore::DocumentStore(const std::string& index_path) {
    std::string doc_store_path = index_path + "/documents.dat";
    std::string doc_offset_path = index_path + "/doc_offsets.dat";
    std::string doc_names_path = index_path + "/doc_names.dat";
    
    std::cout << "Loading document store from: " << doc_store_path << std::endl;
    load_documents(doc_store_path, doc_offset_path);
    std::cout << "  Loaded " << documents_.size() << " documents into memory" << std::endl;
    
    std::cout << "Loading document names from: " << doc_names_path << std::endl;
    load_document_names(doc_names_path);
    std::cout << "  Loaded " << id_to_doc_name_.size() << " document names" << std::endl;
    std::cout << "  Built reverse mapping: " << doc_name_to_id_.size() << " entries" << std::endl;
}

void DocumentStore::load_documents(const std::string& doc_store_path, 
                                   const std::string& doc_offset_path) {
    // Load offset map
    std::unordered_map<unsigned int, long long> offset_map;
    std::ifstream offset_file(doc_offset_path, std::ios::binary);
    
    if (!offset_file) {
        throw std::runtime_error("Cannot open document offset file: " + doc_offset_path);
    }
    
    unsigned int doc_id;
    long long offset;
    while (offset_file.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id)) &&
           offset_file.read(reinterpret_cast<char*>(&offset), sizeof(offset))) {
        offset_map[doc_id] = offset;
    }
    offset_file.close();
    
    // Load documents
    std::ifstream doc_file(doc_store_path, std::ios::binary);
    if (!doc_file) {
        throw std::runtime_error("Cannot open document store file: " + doc_store_path);
    }
    
    documents_.reserve(offset_map.size());
    
    while (true) {
        unsigned int id;
        uint32_t content_length;
        
        if (!doc_file.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
        if (!doc_file.read(reinterpret_cast<char*>(&content_length), sizeof(content_length))) break;
        
        std::string content(content_length, '\0');
        if (!doc_file.read(&content[0], content_length)) break;
        
        documents_.emplace(id, Document(id, content));
    }
    
    doc_file.close();
}

void DocumentStore::load_document_names(const std::string& doc_names_path) {
    std::ifstream names_file(doc_names_path, std::ios::binary);
    
    if (!names_file) {
        std::cerr << "Warning: Cannot open document names file: " << doc_names_path << std::endl;
        std::cerr << "  Document names will not be available for display." << std::endl;
        return;
    }
    
    while (true) {
        unsigned int doc_id;
        uint32_t name_length;
        
        // Read document ID
        if (!names_file.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id))) break;
        
        // Read name length
        if (!names_file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length))) break;
        
        // Read name string
        std::string doc_name(name_length, '\0');
        if (!names_file.read(&doc_name[0], name_length)) break;
        
        // FIXED: Build BOTH mappings
        id_to_doc_name_[doc_id] = doc_name;
        doc_name_to_id_[doc_name] = doc_id;  // <-- THIS WAS MISSING!
    }
    
    names_file.close();
}

const Document* DocumentStore::get_document(unsigned int doc_id) const {
    auto it = documents_.find(doc_id);
    return (it != documents_.end()) ? &it->second : nullptr;
}

const std::string* DocumentStore::get_document_name(unsigned int doc_id) const {
    auto it = id_to_doc_name_.find(doc_id);
    return (it != id_to_doc_name_.end()) ? &it->second : nullptr;
}