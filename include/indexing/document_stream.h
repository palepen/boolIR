#ifndef DOCUMENT_STREAM_H
#define DOCUMENT_STREAM_H

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

/**
 * Lightweight metadata for a document stored on disk
 * Avoids loading content until needed
 */
struct DocumentMetadata {
    unsigned int id;
    std::string filepath;
    size_t file_size;
    
    DocumentMetadata(unsigned int id, const std::string& path, size_t size)
        : id(id), filepath(path), file_size(size) {}
};

/**
 * Memory-mapped file wrapper for efficient document reading
 * Handles platform-specific mmap operations
 */
class MemoryMappedFile {
public:
    MemoryMappedFile();
    ~MemoryMappedFile();
    
    // Disable copy, allow move
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    MemoryMappedFile(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept;
    
    /**
     * Map a file into memory for reading
     * @param filepath Path to the file
     * @return true if successful
     */
    bool open(const std::string& filepath);
    
    /**
     * Unmap the file
     */
    void close();
    
    /**
     * Get pointer to mapped data
     */
    const char* data() const { return data_; }
    
    /**
     * Get size of mapped file
     */
    size_t size() const { return size_; }
    
    /**
     * Check if file is currently mapped
     */
    bool is_open() const { return data_ != nullptr; }
    
    /**
     * Read entire content as string
     */
    std::string read_all() const;

private:
    const char* data_;
    size_t size_;
    int fd_;  // File descriptor (Unix)
    
    void cleanup();
};

/**
 * Manages streaming access to a corpus of documents
 * Loads documents on-demand using memory mapping
 */
class DocumentStream {
public:
    /**
     * Initialize stream from a corpus directory
     * Only loads metadata, not content
     */
    DocumentStream(const std::string& corpus_dir);
    
    /**
     * Get total number of documents
     */
    size_t size() const { return metadata_.size(); }
    
    /**
     * Check if a document ID exists
     */
    bool has_document(unsigned int doc_id) const {
        return doc_id < metadata_.size();
    }
    
    /**
     * Stream a document's content by ID
     * Reads from disk using memory mapping
     * @param doc_id Document ID
     * @return Document content as string
     */
    std::string read_document(unsigned int doc_id) const;
    
    /**
     * Get document metadata without reading content
     */
    const DocumentMetadata& get_metadata(unsigned int doc_id) const {
        return metadata_[doc_id];
    }
    
    /**
     * Get all metadata (for iteration)
     */
    const std::vector<DocumentMetadata>& get_all_metadata() const {
        return metadata_;
    }
    
    /**
     * Get document name mappings
     */
    const std::unordered_map<std::string, unsigned int>& get_name_to_id() const {
        return doc_name_to_id_;
    }
    
    const std::unordered_map<unsigned int, std::string>& get_id_to_name() const {
        return id_to_doc_name_;
    }

private:
    std::vector<DocumentMetadata> metadata_;
    std::unordered_map<std::string, unsigned int> doc_name_to_id_;
    std::unordered_map<unsigned int, std::string> id_to_doc_name_;
    std::string corpus_dir_;
    
    /**
     * Scan corpus directory and build metadata index
     */
    void build_metadata_index(const std::string& corpus_dir);
};

#endif // DOCUMENT_STREAM_H