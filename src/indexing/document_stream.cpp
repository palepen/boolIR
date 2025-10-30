#include "indexing/document_stream.h"
#include "retrieval/query_preprocessor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>

MemoryMappedFile::MemoryMappedFile()
    : data_(nullptr), size_(0), fd_(-1) {}

MemoryMappedFile::~MemoryMappedFile()
{
    cleanup();
}

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile &&other) noexcept
    : data_(other.data_), size_(other.size_), fd_(other.fd_)
{
    other.data_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
}

MemoryMappedFile &MemoryMappedFile::operator=(MemoryMappedFile &&other) noexcept
{
    if (this != &other)
    {
        cleanup();
        data_ = other.data_;
        size_ = other.size_;
        fd_ = other.fd_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

bool MemoryMappedFile::open(const std::string &filepath)
{
    cleanup();

    // Open file
    fd_ = ::open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1)
    {
        std::cerr << "Failed to open file: " << filepath
                  << " (errno: " << errno << ")" << std::endl;
        return false;
    }

    // Get file size
    struct stat sb;
    if (fstat(fd_, &sb) == -1)
    {
        std::cerr << "Failed to get file size: " << filepath << std::endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    size_ = sb.st_size;

    // Handle empty files
    if (size_ == 0)
    {
        data_ = nullptr;
        return true;
    }

    // Memory map the file
    void *mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED)
    {
        std::cerr << "Failed to mmap file: " << filepath
                  << " (errno: " << errno << ")" << std::endl;
        ::close(fd_);
        fd_ = -1;
        size_ = 0;
        return false;
    }

    data_ = static_cast<const char *>(mapped);

    madvise(const_cast<char *>(data_), size_, MADV_SEQUENTIAL);

    return true;
}

void MemoryMappedFile::close()
{
    cleanup();
}

void MemoryMappedFile::cleanup()
{
    if (data_ != nullptr && size_ > 0)
    {
        munmap(const_cast<char *>(data_), size_);
    }
    if (fd_ != -1)
    {
        ::close(fd_);
    }
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
}

std::string MemoryMappedFile::read_all() const
{
    if (!is_open() || size_ == 0)
    {
        return "";
    }
    return std::string(data_, size_);
}

DocumentStream::DocumentStream(const std::string &corpus_dir)
    : corpus_dir_(corpus_dir)
{
    std::cout << "Building document stream index from: " << corpus_dir << std::endl;
    build_metadata_index(corpus_dir);
}

void DocumentStream::build_metadata_index(const std::string &corpus_dir)
{
    if (!fs::exists(corpus_dir) || !fs::is_directory(corpus_dir))
    {
        throw std::runtime_error("Corpus directory does not exist: " + corpus_dir);
    }

    unsigned int doc_id = 0;
    size_t total_size = 0;

    QueryPreprocessor preprocessor;

    // Scan directory for .txt files
    std::vector<fs::path> file_paths;
    for (const auto &entry : fs::directory_iterator(corpus_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
        {
            file_paths.push_back(entry.path());
        }
    }

    std::sort(file_paths.begin(), file_paths.end());

    for (const auto &filepath : file_paths)
    {
        std::string doc_name = filepath.stem().string();
        size_t file_size = fs::file_size(filepath);

        if (file_size == 0)
        {
            continue;
        }


        metadata_.emplace_back(doc_id, filepath.string(), file_size);
        doc_name_to_id_[doc_name] = doc_id;
        id_to_doc_name_[doc_id] = doc_name;

        total_size += file_size;
        doc_id++;

        if (doc_id % 10000 == 0)
        {
            std::cout << "  Indexed " << doc_id << " documents..." << std::endl;
        }
    }

    std::cout << "Document stream index built:" << std::endl;
    std::cout << "  Total documents: " << metadata_.size() << std::endl;
    std::cout << "  Total corpus size: " << (total_size / (1024.0 * 1024.0))
              << " MB" << std::endl;
    std::cout << "  Average document size: " << (total_size / metadata_.size())
              << " bytes" << std::endl;
}

std::string DocumentStream::read_document(unsigned int doc_id) const
{
    if (doc_id >= metadata_.size())
    {
        throw std::out_of_range("Document ID out of range: " + std::to_string(doc_id));
    }

    const auto &meta = metadata_[doc_id];

    MemoryMappedFile mmap_file;
    if (!mmap_file.open(meta.filepath))
    {
        std::cerr << "Warning: Failed to read document " << doc_id
                  << " from " << meta.filepath << std::endl;
        return "";
    }

    std::string content = mmap_file.read_all();

    QueryPreprocessor preprocessor;
    content = preprocessor.preprocess(content);

    content.erase(0, content.find_first_not_of(" \t\n\r\f\v"));
    content.erase(content.find_last_not_of(" \t\n\r\f\v") + 1);

    return content;
}