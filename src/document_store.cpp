#include "document_store.h"
#include "common/progress_bar.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdint.h>

DocumentStore::DocumentStore(const std::string &index_path)
{
    std::string doc_store_path = index_path + "/documents.dat";
    std::string doc_offset_path = index_path + "/doc_offsets.dat";
    std::string doc_names_path = index_path + "/doc_names.dat";

    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "LOADING DOCUMENT STORE" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    load_documents(doc_store_path, doc_offset_path);

    // Load document names with progress
    load_document_names(doc_names_path);

    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Document store loaded successfully!" << std::endl;
    std::cout << "  Total documents: " << documents_.size() << std::endl;
    std::cout << "  Total mappings: " << doc_name_to_id_.size() << std::endl;
    std::cout << std::string(70, '=') << std::endl
              << std::endl;
}

void DocumentStore::load_documents(const std::string &doc_store_path,
                                   const std::string &doc_offset_path)
{
    // First, load offset map to know how many documents to expect
    std::cout << "\nStep 1/2: Loading document offsets..." << std::endl;

    std::unordered_map<unsigned int, long long> offset_map;
    std::ifstream offset_file(doc_offset_path, std::ios::binary);

    if (!offset_file)
    {
        throw std::runtime_error("Cannot open document offset file: " + doc_offset_path);
    }

    unsigned int doc_id;
    long long offset;
    size_t offset_count = 0;

    Spinner offset_spinner("  Reading offsets");
    while (offset_file.read(reinterpret_cast<char *>(&doc_id), sizeof(doc_id)) &&
           offset_file.read(reinterpret_cast<char *>(&offset), sizeof(offset)))
    {
        offset_map[doc_id] = offset;
        offset_count++;

        if (offset_count % 10000 == 0)
        {
            offset_spinner.update();
        }
    }
    offset_spinner.finish("Loaded " + std::to_string(offset_count) + " offsets");
    offset_file.close();

    // Now load documents with progress bar
    std::cout << "\nStep 2/2: Loading document content..." << std::endl;

    std::ifstream doc_file(doc_store_path, std::ios::binary);
    if (!doc_file)
    {
        throw std::runtime_error("Cannot open document store file: " + doc_store_path);
    }

    documents_.reserve(offset_map.size());

    ProgressBar progress(offset_map.size(), "  Loading documents", 50);

    size_t loaded_count = 0;
    while (true)
    {
        unsigned int id;
        uint32_t content_length;

        if (!doc_file.read(reinterpret_cast<char *>(&id), sizeof(id)))
            break;
        if (!doc_file.read(reinterpret_cast<char *>(&content_length), sizeof(content_length)))
            break;

        std::string content(content_length, '\0');
        if (!doc_file.read(&content[0], content_length))
            break;

        documents_.emplace(id, Document(id, content));
        loaded_count++;

        if (loaded_count % 100 == 0 || loaded_count == offset_map.size())
        {
            progress.set_progress(loaded_count);
        }
    }

    progress.finish();
    doc_file.close();
}

void DocumentStore::load_document_names(const std::string &doc_names_path)
{
    std::cout << "\nStep 3/3: Loading document name mappings..." << std::endl;

    std::ifstream names_file(doc_names_path, std::ios::binary);

    if (!names_file)
    {
        std::cerr << "  Warning: Cannot open document names file: " << doc_names_path << std::endl;
        std::cerr << "  Document names will not be available for display." << std::endl;
        return;
    }

    Spinner names_spinner("  Reading document names");

    size_t name_count = 0;
    while (true)
    {
        unsigned int doc_id;
        uint32_t name_length;

        if (!names_file.read(reinterpret_cast<char *>(&doc_id), sizeof(doc_id)))
            break;
        if (!names_file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length)))
            break;

        std::string doc_name(name_length, '\0');
        if (!names_file.read(&doc_name[0], name_length))
            break;

        id_to_doc_name_[doc_id] = doc_name;
        doc_name_to_id_[doc_name] = doc_id;
        name_count++;

        if (name_count % 10000 == 0)
        {
            names_spinner.update();
        }
    }

    names_spinner.finish("Loaded " + std::to_string(name_count) + " document names");
    names_file.close();
}

const Document *DocumentStore::get_document(unsigned int doc_id) const
{
    auto it = documents_.find(doc_id);
    return (it != documents_.end()) ? &it->second : nullptr;
}

const std::string *DocumentStore::get_document_name(unsigned int doc_id) const
{
    auto it = id_to_doc_name_.find(doc_id);
    return (it != id_to_doc_name_.end()) ? &it->second : nullptr;
}