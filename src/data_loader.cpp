#include "data_loader.h"
#include "tokenizer/porter_stemmer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <algorithm>

// Helper function to apply stemming to text
static std::string apply_stemming_to_text(const std::string& text) {
    std::stringstream ss(text);
    std::stringstream result;
    std::string word;
    bool first = true;
    
    while (ss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), 
            [](unsigned char c) { return !std::isalnum(c); }), word.end());
        
        if (!word.empty()) {
            if (!first) result << " ";
            result << PorterStemmer::stem(word);
            first = false;
        }
    }
    return result.str();
}

std::pair<DocumentCollection, DocNameToIdMap> load_trec_documents(const std::string &corpus_dir)
{
    DocumentCollection documents;
    DocNameToIdMap doc_name_to_id;
    unsigned int id_counter = 0;

    std::cout << "Loading TREC documents from: " << corpus_dir << std::endl;

    for (const auto &entry : fs::recursive_directory_iterator(corpus_dir))
    {
        if (entry.is_regular_file())
        {
            std::cout << "Processing file: " << entry.path().string() << std::endl;
            std::ifstream ifs(entry.path().string());
            if (!ifs)
            {
                std::cerr << "Warning: Could not open " << entry.path().string() << std::endl;
                continue;
            }

            std::string line;
            std::string current_content;
            std::string current_docno;
            bool in_doc = false;
            bool in_text = false;

            while (std::getline(ifs, line))
            {
                line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
                line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

                if (line == "<DOC>")
                {
                    in_doc = true;
                    current_content.clear();
                    current_docno.clear();
                }
                else if (line == "</DOC>")
                {
                    if (in_doc && !current_docno.empty() && !current_content.empty())
                    {
                        // Convert content to lowercase
                        std::transform(current_content.begin(), current_content.end(),
                                       current_content.begin(),
                                       [](unsigned char c)
                                       { return std::tolower(c); });
                        
                        // Apply stemming (optional - can be commented out for speed)
                        // Note: Stemming is already applied during indexing in bsbi_indexer.cpp
                        // Doing it here would be redundant, but keeps documents in stemmed form
                        // Uncomment the line below if you want documents stored in stemmed form:
                        current_content = apply_stemming_to_text(current_content);
                        
                        current_content.erase(0, current_content.find_first_not_of(" \t\n\r\f\v"));
                        current_content.erase(current_content.find_last_not_of(" \t\n\r\f\v") + 1);
                        
                        documents.push_back(Document{id_counter, current_content});
                        doc_name_to_id[current_docno] = id_counter++;
                    }
                    in_doc = false;
                    in_text = false;
                }
                else if (in_doc)
                {
                    if (line.find("<DOCNO>") == 0)
                    {
                        size_t start = 7;
                        size_t end = line.find("</DOCNO>");
                        if (end != std::string::npos)
                        {
                            current_docno = line.substr(start, end - start);
                            current_docno.erase(0, current_docno.find_first_not_of(" \t\n\r\f\v"));
                            current_docno.erase(current_docno.find_last_not_of(" \t\n\r\f\v") + 1);
                        }
                    }
                    else if (line == "<TEXT>")
                    {
                        in_text = true;
                    }
                    else if (line == "</TEXT>")
                    {
                        in_text = false;
                    }
                    else if (in_text && !line.empty())
                    {
                        current_content += line + " ";
                    }
                }
            }
            ifs.close();
        }
    }

    std::cout << "Loaded " << documents.size() << " documents" << std::endl;
    return {documents, doc_name_to_id};
}

Qrels load_trec_qrels(const std::string &qrels_path, const DocNameToIdMap &doc_name_to_id)
{
    Qrels qrels;
    std::ifstream ifs(qrels_path);
    if (!ifs)
    {
        std::cerr << "Error: Could not open qrels file: " << qrels_path << std::endl;
        return qrels;
    }

    std::string line;
    size_t relevant_count = 0;
    while (std::getline(ifs, line))
    {
        std::stringstream ss(line);
        std::string query_id, iter, doc_name;
        int rel;
        if (ss >> query_id >> iter >> doc_name >> rel)
        {
            if (rel > 0)
            {
                auto it = doc_name_to_id.find(doc_name);
                if (it != doc_name_to_id.end())
                {
                    qrels[query_id].insert(it->second);
                    relevant_count++;
                }
            }
        }
        else
        {
            std::cerr << "Warning: Skipping malformed qrels line: " << line << std::endl;
        }
    }
    ifs.close();

    std::cout << "Loaded qrels for " << qrels.size() << " queries (" << relevant_count << " relevant judgments)" << std::endl;
    return qrels;
}

std::unordered_map<std::string, std::string> load_trec_topics(const std::string &topics_path)
{
    std::unordered_map<std::string, std::string> topics;
    std::ifstream ifs(topics_path);
    if (!ifs)
    {
        std::cerr << "Error: Could not open topics file: " << topics_path << std::endl;
        return topics;
    }

    std::string line;
    std::string current_id;
    std::string current_title;
    bool in_top = false;

    while (std::getline(ifs, line))
    {
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (line == "<top>")
        {
            in_top = true;
            current_id.clear();
            current_title.clear();
        }
        else if (line == "</top>")
        {
            if (in_top && !current_id.empty() && !current_title.empty())
            {
                std::transform(current_title.begin(), current_title.end(),
                               current_title.begin(),
                               [](unsigned char c)
                               { return std::tolower(c); });
                current_title.erase(0, current_title.find_first_not_of(" \t\n\r\f\v"));
                current_title.erase(current_title.find_last_not_of(" \t\n\r\f\v") + 1);
                
                std::cout << "DEBUG: Loaded query ID='" << current_id << "' title='" << current_title << "'" << std::endl;
                topics[current_id] = current_title;
            }
            in_top = false;
        }
        else if (in_top)
        {
            if (line.find("<num>") != std::string::npos)
            {
                size_t num_start = line.find("<num>");
                size_t content_start = num_start + 5;
                
                std::string num_content = line.substr(content_start);
                
                size_t colon_pos = num_content.find(':');
                if (colon_pos != std::string::npos)
                {
                    num_content = num_content.substr(colon_pos + 1);
                }
                
                size_t close_tag = num_content.find("</num>");
                if (close_tag != std::string::npos)
                {
                    num_content = num_content.substr(0, close_tag);
                }
                
                num_content.erase(0, num_content.find_first_not_of(" \t\n\r\f\v"));
                num_content.erase(num_content.find_last_not_of(" \t\n\r\f\v") + 1);
                current_id = num_content;
            }
            else if (line.find("<title>") != std::string::npos)
            {
                size_t title_start = line.find("<title>");
                size_t content_start = title_start + 7;
                
                std::string title_content = line.substr(content_start);
                
                size_t close_tag = title_content.find("</title>");
                if (close_tag != std::string::npos)
                {
                    title_content = title_content.substr(0, close_tag);
                }
                
                title_content.erase(0, title_content.find_first_not_of(" \t\n\r\f\v"));
                title_content.erase(title_content.find_last_not_of(" \t\n\r\f\v") + 1);
                current_title = title_content;
            }
        }
    }
    ifs.close();

    std::cout << "Loaded " << topics.size() << " topics" << std::endl;
    return topics;
}