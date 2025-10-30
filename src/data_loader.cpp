#include "data_loader.h"
#include "retrieval/query_preprocessor.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <algorithm>


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
    size_t missing_docs = 0;

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
                else
                {
                    missing_docs++;
                }
            }
        }
        else
        {
            std::cerr << "Warning: Skipping malformed qrels line: " << line << std::endl;
        }
    }
    ifs.close();

    std::cout << "Loaded qrels for " << qrels.size() << " queries" << std::endl;
    std::cout << "  Relevant judgments: " << relevant_count << std::endl;
    if (missing_docs > 0)
    {
        std::cout << "  Warning: " << missing_docs << " referenced documents not found in corpus" << std::endl;
    }

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

    // Create query preprocessor for consistent preprocessing
    QueryPreprocessor preprocessor;

    std::string line;
    std::string current_id;
    std::string current_title;
    bool in_top = false;

    while (std::getline(ifs, line))
    {
        // Trim line
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
                // Apply query preprocessing for consistency with document preprocessing
                std::string preprocessed_title = preprocessor.preprocess(current_title);

                if (!preprocessed_title.empty())
                {
                    topics[current_id] = preprocessed_title;
                }
                else
                {
                    std::cerr << "Warning: Query " << current_id
                              << " became empty after preprocessing" << std::endl;
                    // Fallback to basic lowercase if preprocessing removes everything
                    std::string fallback = current_title;
                    std::transform(fallback.begin(), fallback.end(), fallback.begin(),
                                   [](unsigned char c)
                                   { return std::tolower(c); });
                    topics[current_id] = fallback;
                }
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

                // Remove "Number:" prefix if present
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

                // Trim whitespace
                num_content.erase(0, num_content.find_first_not_of(" \t\n\r\f\v"));
                num_content.erase(num_content.find_last_not_of(" \t\n\r\f\v") + 1);
                current_id = num_content;
            }
            else if (line.find("<title>") != std::string::npos)
            {
                size_t title_start = line.find("<title>");
                size_t content_start = title_start + 7;

                std::string title_content = line.substr(content_start);

                // Remove closing tag if present
                size_t close_tag = title_content.find("</title>");
                if (close_tag != std::string::npos)
                {
                    title_content = title_content.substr(0, close_tag);
                }

                // Trim whitespace
                title_content.erase(0, title_content.find_first_not_of(" \t\n\r\f\v"));
                title_content.erase(title_content.find_last_not_of(" \t\n\r\f\v") + 1);
                current_title = title_content;
            }
        }
    }
    ifs.close();

    std::cout << "Loaded " << topics.size() << " topics (with preprocessing)" << std::endl;
    return topics;
}