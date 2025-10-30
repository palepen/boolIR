#include "retrieval/query_expander.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <stdexcept>

QueryExpander::QueryExpander(const std::string &synonym_file_path)
{
    load_synonyms(synonym_file_path);
}

void QueryExpander::load_synonyms(const std::string &synonym_file_path)
{
    std::ifstream file(synonym_file_path);
    if (!file.is_open())
    {
        std::cerr << "Warning: Could not open synonym file: " << synonym_file_path
                  << ". Query expansion will be disabled." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        std::stringstream ss(line);
        std::string term;
        std::string synonyms_str;

        if (std::getline(ss, term, ':') && std::getline(ss, synonyms_str))
        {
            // Trim and convert term to lowercase
            term.erase(0, term.find_first_not_of(" \t\n\r\f\v"));
            term.erase(term.find_last_not_of(" \t\n\r\f\v") + 1);
            std::transform(term.begin(), term.end(), term.begin(), ::tolower);

            std::vector<std::string> synonym_list;
            std::stringstream syn_ss(synonyms_str);
            std::string synonym;

            while (std::getline(syn_ss, synonym, ','))
            {
                synonym.erase(0, synonym.find_first_not_of(" \t\n\r\f\v"));
                synonym.erase(synonym.find_last_not_of(" \t\n\r\f\v") + 1);
                if (!synonym.empty())
                {
                    std::transform(synonym.begin(), synonym.end(), synonym.begin(), ::tolower);
                    synonym_list.push_back(synonym);
                }
            }
            if (!synonym_list.empty())
            {
                synonym_map_[term] = synonym_list;
            }
        }
    }
    std::cout << "Loaded " << synonym_map_.size() << " terms for query expansion from "
              << synonym_file_path << std::endl;
}

std::unique_ptr<QueryNode> QueryExpander::expand_query(const std::string &query_str)
{


    tokens_.clear();
    current_token_index_ = 0;
    std::stringstream ss(query_str);
    std::string token;
    while (ss >> token)
    {
        tokens_.push_back(token);
    }

    if (tokens_.empty())
    {
        return std::make_unique<QueryNode>(QueryOperator::AND); 
    }


    auto query_tree = parse_expression();

    if (!is_at_end())
    {
        std::cerr << "Warning: Could not parse entire query. Stopped at: " << peek() << std::endl;
    }

    return query_tree;
}

std::unique_ptr<QueryNode> QueryExpander::create_synonym_node(const std::string &term)
{
    auto or_node = std::make_unique<QueryNode>(QueryOperator::OR);

    std::set<std::string> term_variations;

    term_variations.insert(term);


    if (synonym_map_.count(term))
    {
        for (const auto &synonym : synonym_map_.at(term))
        {
            term_variations.insert(synonym);
        }
    }

    for (const auto &variation : term_variations)
    {
        or_node->children.push_back(std::make_unique<QueryNode>(variation));
    }

    if (or_node->children.size() == 1)
    {
        return std::move(or_node->children[0]);
    }
    else
    {
        return or_node;
    }
}

std::unique_ptr<QueryNode> QueryExpander::parse_expression()
{
    // Parses OR (lowest precedence)
    auto left_node = parse_term();

    while (!is_at_end() && peek() == "or")
    {
        consume(); // Consume "or"
        auto right_node = parse_term();

        // Create a new OR node and combine
        auto or_node = std::make_unique<QueryNode>(QueryOperator::OR);
        or_node->children.push_back(std::move(left_node));
        or_node->children.push_back(std::move(right_node));
        left_node = std::move(or_node);
    }

    return left_node;
}

std::unique_ptr<QueryNode> QueryExpander::parse_term()
{
    auto left_node = parse_factor();

    while (!is_at_end() && (peek() != "or" && peek() != ")"))
    {

        if (peek() == "and")
        {
            consume(); 
        }

        auto right_node = parse_factor();


        auto and_node = std::make_unique<QueryNode>(QueryOperator::AND);
        and_node->children.push_back(std::move(left_node));
        and_node->children.push_back(std::move(right_node));
        left_node = std::move(and_node);
    }

    return left_node;
}

std::unique_ptr<QueryNode> QueryExpander::parse_factor()
{
    std::string token = consume();

    if (token == "not")
    {
        auto child = parse_factor();
        auto not_node = std::make_unique<QueryNode>(QueryOperator::NOT);
        not_node->children.push_back(std::move(child));
        return not_node;
    }

    if (token == "(")
    {
        auto node = parse_expression();
        if (is_at_end() || consume() != ")")
        {
            throw std::runtime_error("Mismatched parentheses in query!");
        }
        return node;
    }

    // It's a regular term, create a synonym node for it
    return create_synonym_node(token);
}


bool QueryExpander::is_at_end() const
{
    return current_token_index_ >= tokens_.size();
}

std::string QueryExpander::peek() const
{
    if (is_at_end())
    {
        return ""; 
    }
    return tokens_[current_token_index_];
}

std::string QueryExpander::consume()
{
    if (is_at_end())
    {
        throw std::runtime_error("Unexpected end of query!");
    }
    return tokens_[current_token_index_++];
}