#ifndef PORTER_STEMMER_H
#define PORTER_STEMMER_H

#include <string>

/**
 * @class PorterStemmer
 * @brief Wrapper around the Porter2 stemming library.
 * 
 * This uses the Snowball C++ implementation of the Porter stemmer.
 * The library handles all stemming rules internally.
 */
class PorterStemmer {
public:
    /**
     * @brief Stems a single word to its root form.
     * @param word The word to stem.
     * @return The stemmed word.
     */
    static std::string stem(const std::string& word);
};

#endif // PORTER_STEMMER_H