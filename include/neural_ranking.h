#ifndef NEURAL_RANKING_H
#define NEURAL_RANKING_H

#include "retrieval.h"

typedef struct NeuralRanker {
    void *bert_model;
    float *embeddings_cache;
    int cache_size;
} NeuralRanker;

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations
NeuralRanker* initialize_neural_ranker(void);
void free_neural_ranker(NeuralRanker *ranker);
void rerank_results_neural(NeuralRanker *ranker, ResultSet *results, const char *query);
double compute_semantic_similarity(NeuralRanker *ranker, const char *query, const char *document);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_RANKING_H