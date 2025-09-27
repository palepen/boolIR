#ifndef UI_H
#define UI_H

// Forward declare the structs to avoid circular dependencies
struct InvertedIndex;
struct NeuralRanker;
struct ResultSet;

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations
void show_usage(const char *program_name);
void run_interactive_mode(InvertedIndex *index, NeuralRanker *ranker);
void run_baseline_mode(InvertedIndex *index, int argc, char *argv[]);
void run_neural_mode(InvertedIndex *index, NeuralRanker *ranker, int argc, char *argv[]);
void display_results(ResultSet *results);

#ifdef __cplusplus
}
#endif

#endif // UI_H