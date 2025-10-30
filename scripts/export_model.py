import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import sys

DEFAULT_CSV_PATH = "results/all_benchmarks.csv"
DEFAULT_PLOTS_DIR = "results/plots"
# --- 1. ADDED: Path to the new indexing results CSV ---
INDEXING_CSV_PATH = "results/indexing_benchmarks.csv"

# Set a professional style for the plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 12, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'legend.fontsize': 12, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'figure.dpi': 150
})

def load_data(csv_file):
    """Load and preprocess benchmark results from the CSV file."""
    if not Path(csv_file).exists():
        print(f"Error: Results file not found at {csv_file}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_file)
        # Handle potential duplicate runs by averaging results for the same configuration
        group_cols = ['label', 'num_cpu_workers', 'use_reranking']
        df = df.groupby(group_cols).mean().reset_index()
        
        # Infer 'use_reranking' from the label for easier filtering if not present
        if 'use_reranking' not in df.columns:
            df['use_reranking'] = df['label'].str.contains('_Rerank')
            
        print(f"Loaded {len(df)} benchmark results from {csv_file}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Available worker counts: {sorted(df['num_cpu_workers'].unique())}")
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: Results file {csv_file} is empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# --- 2. ADDED: New function to plot indexing scalability ---
def plot_indexing_scalability(output_dir):
    """Loads indexing data and plots scalability."""
    indexing_csv = Path(INDEXING_CSV_PATH)
    if not indexing_csv.exists():
        print(f"\nWarning: Indexing benchmark file not found at {INDEXING_CSV_PATH}")
        print("         Run 'make benchmark-indexing' to generate it.")
        return

    try:
        df = pd.read_csv(indexing_csv)
    except pd.errors.EmptyDataError:
        print(f"Warning: Indexing benchmark file {INDEXING_CSV_PATH} is empty.")
        return
    
    if df.empty or 'num_cpu_workers' not in df.columns:
        print("Skipping indexing plots: No data or 'num_cpu_workers' column missing.")
        return

    print(f"\nLoaded {len(df)} indexing benchmark results.")
    df = df.sort_values('num_cpu_workers')
    
    # Plot 1: Indexing Throughput
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['num_cpu_workers'], df['throughput_docs_per_sec'], 
            marker='D', markersize=8, linestyle='-', label='Indexing Throughput')
    ax.set_xlabel('Number of CPU Workers (from C++ loop)')
    ax.set_ylabel('Throughput (Documents/Second)')
    ax.set_title('Indexing Throughput vs. CPU Workers')
    ax.set_xticks(sorted(df['num_cpu_workers'].unique()))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    filename_throughput = output_dir / "scalability_indexing_throughput.png"
    plt.savefig(filename_throughput, dpi=300)
    print(f"✓ Saved indexing throughput plot: {filename_throughput}")
    plt.close()

    # Plot 2: Indexing Time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['num_cpu_workers'], df['indexing_time_ms'], 
            marker='D', markersize=8, linestyle='--', color='darkred', label='Indexing Time')
    ax.set_xlabel('Number of CPU Workers (from C++ loop)')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title('Indexing Time vs. CPU Workers')
    ax.set_xticks(sorted(df['num_cpu_workers'].unique()))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    filename_time = output_dir / "scalability_indexing_time.png"
    plt.savefig(filename_time, dpi=300)
    print(f"✓ Saved indexing time plot: {filename_time}")
    plt.close()

def plot_scalability(df, output_dir, metric, title, ylabel, log_scale=False):
    """Generic function to plot scalability of a given metric vs. CPU workers."""
    if df.empty or metric not in df.columns or 'num_cpu_workers' not in df.columns:
        print(f"Skipping plot '{title}': Missing required data columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Boolean Retrieval performance
    boolean_df = df[df['use_reranking'] == False].sort_values('num_cpu_workers')
    if not boolean_df.empty:
        ax.plot(boolean_df['num_cpu_workers'], boolean_df[metric], 
                marker='s', markersize=8, linestyle='-', label='Boolean Retrieval')

    # Plot Reranking (End-to-End) performance
    rerank_df = df[df['use_reranking'] == True].sort_values('num_cpu_workers')
    if not rerank_df.empty:
        ax.plot(rerank_df['num_cpu_workers'], rerank_df[metric], 
                marker='o', markersize=8, linestyle='--', label='End-to-End Reranking')

    ax.set_xlabel('Number of CPU Workers (CILK_NWORKERS)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    worker_ticks = sorted(df['num_cpu_workers'].unique())
    if worker_ticks:
        ax.set_xticks(worker_ticks)
        
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel(f"{ylabel} (Log Scale)")

    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    filename = output_dir / f"scalability_{metric}.png"
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved scalability plot: {filename}")
    plt.close()

def plot_effectiveness_comparison(df, output_dir):
    """Plot a side-by-side comparison of effectiveness metrics at max CPU workers."""
    if df.empty:
        print("Skipping effectiveness bar plot: DataFrame is empty.")
        return

    # Use a representative configuration (e.g., max CPU workers) for comparison
    max_cpu = df['num_cpu_workers'].max()
    if pd.isna(max_cpu):
        print("Skipping effectiveness bar plot: No CPU worker data found.")
        return
        
    boolean_metrics = df[(df['use_reranking'] == False) & (df['num_cpu_workers'] == max_cpu)]
    rerank_metrics = df[(df['use_reranking'] == True) & (df['num_cpu_workers'] == max_cpu)]

    if boolean_metrics.empty or rerank_metrics.empty:
        print(f"Skipping effectiveness bar plot: missing data for {max_cpu} workers.")
        return
        
    p10_vals = [boolean_metrics['precision_at_10'].mean(), rerank_metrics['precision_at_10'].mean()]
    map_vals = [boolean_metrics['map'].mean(), rerank_metrics['map'].mean()]
    ndcg10_vals = [boolean_metrics['ndcg_at_10'].mean(), rerank_metrics['ndcg_at_10'].mean()]
    
    labels = ['Boolean', 'Reranking']
    x = np.arange(len(labels))
    width = 0.4

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    def add_bar_labels(ax, values):
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # P@10 bar chart
    ax1.bar(x, p10_vals, width, color=['#3498db', '#e74c3c'], alpha=0.85, edgecolor='black')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Precision@10 (at {max_cpu} CPU workers)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(p10_vals) * 1.25 if max(p10_vals) > 0 else 1)
    add_bar_labels(ax1, p10_vals)

    # MAP bar chart
    ax2.bar(x, map_vals, width, color=['#3498db', '#e74c3c'], alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Score')
    ax2.set_title(f'Mean Average Precision (MAP) (at {max_cpu} CPU workers)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(map_vals) * 1.25 if max(map_vals) > 0 else 1)
    add_bar_labels(ax2, map_vals)
    
    # NDCG@10 bar chart
    ax3.bar(x, ndcg10_vals, width, color=['#3498db', '#e74c3c'], alpha=0.85, edgecolor='black')
    ax3.set_ylabel('Score')
    ax3.set_title(f'nDCG@10 (at {max_cpu} CPU workers)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, max(ndcg10_vals) * 1.25 if max(ndcg10_vals) > 0 else 1)
    add_bar_labels(ax3, ndcg10_vals)
        
    fig.suptitle('Effectiveness Comparison: Boolean vs. Reranking', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = output_dir / "effectiveness_comparison_bars.png"
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved effectiveness bar plot: {filename}")
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate a clean text summary report of the benchmark results."""
    if df.empty:
        print("Skipping summary report: DataFrame is empty.")
        return
        
    report_file = output_dir / "benchmark_summary.txt"
    
    boolean_df = df[df['use_reranking'] == False].sort_values('num_cpu_workers')
    rerank_df = df[df['use_reranking'] == True].sort_values('num_cpu_workers')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:,.2f}'.format)

    with open(report_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 120 + "\n\n")

        if not boolean_df.empty:
            f.write("--- [ TABLE 1: Pure Boolean Retrieval Performance ] ---\n\n")
            summary_cols = ['num_cpu_workers', 'throughput_qps', 'median_latency_ms', 'p95_latency_ms', 'precision_at_10', 'map', 'ndcg_at_10']
            f.write(boolean_df[summary_cols].to_string(index=False))
            f.write("\n\n")
        else:
            f.write("--- No data for Pure Boolean Retrieval ---\n\n")

        if not rerank_df.empty:
            f.write("--- [ TABLE 2: End-to-End Reranking Performance ] ---\n\n")
            summary_cols = ['num_cpu_workers', 'throughput_qps', 'median_latency_ms', 'p95_latency_ms', 'precision_at_10', 'map', 'ndcg_at_10']
            f.write(rerank_df[summary_cols].to_string(index=False))
            f.write("\n\n")
        else:
            f.write("--- No data for End-to-End Reranking ---\n\n")

        if not boolean_df.empty and not rerank_df.empty:
            f.write("--- [ TABLE 3: Reranking vs. Boolean (% Change) ] ---\n\n")
            
            # Align data on worker count
            merged_df = pd.merge(
                boolean_df, 
                rerank_df, 
                on='num_cpu_workers', 
                suffixes=('_bool', '_rerank'),
                how='inner'
            )
            
            if not merged_df.empty:
                comparison_df = pd.DataFrame()
                comparison_df['num_cpu_workers'] = merged_df['num_cpu_workers']
                
                # Calculate % change
                def percent_change(new, old):
                    return ((new - old) / old) * 100
                
                comparison_df['throughput_%_change'] = percent_change(merged_df['throughput_qps_rerank'], merged_df['throughput_qps_bool'])
                comparison_df['latency_%_change'] = percent_change(merged_df['median_latency_ms_rerank'], merged_df['median_latency_ms_bool'])
                comparison_df['p10_%_change'] = percent_change(merged_df['precision_at_10_rerank'], merged_df['precision_at_10_bool'])
                comparison_df['map_%_change'] = percent_change(merged_df['map_rerank'], merged_df['map_bool'])
                comparison_df['ndcg10_%_change'] = percent_change(merged_df['ndcg_at_10_rerank'], merged_df['ndcg_at_10_bool'])
                
                f.write(comparison_df.to_string(index=False))
                f.write("\n\n")
            else:
                f.write("--- No matching worker counts found for comparison ---\n\n")

        f.write("=" * 120 + "\n")
        f.write("Report generation complete.\n")
        f.write("=" * 120 + "\n")

    print(f"✓ Saved summary report: {report_file}")
    
    # Print the report to console as well
    with open(report_file, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description='Generate IR benchmark visualizations from a consolidated CSV file.')
    parser.add_argument('--results', required=False, default=DEFAULT_CSV_PATH,
                        help=f'Path to the consolidated results CSV file (default: {DEFAULT_CSV_PATH})')
    parser.add_argument('--output-dir', default=DEFAULT_PLOTS_DIR, 
                        help=f'Directory to save plots and reports (default: {DEFAULT_PLOTS_DIR})')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("BENCHMARK VISUALIZATION SUITE")
    print("=" * 80 + "\n")
    
    df = load_data(args.results)
    
    if not df.empty:
        print("\n--- Generating Query Performance Plots ---")
        # Generate the core scalability plots
        plot_scalability(df, output_dir, 'throughput_qps', 
                         'System Throughput vs. CPU Workers', 
                         'Throughput (Queries/Second)', log_scale=True)
                         
        plot_scalability(df, output_dir, 'median_latency_ms', 
                         'Median Query Latency vs. CPU Workers', 
                         'Median Latency (ms)', log_scale=True)
                         
        # Generate effectiveness scalability plots
        plot_scalability(df, output_dir, 'precision_at_10', 
                         'Precision@10 vs. CPU Workers', 
                         'Precision@10 Score')
                         
        plot_scalability(df, output_dir, 'map', 
                         'MAP vs. CPU Workers', 
                         'MAP Score')
                         
        plot_scalability(df, output_dir, 'ndcg_at_10', 
                         'nDCG@10 vs. CPU Workers', 
                         'nDCG@10 Score')
        
        # Generate the summary bar chart
        plot_effectiveness_comparison(df, output_dir)
        
        # Generate the detailed text summary
        generate_summary_report(df, output_dir)
    else:
        print("Could not generate query plots due to missing or empty data file.")
        # Do not exit, we might still have indexing data
    
    # --- 3. ADDED: Call the new indexing plot function ---
    print("\n--- Generating Indexing Performance Plots ---")
    plot_indexing_scalability(output_dir)
    # --- END ADDITION ---

    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")
    # else: # Removed this 'else' block
    #     print("Could not generate plots due to missing or empty data file.")
    #     sys.exit(1)

if __name__ == "__main__":
    main()

