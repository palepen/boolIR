import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 12, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'legend.fontsize': 12, 'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def load_data(csv_file):
    """Load benchmark results from CSV"""
    if not Path(csv_file).exists():
        print(f"Error: Results file not found at {csv_file}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} benchmark results from {csv_file}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: Results file {csv_file} is empty.")
        return pd.DataFrame()

def plot_scalability(df, output_dir, metric, title, ylabel):
    """Generic function to plot scalability of a given metric."""
    if df.empty or metric not in df.columns or 'num_cpu_workers' not in df.columns:
        print(f"Skipping plot '{title}': Missing required columns ('{metric}', 'num_cpu_workers').")
        return

    fig, ax = plt.subplots()
    
    # Boolean scalability vs CPU workers
    boolean_df = df[df['use_reranking'] == False].sort_values('num_cpu_workers')
    if not boolean_df.empty:
        ax.plot(boolean_df['num_cpu_workers'], boolean_df[metric], 
                marker='s', markersize=8, linestyle='-', label='Boolean (vs. CPU Workers)')

    # Reranking scalability vs CPU workers (for a fixed number of GPU workers)
    rerank_df = df[df['use_reranking'] == True]
    if not rerank_df.empty:
        # Plot a line for each number of GPU workers tested
        for gpu_w in sorted(rerank_df['num_gpu_workers'].unique()):
            subset = rerank_df[rerank_df['num_gpu_workers'] == gpu_w].sort_values('num_cpu_workers')
            if not subset.empty:
                ax.plot(subset['num_cpu_workers'], subset[metric], 
                        marker='o', markersize=8, linestyle='--', label=f'Reranking (vs. CPU, {gpu_w} GPU worker(s))')

    ax.set_xlabel('Number of CPU Workers')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(df['num_cpu_workers'].unique()))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    filename = f"{output_dir}/scalability_{metric}.png"
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved scalability plot: {filename}")
    plt.close()

def plot_effectiveness_comparison(df, output_dir):
    """Plot a comparison of effectiveness metrics (P@10, MAP)."""
    if df.empty:
        print("Skipping effectiveness plot: DataFrame is empty.")
        return

    # Use a representative configuration (e.g., max CPU workers, median GPU workers)
    max_cpu = df['num_cpu_workers'].max()
    
    boolean_metrics = df[(df['use_reranking'] == False) & (df['num_cpu_workers'] == max_cpu)]
    rerank_metrics = df[df['use_reranking'] == True]

    if boolean_metrics.empty or rerank_metrics.empty:
        print("Skipping effectiveness plot: missing data for comparison.")
        return
        
    median_gpu = int(rerank_metrics['num_gpu_workers'].median())
    rerank_metrics_rep = rerank_metrics[(rerank_metrics['num_cpu_workers'] == max_cpu) & (rerank_metrics['num_gpu_workers'] == median_gpu)]

    if rerank_metrics_rep.empty:
        print(f"Skipping effectiveness plot: no reranking data for max_cpu={max_cpu} and median_gpu={median_gpu}.")
        return

    p10_vals = [boolean_metrics['precision_at_10'].mean(), rerank_metrics_rep['precision_at_10'].mean()]
    map_vals = [boolean_metrics['map'].mean(), rerank_metrics_rep['map'].mean()]
    
    labels = ['Boolean', 'Reranking']
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(x, p10_vals, width, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Score')
    ax1.set_title(f'P@10 ({max_cpu} CPU / {median_gpu} GPU workers)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(p10_vals) * 1.2)
    for i, v in enumerate(p10_vals):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

    ax2.bar(x, map_vals, width, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Score')
    ax2.set_title(f'MAP ({max_cpu} CPU / {median_gpu} GPU workers)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(map_vals) * 1.2)
    for i, v in enumerate(map_vals):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    filename = f"{output_dir}/effectiveness_comparison.png"
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved effectiveness plot: {filename}")
    plt.close()


def generate_summary_report(df, output_dir):
    """Generate a text summary report of the benchmark results."""
    if df.empty:
        return
        
    report_file = f"{output_dir}/benchmark_summary.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        boolean_df = df[df['use_reranking'] == False]
        rerank_df = df[df['use_reranking'] == True]

        if not boolean_df.empty:
            f.write("--- Boolean Retrieval Performance ---\n")
            best_throughput_bool = boolean_df.loc[boolean_df['throughput_qps'].idxmax()]
            f.write(f"  Best Throughput: {best_throughput_bool['throughput_qps']:.2f} q/s (at {int(best_throughput_bool['num_cpu_workers'])} CPU workers)\n")
            f.write(f"  Average P@10: {boolean_df['precision_at_10'].mean():.4f}\n")
            f.write(f"  Average MAP:  {boolean_df['map'].mean():.4f}\n\n")

        if not rerank_df.empty:
            f.write("--- Reranking Performance ---\n")
            best_throughput_rerank = rerank_df.loc[rerank_df['throughput_qps'].idxmax()]
            f.write(f"  Best Throughput: {best_throughput_rerank['throughput_qps']:.2f} q/s (at {int(best_throughput_rerank['num_cpu_workers'])} CPU / {int(best_throughput_rerank['num_gpu_workers'])} GPU workers)\n")
            f.write(f"  Average P@10: {rerank_df['precision_at_10'].mean():.4f}\n")
            f.write(f"  Average MAP:  {rerank_df['map'].mean():.4f}\n\n")

        if not boolean_df.empty and not rerank_df.empty:
             f.write("--- Comparison (Reranking vs. Boolean) ---\n")
             map_improvement = ((rerank_df['map'].mean() - boolean_df['map'].mean()) / boolean_df['map'].mean()) * 100
             f.write(f"  MAP Improvement with Reranking: {map_improvement:+.2f}%\n")
             p10_improvement = ((rerank_df['precision_at_10'].mean() - boolean_df['precision_at_10'].mean()) / boolean_df['precision_at_10'].mean()) * 100
             f.write(f"  P@10 Improvement with Reranking: {p10_improvement:+.2f}%\n")

    print(f"✓ Saved summary report: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate IR benchmark visualizations from a consolidated CSV file.')
    parser.add_argument('--results', required=True, help='Path to the consolidated results CSV file (e.g., all_benchmarks.csv)')
    parser.add_argument('--output-dir', default='results/plots', help='Directory to save plots and reports')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("BENCHMARK VISUALIZATION SUITE")
    print("=" * 80 + "\n")
    
    df = load_data(args.results)
    
    if not df.empty:
        plot_scalability(df, output_dir, 'throughput_qps', 'System Throughput vs. CPU Workers', 'Throughput (Queries/Second)')
        plot_scalability(df, output_dir, 'median_latency_ms', 'Median Query Latency vs. CPU Workers', 'Median Latency (ms)')
        plot_effectiveness_comparison(df, output_dir)
        generate_summary_report(df, output_dir)
        print("\n" + "=" * 80)
        print("✅ All visualizations completed!")
        print(f"📁 Results saved to: {output_dir}")
        print("=" * 80 + "\n")
    else:
        print("Could not generate plots due to missing or empty data file.")

if __name__ == "__main__":
    main()

