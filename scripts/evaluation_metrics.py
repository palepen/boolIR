import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.size': 12, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'legend.fontsize': 12, 'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def load_data(csv_file):
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
    if df.empty or metric not in df.columns or 'num_cpu_workers' not in df.columns:
        print(f"Skipping plot '{title}': Missing required columns.")
        return

    fig, ax = plt.subplots()
    
    # Monolithic Boolean
    boolean_mono_df = df[(df['use_reranking'] == False) & (df['use_partitioned'] == False)].sort_values('num_cpu_workers')
    if not boolean_mono_df.empty:
        ax.plot(boolean_mono_df['num_cpu_workers'], boolean_mono_df[metric], 
                marker='s', markersize=8, linestyle='-', label='Boolean (Monolithic)')

    # Partitioned Boolean
    boolean_part_df = df[(df['use_reranking'] == False) & (df['use_partitioned'] == True)].sort_values('num_cpu_workers')
    if not boolean_part_df.empty:
        ax.plot(boolean_part_df['num_cpu_workers'], boolean_part_df[metric], 
                marker='^', markersize=8, linestyle='--', label='Boolean (Partitioned)')

    # Monolithic Reranking
    rerank_mono_df = df[(df['use_reranking'] == True) & (df['use_partitioned'] == False)].sort_values('num_cpu_workers')
    if not rerank_mono_df.empty:
        ax.plot(rerank_mono_df['num_cpu_workers'], rerank_mono_df[metric], 
                marker='o', markersize=8, linestyle='-', label='Reranking (Monolithic)')

    # Partitioned Reranking
    rerank_part_df = df[(df['use_reranking'] == True) & (df['use_partitioned'] == True)].sort_values('num_cpu_workers')
    if not rerank_part_df.empty:
        ax.plot(rerank_part_df['num_cpu_workers'], rerank_part_df[metric], 
                marker='d', markersize=8, linestyle='--', label='Reranking (Partitioned)')

    ax.set_xlabel('Number of CPU Workers')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(df['num_cpu_workers'].unique()))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    filename = f"{output_dir}/scalability_{metric}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved scalability plot: {filename}")
    plt.close()

def plot_effectiveness_comparison(df, output_dir):
    if df.empty:
        print("Skipping effectiveness plot: DataFrame is empty.")
        return

    max_cpu = df['num_cpu_workers'].max()
    
    boolean_mono = df[(df['use_reranking'] == False) & (df['use_partitioned'] == False) & (df['num_cpu_workers'] == max_cpu)]
    boolean_part = df[(df['use_reranking'] == False) & (df['use_partitioned'] == True) & (df['num_cpu_workers'] == max_cpu)]
    rerank_mono = df[(df['use_reranking'] == True) & (df['use_partitioned'] == False) & (df['num_cpu_workers'] == max_cpu)]
    rerank_part = df[(df['use_reranking'] == True) & (df['use_partitioned'] == True) & (df['num_cpu_workers'] == max_cpu)]

    if boolean_mono.empty and boolean_part.empty and rerank_mono.empty and rerank_part.empty:
        print("Skipping effectiveness plot: no data for comparison.")
        return
    
    categories = []
    p10_vals = []
    map_vals = []
    
    if not boolean_mono.empty:
        categories.append('Boolean\n(Mono)')
        p10_vals.append(boolean_mono['precision_at_10'].mean())
        map_vals.append(boolean_mono['map'].mean())
    
    if not boolean_part.empty:
        categories.append('Boolean\n(Part)')
        p10_vals.append(boolean_part['precision_at_10'].mean())
        map_vals.append(boolean_part['map'].mean())
    
    if not rerank_mono.empty:
        categories.append('Rerank\n(Mono)')
        p10_vals.append(rerank_mono['precision_at_10'].mean())
        map_vals.append(rerank_mono['map'].mean())
    
    if not rerank_part.empty:
        categories.append('Rerank\n(Part)')
        p10_vals.append(rerank_part['precision_at_10'].mean())
        map_vals.append(rerank_part['map'].mean())
    
    x = np.arange(len(categories))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(x, p10_vals, width, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(categories)], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Score')
    ax1.set_title(f'P@10 Comparison ({max_cpu} CPU workers)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, max(p10_vals) * 1.2)
    for i, v in enumerate(p10_vals):
        ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

    ax2.bar(x, map_vals, width, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(categories)], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Score')
    ax2.set_title(f'MAP Comparison ({max_cpu} CPU workers)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, max(map_vals) * 1.2)
    for i, v in enumerate(map_vals):
        ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    filename = f"{output_dir}/effectiveness_comparison.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved effectiveness plot: {filename}")
    plt.close()

def plot_partitioned_vs_monolithic(df, output_dir):
    """Compare partitioned vs monolithic retrieval performance"""
    if df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Throughput comparison
    boolean_mono = df[(df['use_reranking'] == False) & (df['use_partitioned'] == False)].sort_values('num_cpu_workers')
    boolean_part = df[(df['use_reranking'] == False) & (df['use_partitioned'] == True)].sort_values('num_cpu_workers')
    
    if not boolean_mono.empty:
        ax1.plot(boolean_mono['num_cpu_workers'], boolean_mono['throughput_qps'], 
                marker='s', markersize=8, linestyle='-', label='Monolithic')
    if not boolean_part.empty:
        ax1.plot(boolean_part['num_cpu_workers'], boolean_part['throughput_qps'], 
                marker='^', markersize=8, linestyle='--', label='Partitioned')
    
    ax1.set_xlabel('Number of CPU Workers')
    ax1.set_ylabel('Throughput (Queries/Second)')
    ax1.set_title('Throughput: Partitioned vs Monolithic')
    ax1.legend()
    ax1.grid(True)
    
    # Latency comparison
    if not boolean_mono.empty:
        ax2.plot(boolean_mono['num_cpu_workers'], boolean_mono['median_latency_ms'], 
                marker='s', markersize=8, linestyle='-', label='Monolithic')
    if not boolean_part.empty:
        ax2.plot(boolean_part['num_cpu_workers'], boolean_part['median_latency_ms'], 
                marker='^', markersize=8, linestyle='--', label='Partitioned')
    
    ax2.set_xlabel('Number of CPU Workers')
    ax2.set_ylabel('Median Latency (ms)')
    ax2.set_title('Latency: Partitioned vs Monolithic')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    filename = f"{output_dir}/partitioned_vs_monolithic.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved partitioned comparison plot: {filename}")
    plt.close()

def generate_summary_report(df, output_dir):
    if df.empty:
        return
        
    report_file = f"{output_dir}/benchmark_summary.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Monolithic Boolean
        boolean_mono_df = df[(df['use_reranking'] == False) & (df['use_partitioned'] == False)]
        if not boolean_mono_df.empty:
            f.write("--- Boolean Retrieval Performance (Monolithic) ---\n")
            best_throughput = boolean_mono_df.loc[boolean_mono_df['throughput_qps'].idxmax()]
            f.write(f"  Best Throughput: {best_throughput['throughput_qps']:.2f} q/s (at {int(best_throughput['num_cpu_workers'])} CPU workers)\n")
            f.write(f"  Average P@10: {boolean_mono_df['precision_at_10'].mean():.4f}\n")
            f.write(f"  Average MAP:  {boolean_mono_df['map'].mean():.4f}\n\n")

        # Partitioned Boolean
        boolean_part_df = df[(df['use_reranking'] == False) & (df['use_partitioned'] == True)]
        if not boolean_part_df.empty:
            f.write("--- Boolean Retrieval Performance (Partitioned) ---\n")
            best_throughput = boolean_part_df.loc[boolean_part_df['throughput_qps'].idxmax()]
            f.write(f"  Best Throughput: {best_throughput['throughput_qps']:.2f} q/s (at {int(best_throughput['num_cpu_workers'])} CPU workers)\n")
            f.write(f"  Average P@10: {boolean_part_df['precision_at_10'].mean():.4f}\n")
            f.write(f"  Average MAP:  {boolean_part_df['map'].mean():.4f}\n\n")

        # Monolithic Reranking
        rerank_mono_df = df[(df['use_reranking'] == True) & (df['use_partitioned'] == False)]
        if not rerank_mono_df.empty:
            f.write("--- Reranking Performance (Monolithic) ---\n")
            best_throughput = rerank_mono_df.loc[rerank_mono_df['throughput_qps'].idxmax()]
            f.write(f"  Best Throughput: {best_throughput['throughput_qps']:.2f} q/s (at {int(best_throughput['num_cpu_workers'])} CPU workers)\n")
            f.write(f"  Average P@10: {rerank_mono_df['precision_at_10'].mean():.4f}\n")
            f.write(f"  Average MAP:  {rerank_mono_df['map'].mean():.4f}\n\n")

        # Partitioned Reranking
        rerank_part_df = df[(df['use_reranking'] == True) & (df['use_partitioned'] == True)]
        if not rerank_part_df.empty:
            f.write("--- Reranking Performance (Partitioned) ---\n")
            best_throughput = rerank_part_df.loc[rerank_part_df['throughput_qps'].idxmax()]
            f.write(f"  Best Throughput: {best_throughput['throughput_qps']:.2f} q/s (at {int(best_throughput['num_cpu_workers'])} CPU workers)\n")
            f.write(f"  Average P@10: {rerank_part_df['precision_at_10'].mean():.4f}\n")
            f.write(f"  Average MAP:  {rerank_part_df['map'].mean():.4f}\n\n")

        # Comparisons
        if not boolean_mono_df.empty and not rerank_mono_df.empty:
             f.write("--- Comparison (Monolithic: Reranking vs Boolean) ---\n")
             map_improvement = ((rerank_mono_df['map'].mean() - boolean_mono_df['map'].mean()) / boolean_mono_df['map'].mean()) * 100
             f.write(f"  MAP Improvement with Reranking: {map_improvement:+.2f}%\n")
             p10_improvement = ((rerank_mono_df['precision_at_10'].mean() - boolean_mono_df['precision_at_10'].mean()) / boolean_mono_df['precision_at_10'].mean()) * 100
             f.write(f"  P@10 Improvement with Reranking: {p10_improvement:+.2f}%\n\n")

        if not boolean_mono_df.empty and not boolean_part_df.empty:
             f.write("--- Comparison (Boolean: Partitioned vs Monolithic) ---\n")
             throughput_change = ((boolean_part_df['throughput_qps'].mean() - boolean_mono_df['throughput_qps'].mean()) / boolean_mono_df['throughput_qps'].mean()) * 100
             f.write(f"  Throughput Change: {throughput_change:+.2f}%\n")
             latency_change = ((boolean_part_df['median_latency_ms'].mean() - boolean_mono_df['median_latency_ms'].mean()) / boolean_mono_df['median_latency_ms'].mean()) * 100
             f.write(f"  Latency Change: {latency_change:+.2f}%\n")

    print(f"Saved summary report: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate IR benchmark visualizations.')
    parser.add_argument('--results', required=True, help='Path to the consolidated results CSV file')
    parser.add_argument('--output-dir', default='results/plots', help='Directory to save plots and reports')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("BENCHMARK VISUALIZATION SUITE")
    print("=" * 80 + "\n")
    
    df = load_data(args.results)
    
    if not df.empty:
        plot_scalability(df, output_dir, 'throughput_qps', 'System Throughput vs CPU Workers', 'Throughput (Queries/Second)')
        plot_scalability(df, output_dir, 'median_latency_ms', 'Median Query Latency vs CPU Workers', 'Median Latency (ms)')
        plot_effectiveness_comparison(df, output_dir)
        plot_partitioned_vs_monolithic(df, output_dir)
        generate_summary_report(df, output_dir)
        print("\n" + "=" * 80)
        print("All visualizations completed!")
        print(f"Results saved to: {output_dir}")
        print("=" * 80 + "\n")
    else:
        print("Could not generate plots due to missing or empty data file.")

if __name__ == "__main__":
    main()