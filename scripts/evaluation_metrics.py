import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

def load_data(csv_file):
    """Load benchmark results from CSV"""
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} benchmark results from {csv_file}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def plot_scalability_throughput(df, output_dir):
    """Plot throughput vs number of workers"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rerank in [False, True]:
        subset = df[df['use_reranking'] == rerank]
        label = "With Reranking" if rerank else "Boolean Only"
        marker = 'o' if rerank else 's'
        ax.plot(subset['num_workers'], subset['throughput_qps'], 
                marker=marker, markersize=10, linewidth=2, label=label)
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Throughput (queries/second)')
    ax.set_title('Query Processing Throughput vs Parallelism')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scalability_throughput.png", dpi=300)
    print(f"✓ Saved: {output_dir}/scalability_throughput.png")
    plt.close()

def plot_scalability_speedup(df, output_dir):
    """Plot speedup vs number of workers"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rerank in [False, True]:
        subset = df[df['use_reranking'] == rerank].sort_values('num_workers')
        if len(subset) == 0:
            continue
            
        baseline_time = subset.iloc[0]['query_processing_time_ms']
        speedup = baseline_time / subset['query_processing_time_ms']
        
        label = "With Reranking" if rerank else "Boolean Only"
        marker = 'o' if rerank else 's'
        ax.plot(subset['num_workers'], speedup, 
                marker=marker, markersize=10, linewidth=2, label=label)
    
    # Add ideal speedup line
    max_workers = df['num_workers'].max()
    ax.plot([1, max_workers], [1, max_workers], 
            'k--', linewidth=1, alpha=0.5, label='Ideal Linear Speedup')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup')
    ax.set_title('Parallel Speedup Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scalability_speedup.png", dpi=300)
    print(f"✓ Saved: {output_dir}/scalability_speedup.png")
    plt.close()

def plot_effectiveness_comparison(df, output_dir):
    """Plot P@10, MAP, MRR comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['precision_at_10', 'map', 'mrr']
    titles = ['Precision@10', 'Mean Average Precision', 'Mean Reciprocal Rank']
    
    for ax, metric, title in zip(axes, metrics, titles):
        boolean_vals = df[df['use_reranking'] == False][metric]
        rerank_vals = df[df['use_reranking'] == True][metric]
        
        x = np.arange(2)
        values = [boolean_vals.mean(), rerank_vals.mean()]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['Boolean\nRetrieval', 'Neural\nReranking'])
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_ylim([0, max(values) * 1.2])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/effectiveness_comparison.png", dpi=300)
    print(f"✓ Saved: {output_dir}/effectiveness_comparison.png")
    plt.close()

def plot_timing_breakdown(df, output_dir):
    """Plot timing breakdown by component"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Select a representative configuration (e.g., max workers)
    max_workers = df['num_workers'].max()
    subset = df[df['num_workers'] == max_workers]
    
    labels = []
    indexing_times = []
    retrieval_times = []
    
    for _, row in subset.iterrows():
        label = "Boolean" if not row['use_reranking'] else "Reranking"
        labels.append(label)
        indexing_times.append(row['indexing_time_ms'])
        retrieval_times.append(row['query_processing_time_ms'])
    
    x = np.arange(len(labels))
    width = 0.35
    
    p1 = ax.bar(x, indexing_times, width, label='Indexing', color='#3498db')
    p2 = ax.bar(x, retrieval_times, width, bottom=indexing_times, 
                label='Query Processing', color='#e74c3c')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Timing Breakdown ({max_workers} workers)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timing_breakdown.png", dpi=300)
    print(f"✓ Saved: {output_dir}/timing_breakdown.png")
    plt.close()

def plot_latency_percentiles(df, output_dir):
    """Plot latency percentiles"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select a representative configuration
    max_workers = df['num_workers'].max()
    subset = df[df['num_workers'] == max_workers]
    
    percentiles = ['avg_retrieval_ms', 'median_retrieval_ms', 'p95_retrieval_ms']
    percentile_labels = ['Average', 'Median (P50)', '95th Percentile']
    
    x = np.arange(len(percentile_labels))
    width = 0.35
    
    boolean_data = subset[subset['use_reranking'] == False]
    rerank_data = subset[subset['use_reranking'] == True]
    
    boolean_vals = [boolean_data[p].values[0] if len(boolean_data) > 0 else 0 for p in percentiles]
    rerank_vals = [rerank_data[p].values[0] if len(rerank_data) > 0 else 0 for p in percentiles]
    
    bars1 = ax.bar(x - width/2, boolean_vals, width, label='Boolean', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, rerank_vals, width, label='Reranking', 
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Query Latency Percentiles ({max_workers} workers)')
    ax.set_xticks(x)
    ax.set_xticklabels(percentile_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_percentiles.png", dpi=300)
    print(f"✓ Saved: {output_dir}/latency_percentiles.png")
    plt.close()

def plot_efficiency_vs_effectiveness(df, output_dir):
    """Plot efficiency (throughput) vs effectiveness (MAP)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rerank in [False, True]:
        subset = df[df['use_reranking'] == rerank]
        label = "With Reranking" if rerank else "Boolean Only"
        marker = 'o' if rerank else 's'
        color = '#e74c3c' if rerank else '#3498db'
        
        ax.scatter(subset['throughput_qps'], subset['map'], 
                  s=subset['num_workers']*20, alpha=0.6,
                  marker=marker, label=label, c=color, edgecolors='black', linewidth=1.5)
        
        # Add worker count labels
        for _, row in subset.iterrows():
            ax.annotate(f"{int(row['num_workers'])}w", 
                       (row['throughput_qps'], row['map']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Throughput (queries/second)')
    ax.set_ylabel('Mean Average Precision (MAP)')
    ax.set_title('Efficiency vs Effectiveness Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/efficiency_vs_effectiveness.png", dpi=300)
    print(f"✓ Saved: {output_dir}/efficiency_vs_effectiveness.png")
    plt.close()

def plot_per_query_distribution(query_csv, output_dir):
    """Plot per-query metric distributions"""
    if not Path(query_csv).exists():
        print(f"⚠ Query metrics file not found: {query_csv}")
        return
    
    df_query = pd.read_csv(query_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P@10 distribution
    axes[0, 0].hist(df_query['precision_at_10'], bins=20, color='#3498db', 
                    alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Precision@10')
    axes[0, 0].set_ylabel('Number of Queries')
    axes[0, 0].set_title('Distribution of Precision@10')
    axes[0, 0].axvline(df_query['precision_at_10'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df_query["precision_at_10"].mean():.3f}')
    axes[0, 0].legend()
    
    # AP distribution
    axes[0, 1].hist(df_query['average_precision'], bins=20, color='#e74c3c', 
                    alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Average Precision')
    axes[0, 1].set_ylabel('Number of Queries')
    axes[0, 1].set_title('Distribution of Average Precision')
    axes[0, 1].axvline(df_query['average_precision'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df_query["average_precision"].mean():.3f}')
    axes[0, 1].legend()
    
    # Retrieval time distribution
    axes[1, 0].hist(df_query['retrieval_time_ms'], bins=30, color='#2ecc71', 
                    alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Retrieval Time (ms)')
    axes[1, 0].set_ylabel('Number of Queries')
    axes[1, 0].set_title('Distribution of Query Latency')
    axes[1, 0].axvline(df_query['retrieval_time_ms'].median(), color='red', 
                       linestyle='--', linewidth=2, label=f'Median: {df_query["retrieval_time_ms"].median():.1f}ms')
    axes[1, 0].legend()
    
    # Candidates vs retrieval time
    axes[1, 1].scatter(df_query['num_candidates'], df_query['retrieval_time_ms'], 
                      alpha=0.5, c='#9b59b6', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Number of Candidate Documents')
    axes[1, 1].set_ylabel('Retrieval Time (ms)')
    axes[1, 1].set_title('Query Complexity vs Latency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_query_distributions.png", dpi=300)
    print(f"✓ Saved: {output_dir}/per_query_distributions.png")
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate text summary report"""
    report_file = f"{output_dir}/benchmark_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total configurations tested: {len(df)}\n")
        f.write(f"Worker counts: {sorted(df['num_workers'].unique())}\n\n")
        
        # Best configurations
        f.write("BEST CONFIGURATIONS\n")
        f.write("-" * 80 + "\n")
        
        best_throughput = df.loc[df['throughput_qps'].idxmax()]
        f.write(f"Highest Throughput: {best_throughput['throughput_qps']:.2f} qps\n")
        f.write(f"  Configuration: {best_throughput['label']}\n")
        f.write(f"  Workers: {int(best_throughput['num_workers'])}\n\n")
        
        best_map = df.loc[df['map'].idxmax()]
        f.write(f"Best MAP: {best_map['map']:.4f}\n")
        f.write(f"  Configuration: {best_map['label']}\n")
        f.write(f"  Workers: {int(best_map['num_workers'])}\n\n")
        
        best_p10 = df.loc[df['precision_at_10'].idxmax()]
        f.write(f"Best P@10: {best_p10['precision_at_10']:.4f}\n")
        f.write(f"  Configuration: {best_p10['label']}\n")
        f.write(f"  Workers: {int(best_p10['num_workers'])}\n\n")
        
        # Comparison: Boolean vs Reranking
        f.write("BOOLEAN VS NEURAL RERANKING COMPARISON\n")
        f.write("-" * 80 + "\n")
        
        boolean_avg = df[df['use_reranking'] == False].mean()
        rerank_avg = df[df['use_reranking'] == True].mean()
        
        f.write(f"Boolean Retrieval (avg across all configs):\n")
        f.write(f"  Throughput: {boolean_avg['throughput_qps']:.2f} qps\n")
        f.write(f"  P@10: {boolean_avg['precision_at_10']:.4f}\n")
        f.write(f"  MAP: {boolean_avg['map']:.4f}\n")
        f.write(f"  MRR: {boolean_avg['mrr']:.4f}\n\n")
        
        f.write(f"Neural Reranking (avg across all configs):\n")
        f.write(f"  Throughput: {rerank_avg['throughput_qps']:.2f} qps\n")
        f.write(f"  P@10: {rerank_avg['precision_at_10']:.4f}\n")
        f.write(f"  MAP: {rerank_avg['map']:.4f}\n")
        f.write(f"  MRR: {rerank_avg['mrr']:.4f}\n\n")
        
        # Improvements
        throughput_change = ((rerank_avg['throughput_qps'] - boolean_avg['throughput_qps']) / 
                            boolean_avg['throughput_qps'] * 100)
        p10_improvement = ((rerank_avg['precision_at_10'] - boolean_avg['precision_at_10']) / 
                          (boolean_avg['precision_at_10'] + 1e-10) * 100)
        map_improvement = ((rerank_avg['map'] - boolean_avg['map']) / 
                          (boolean_avg['map'] + 1e-10) * 100)
        
        f.write(f"Impact of Neural Reranking:\n")
        f.write(f"  Throughput change: {throughput_change:+.1f}%\n")
        f.write(f"  P@10 improvement: {p10_improvement:+.1f}%\n")
        f.write(f"  MAP improvement: {map_improvement:+.1f}%\n\n")
        
        # Scalability analysis
        f.write("SCALABILITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        for rerank in [False, True]:
            subset = df[df['use_reranking'] == rerank].sort_values('num_workers')
            if len(subset) < 2:
                continue
            
            label = "With Reranking" if rerank else "Boolean Only"
            baseline = subset.iloc[0]
            final = subset.iloc[-1]
            
            speedup = baseline['query_processing_time_ms'] / final['query_processing_time_ms']
            efficiency = (speedup / final['num_workers']) * 100
            
            f.write(f"{label}:\n")
            f.write(f"  Workers: {int(baseline['num_workers'])} → {int(final['num_workers'])}\n")
            f.write(f"  Speedup: {speedup:.2f}x\n")
            f.write(f"  Parallel Efficiency: {efficiency:.1f}%\n\n")
    
    print(f"✓ Saved: {report_file}")

def main():
    """Main visualization pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IR benchmark visualizations')
    parser.add_argument('--results', required=True, help='Path to results CSV file')
    parser.add_argument('--query-metrics', help='Path to per-query metrics CSV file')
    parser.add_argument('--output-dir', default='results/plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("BENCHMARK VISUALIZATION SUITE")
    print("=" * 80 + "\n")
    
    # Load data
    df = load_data(args.results)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_scalability_throughput(df, output_dir)
    plot_scalability_speedup(df, output_dir)
    plot_effectiveness_comparison(df, output_dir)
    plot_timing_breakdown(df, output_dir)
    plot_latency_percentiles(df, output_dir)
    plot_efficiency_vs_effectiveness(df, output_dir)
    
    if args.query_metrics:
        plot_per_query_distribution(args.query_metrics, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ All visualizations completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()