import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import sys

# Configuration
DEFAULT_QUERY_CSV = "results/all_benchmarks.csv"
DEFAULT_INDEX_CSV = "results/indexing_benchmarks.csv"
DEFAULT_PLOTS_DIR = "results/plots"

# Professional plot styling
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D84A05'
}


def load_query_data(csv_path):
    """Load and preprocess query benchmark results."""
    if not Path(csv_path).exists():
        print(f"Warning: Query results file not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        
        # Handle duplicates by averaging
        group_cols = ['label', 'num_cpu_workers', 'use_reranking']
        if all(col in df.columns for col in group_cols):
            df = df.groupby(group_cols, as_index=False).mean()
        
        print(f"Loaded {len(df)} query benchmark results")
        return df
    except Exception as e:
        print(f"Error loading query data: {e}")
        return pd.DataFrame()


def load_indexing_data(csv_path):
    """Load and preprocess indexing benchmark results."""
    if not Path(csv_path).exists():
        print(f"Warning: Indexing results file not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} indexing benchmark results")
        return df
    except Exception as e:
        print(f"Error loading indexing data: {e}")
        return pd.DataFrame()


# =====================================================================
# INDEXING PERFORMANCE PLOTS
# =====================================================================

def plot_indexing_throughput(df, output_dir):
    """Plot indexing throughput vs workers with ideal scaling reference."""
    if df.empty or 'num_cpu_workers' not in df.columns:
        print("Skipping indexing throughput plot: missing data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_sorted = df.sort_values('num_cpu_workers')
    workers = df_sorted['num_cpu_workers'].values
    throughput = df_sorted['throughput_docs_per_sec'].values
    
    # Actual throughput
    ax.plot(workers, throughput, marker='o', linewidth=2.5, markersize=10,
            label='Actual Throughput', color=COLORS['primary'])
    
    # Ideal linear scaling reference (from first data point)
    if len(workers) > 0:
        base_throughput = throughput[0]
        base_workers = workers[0]
        ideal_throughput = base_throughput * (workers / base_workers)
        ax.plot(workers, ideal_throughput, '--', linewidth=2, alpha=0.6,
                label='Ideal Linear Scaling', color=COLORS['warning'])
    
    ax.set_xlabel('Number of CPU Workers', fontweight='bold')
    ax.set_ylabel('Throughput (Documents/Second)', fontweight='bold')
    ax.set_title('Indexing Throughput Scalability', fontweight='bold', pad=20)
    ax.set_xticks(workers)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (w, t) in enumerate(zip(workers, throughput)):
        ax.annotate(f'{int(t):,}', xy=(w, t), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9)
    
    filename = output_dir / "indexing_throughput_scalability.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_indexing_time(df, output_dir):
    """Plot indexing time vs workers."""
    if df.empty or 'num_cpu_workers' not in df.columns:
        print("Skipping indexing time plot: missing data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_sorted = df.sort_values('num_cpu_workers')
    workers = df_sorted['num_cpu_workers'].values
    time_s = df_sorted['indexing_time_ms'].values / 1000  # Convert to seconds
    
    ax.plot(workers, time_s, marker='s', linewidth=2.5, markersize=10,
            color=COLORS['secondary'], label='Indexing Time')
    
    ax.set_xlabel('Number of CPU Workers', fontweight='bold')
    ax.set_ylabel('Total Indexing Time (seconds)', fontweight='bold')
    ax.set_title('Indexing Time vs Parallelism', fontweight='bold', pad=20)
    ax.set_xticks(workers)
    ax.legend(framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Add time labels
    for w, t in zip(workers, time_s):
        ax.annotate(f'{t:.1f}s', xy=(w, t), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9)
    
    filename = output_dir / "indexing_time_scalability.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_indexing_speedup(df, output_dir):
    """Plot speedup and parallel efficiency for indexing."""
    if df.empty or 'num_cpu_workers' not in df.columns:
        print("Skipping indexing speedup plot: missing data")
        return
    
    df_sorted = df.sort_values('num_cpu_workers')
    workers = df_sorted['num_cpu_workers'].values
    time_ms = df_sorted['indexing_time_ms'].values
    
    if len(workers) == 0:
        return
    
    # Calculate speedup (relative to first measurement)
    base_time = time_ms[0]
    speedup = base_time / time_ms
    
    # Calculate parallel efficiency
    efficiency = (speedup / workers) * 100  # As percentage
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Speedup plot
    ax1.plot(workers, speedup, marker='o', linewidth=2.5, markersize=10,
             label='Actual Speedup', color=COLORS['primary'])
    ax1.plot(workers, workers, '--', linewidth=2, alpha=0.6,
             label='Ideal Speedup', color=COLORS['warning'])
    ax1.set_xlabel('Number of CPU Workers', fontweight='bold')
    ax1.set_ylabel('Speedup', fontweight='bold')
    ax1.set_title('Indexing Speedup', fontweight='bold', pad=15)
    ax1.set_xticks(workers)
    ax1.legend(framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    for w, s in zip(workers, speedup):
        ax1.annotate(f'{s:.2f}x', xy=(w, s), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)
    
    # Efficiency plot
    ax2.plot(workers, efficiency, marker='s', linewidth=2.5, markersize=10,
             color=COLORS['success'])
    ax2.axhline(y=100, linestyle='--', linewidth=2, alpha=0.6,
                color=COLORS['warning'], label='Ideal (100%)')
    ax2.set_xlabel('Number of CPU Workers', fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
    ax2.set_title('Indexing Parallel Efficiency', fontweight='bold', pad=15)
    ax2.set_xticks(workers)
    ax2.set_ylim([0, 110])
    ax2.legend(framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    
    for w, e in zip(workers, efficiency):
        ax2.annotate(f'{e:.1f}%', xy=(w, e), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    filename = output_dir / "indexing_speedup_efficiency.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =====================================================================
# QUERY PERFORMANCE PLOTS
# =====================================================================

def plot_query_throughput(df, output_dir):
    """Plot query throughput for Boolean and Reranking."""
    if df.empty:
        print("Skipping query throughput plot: missing data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    boolean_df = df[df['use_reranking'] == False].sort_values('num_cpu_workers')
    rerank_df = df[df['use_reranking'] == True].sort_values('num_cpu_workers')
    
    if not boolean_df.empty:
        ax.plot(boolean_df['num_cpu_workers'], boolean_df['throughput_qps'],
                marker='o', linewidth=2.5, markersize=10, label='Boolean Retrieval',
                color=COLORS['primary'])
    
    if not rerank_df.empty:
        ax.plot(rerank_df['num_cpu_workers'], rerank_df['throughput_qps'],
                marker='s', linewidth=2.5, markersize=10, label='End-to-End Reranking',
                color=COLORS['secondary'])
    
    ax.set_xlabel('Number of CPU Workers', fontweight='bold')
    ax.set_ylabel('Throughput (Queries/Second)', fontweight='bold')
    ax.set_title('Query Processing Throughput', fontweight='bold', pad=20)
    ax.set_xticks(sorted(df['num_cpu_workers'].unique()))
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    filename = output_dir / "query_throughput_scalability.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_query_latency(df, output_dir):
    """Plot query latency (median and p95)."""
    if df.empty:
        print("Skipping query latency plot: missing data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    boolean_df = df[df['use_reranking'] == False].sort_values('num_cpu_workers')
    rerank_df = df[df['use_reranking'] == True].sort_values('num_cpu_workers')
    
    # Median latency
    if not boolean_df.empty:
        ax1.plot(boolean_df['num_cpu_workers'], boolean_df['median_latency_ms'],
                marker='o', linewidth=2.5, markersize=10, label='Boolean',
                color=COLORS['primary'])
    if not rerank_df.empty:
        ax1.plot(rerank_df['num_cpu_workers'], rerank_df['median_latency_ms'],
                marker='s', linewidth=2.5, markersize=10, label='Reranking',
                color=COLORS['secondary'])
    
    ax1.set_xlabel('Number of CPU Workers', fontweight='bold')
    ax1.set_ylabel('Median Latency (ms)', fontweight='bold')
    ax1.set_title('Median Query Latency', fontweight='bold', pad=15)
    ax1.set_xticks(sorted(df['num_cpu_workers'].unique()))
    ax1.legend(framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # P95 latency
    if not boolean_df.empty:
        ax2.plot(boolean_df['num_cpu_workers'], boolean_df['p95_latency_ms'],
                marker='o', linewidth=2.5, markersize=10, label='Boolean',
                color=COLORS['primary'])
    if not rerank_df.empty:
        ax2.plot(rerank_df['num_cpu_workers'], rerank_df['p95_latency_ms'],
                marker='s', linewidth=2.5, markersize=10, label='Reranking',
                color=COLORS['secondary'])
    
    ax2.set_xlabel('Number of CPU Workers', fontweight='bold')
    ax2.set_ylabel('P95 Latency (ms)', fontweight='bold')
    ax2.set_title('95th Percentile Query Latency', fontweight='bold', pad=15)
    ax2.set_xticks(sorted(df['num_cpu_workers'].unique()))
    ax2.legend(framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = output_dir / "query_latency_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =====================================================================
# EFFECTIVENESS PLOTS
# =====================================================================

def plot_effectiveness_metrics(df, output_dir):
    """Plot effectiveness metrics across worker counts."""
    if df.empty:
        print("Skipping effectiveness metrics plot: missing data")
        return
    
    metrics = ['precision_at_10', 'map', 'ndcg_at_10']
    metric_labels = ['Precision@10', 'MAP', 'nDCG@10']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    boolean_df = df[df['use_reranking'] == False].sort_values('num_cpu_workers')
    rerank_df = df[df['use_reranking'] == True].sort_values('num_cpu_workers')
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        if not boolean_df.empty and metric in boolean_df.columns:
            ax.plot(boolean_df['num_cpu_workers'], boolean_df[metric],
                   marker='o', linewidth=2.5, markersize=10, label='Boolean',
                   color=COLORS['primary'])
        
        if not rerank_df.empty and metric in rerank_df.columns:
            ax.plot(rerank_df['num_cpu_workers'], rerank_df[metric],
                   marker='s', linewidth=2.5, markersize=10, label='Reranking',
                   color=COLORS['secondary'])
        
        ax.set_xlabel('Number of CPU Workers', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.set_title(f'{label} vs Workers', fontweight='bold', pad=15)
        if len(df['num_cpu_workers'].unique()) > 0:
            ax.set_xticks(sorted(df['num_cpu_workers'].unique()))
        ax.legend(framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = output_dir / "effectiveness_metrics_scalability.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_effectiveness_bars(df, output_dir):
    """Bar chart comparing Boolean vs Reranking effectiveness."""
    if df.empty:
        print("Skipping effectiveness bars: missing data")
        return
    
    max_workers = df['num_cpu_workers'].max()
    boolean_metrics = df[(df['use_reranking'] == False) & 
                         (df['num_cpu_workers'] == max_workers)]
    rerank_metrics = df[(df['use_reranking'] == True) & 
                        (df['num_cpu_workers'] == max_workers)]
    
    if boolean_metrics.empty or rerank_metrics.empty:
        print("Skipping effectiveness bars: missing data for max workers")
        return
    
    metrics = ['precision_at_10', 'map', 'ndcg_at_10']
    labels = ['P@10', 'MAP', 'nDCG@10']
    
    boolean_vals = [boolean_metrics[m].mean() for m in metrics]
    rerank_vals = [rerank_metrics[m].mean() for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, boolean_vals, width, label='Boolean',
                   color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, rerank_vals, width, label='Reranking',
                   color=COLORS['secondary'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'Effectiveness Comparison (at {int(max_workers)} workers)',
                fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    filename = output_dir / "effectiveness_comparison_bars.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


# =====================================================================
# SUMMARY REPORT
# =====================================================================

def generate_summary_report(query_df, index_df, output_dir):
    """Generate comprehensive text summary."""
    report_file = output_dir / "performance_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("IR SYSTEM PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Indexing performance
        if not index_df.empty:
            f.write("INDEXING PERFORMANCE\n")
            f.write("-" * 80 + "\n\n")
            
            for _, row in index_df.iterrows():
                workers = int(row['num_cpu_workers'])
                time_s = row['indexing_time_ms'] / 1000
                throughput = row['throughput_docs_per_sec']
                f.write(f"  {workers} workers:\n")
                f.write(f"    Time: {time_s:.2f} seconds\n")
                f.write(f"    Throughput: {throughput:,.0f} docs/sec\n\n")
            
            # Calculate speedup
            if len(index_df) > 1:
                base_time = index_df.iloc[0]['indexing_time_ms']
                f.write("  Speedup Analysis:\n")
                for _, row in index_df.iterrows():
                    workers = int(row['num_cpu_workers'])
                    speedup = base_time / row['indexing_time_ms']
                    efficiency = (speedup / workers) * 100
                    f.write(f"    {workers} workers: {speedup:.2f}x speedup, "
                           f"{efficiency:.1f}% efficiency\n")
                f.write("\n")
        
        # Query performance
        if not query_df.empty:
            f.write("\n" + "=" * 80 + "\n")
            f.write("QUERY PROCESSING PERFORMANCE\n")
            f.write("=" * 80 + "\n\n")
            
            boolean_df = query_df[query_df['use_reranking'] == False]
            rerank_df = query_df[query_df['use_reranking'] == True]
            
            if not boolean_df.empty:
                f.write("Boolean Retrieval:\n")
                f.write("-" * 80 + "\n")
                for _, row in boolean_df.iterrows():
                    workers = int(row['num_cpu_workers'])
                    f.write(f"\n  {workers} workers:\n")
                    f.write(f"    Throughput: {row['throughput_qps']:.2f} q/s\n")
                    f.write(f"    Median Latency: {row['median_latency_ms']:.2f} ms\n")
                    f.write(f"    P95 Latency: {row['p95_latency_ms']:.2f} ms\n")
                    f.write(f"    P@10: {row['precision_at_10']:.4f}\n")
                    f.write(f"    MAP: {row['map']:.4f}\n")
                    f.write(f"    nDCG@10: {row['ndcg_at_10']:.4f}\n")
            
            if not rerank_df.empty:
                f.write("\n\nEnd-to-End Reranking:\n")
                f.write("-" * 80 + "\n")
                for _, row in rerank_df.iterrows():
                    workers = int(row['num_cpu_workers'])
                    f.write(f"\n  {workers} workers:\n")
                    f.write(f"    Throughput: {row['throughput_qps']:.2f} q/s\n")
                    f.write(f"    Median Latency: {row['median_latency_ms']:.2f} ms\n")
                    f.write(f"    P95 Latency: {row['p95_latency_ms']:.2f} ms\n")
                    f.write(f"    P@10: {row['precision_at_10']:.4f}\n")
                    f.write(f"    MAP: {row['map']:.4f}\n")
                    f.write(f"    nDCG@10: {row['ndcg_at_10']:.4f}\n")
            
            # Effectiveness improvement
            if not boolean_df.empty and not rerank_df.empty:
                max_workers = query_df['num_cpu_workers'].max()
                bool_max = boolean_df[boolean_df['num_cpu_workers'] == max_workers]
                rerank_max = rerank_df[rerank_df['num_cpu_workers'] == max_workers]
                
                if not bool_max.empty and not rerank_max.empty:
                    f.write("\n\n" + "=" * 80 + "\n")
                    f.write(f"IMPROVEMENT WITH RERANKING (at {int(max_workers)} workers)\n")
                    f.write("=" * 80 + "\n\n")
                    
                    metrics = [('P@10', 'precision_at_10'),
                              ('MAP', 'map'),
                              ('nDCG@10', 'ndcg_at_10')]
                    
                    for label, col in metrics:
                        bool_val = bool_max[col].values[0]
                        rerank_val = rerank_max[col].values[0]
                        improvement = ((rerank_val - bool_val) / bool_val) * 100
                        f.write(f"  {label}: {bool_val:.4f} -> {rerank_val:.4f} "
                               f"({improvement:+.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated successfully\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved: {report_file}")
    
    # Print to console as well
    with open(report_file, 'r') as f:
        print("\n" + f.read())


# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive visualizations for IR system benchmarks'
    )
    parser.add_argument('--query-results', default=DEFAULT_QUERY_CSV,
                       help=f'Path to query results CSV (default: {DEFAULT_QUERY_CSV})')
    parser.add_argument('--index-results', default=DEFAULT_INDEX_CSV,
                       help=f'Path to indexing results CSV (default: {DEFAULT_INDEX_CSV})')
    parser.add_argument('--output-dir', default=DEFAULT_PLOTS_DIR,
                       help=f'Output directory for plots (default: {DEFAULT_PLOTS_DIR})')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("IR SYSTEM COMPREHENSIVE EVALUATION")
    print("=" * 80 + "\n")
    
    # Load data
    print("Loading data...")
    query_df = load_query_data(args.query_results)
    index_df = load_indexing_data(args.index_results)
    
    # Generate indexing plots
    if not index_df.empty:
        print("\nGenerating indexing performance plots...")
        plot_indexing_throughput(index_df, output_dir)
        plot_indexing_time(index_df, output_dir)
        plot_indexing_speedup(index_df, output_dir)
    
    # Generate query plots
    if not query_df.empty:
        print("\nGenerating query performance plots...")
        plot_query_throughput(query_df, output_dir)
        plot_query_latency(query_df, output_dir)
        plot_effectiveness_metrics(query_df, output_dir)
        plot_effectiveness_bars(query_df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(query_df, index_df, output_dir)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print(f"All plots and reports saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()