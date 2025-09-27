import sys
import matplotlib.pyplot as plt
import numpy as np

def create_comparison_plots(results_dir):
    """Create comparison plots for baseline vs neural results"""
    
    # Create sample plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Precision@K plot
    k_values = [1, 5, 10, 20, 50]
    baseline_precision = [0.8, 0.7, 0.6, 0.5, 0.4]  # Sample data
    neural_precision = [0.9, 0.85, 0.8, 0.75, 0.7]  # Sample data
    
    axes[0,0].plot(k_values, baseline_precision, 'b-o', label='Baseline')
    axes[0,0].plot(k_values, neural_precision, 'r-o', label='Neural Re-ranking')
    axes[0,0].set_xlabel('k')
    axes[0,0].set_ylabel('Precision@k')
    axes[0,0].set_title('Precision@k Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Similar plots for Recall@K, MAP, etc.
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved to {results_dir}/comparison_plots.png")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 plot_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    create_comparison_plots(results_dir)

if __name__ == "__main__":
    main()