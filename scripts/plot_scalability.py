import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_scalability(csv_file):
    """Plot OpenCilk scalability results"""
    try:
        # Read the CSV data
        df = pd.read_csv(csv_file, names=['workers', 'wall_time', 'user_time', 'sys_time', 'memory'])
        
        # Convert time formats (assuming format like "0:01.23")
        def parse_time(time_str):
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(time_str)
        
        df['wall_seconds'] = df['wall_time'].apply(parse_time)
        df['user_seconds'] = df['user_time'].apply(parse_time)
        
        # Calculate speedup (relative to single worker)
        baseline_time = df[df['workers'] == 1]['wall_seconds'].iloc[0]
        df['speedup'] = baseline_time / df['wall_seconds']
        df['efficiency'] = df['speedup'] / df['workers'] * 100
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Wall time vs workers
        ax1.plot(df['workers'], df['wall_seconds'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Workers')
        ax1.set_ylabel('Wall Clock Time (seconds)')
        ax1.set_title('Execution Time vs Number of Workers')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Plot 2: Speedup vs workers
        ax2.plot(df['workers'], df['speedup'], 'ro-', linewidth=2, markersize=8, label='Actual Speedup')
        ax2.plot(df['workers'], df['workers'], 'g--', linewidth=2, label='Ideal Speedup')
        ax2.set_xlabel('Number of Workers')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup vs Number of Workers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log', base=2)
        
        # Plot 3: Parallel efficiency
        ax3.plot(df['workers'], df['efficiency'], 'mo-', linewidth=2, markersize=8)
        ax3.axhline(y=100, color='g', linestyle='--', label='Perfect Efficiency')
        ax3.set_xlabel('Number of Workers')
        ax3.set_ylabel('Parallel Efficiency (%)')
        ax3.set_title('Parallel Efficiency vs Number of Workers')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Plot 4: Memory usage
        ax4.plot(df['workers'], df['memory']/1024, 'co-', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Workers')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Usage vs Number of Workers')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.suptitle('OpenCilk Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        output_file = csv_file.replace('.csv', '_scalability.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Scalability plots saved to {output_file}")
        
        # Print summary statistics
        print("\n=== OpenCilk Scalability Summary ===")
        print(f"Best speedup: {df['speedup'].max():.2f}x with {df.loc[df['speedup'].idxmax(), 'workers']} workers")
        print(f"Best efficiency: {df['efficiency'].max():.1f}% with {df.loc[df['efficiency'].idxmax(), 'workers']} workers")
        
    except Exception as e:
        print(f"Error plotting scalability: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 plot_scalability.py <scalability_results.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    plot_scalability(csv_file)

if __name__ == "__main__":
    main()