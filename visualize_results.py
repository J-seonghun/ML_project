#!/usr/bin/env python3
"""
Visualization Script for ML Scheduler Performance
Creates comparison graphs for Rodinia and Deepbench benchmarks
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

def load_csv_results(filepath):
    """Load CSV file and extract IPC data"""
    results = {}
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find the IPC section
        ipc_start = None
        for i, line in enumerate(lines):
            if 'gpu_tot_ipc' in line or 'gpu_ipc' in line:
                ipc_start = i
                break
        
        if ipc_start is None:
            return results
        
        # Parse IPC data
        for line in lines[ipc_start+2:]:
            line = line.strip()
            if not line or line.startswith('---'):
                break
            
            parts = line.split(',')
            if len(parts) >= 2:
                benchmark = parts[0].strip()
                try:
                    ipc = float(parts[1].strip())
                    results[benchmark] = ipc
                except ValueError:
                    continue
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    
    return results

def create_comparison_plots(base_dir="/home/HDD/jeong/accel_sim_traces"):
    """Create comparison plots for Rodinia and Deepbench"""
    
    # Load data
    rodinia_baseline = load_csv_results(f"{base_dir}/accel-Ampere_251118.csv")
    rodinia_ml = load_csv_results(f"{base_dir}/ML_rodinia_ampere_251124.csv")
    deepbench_baseline = load_csv_results(f"{base_dir}/deepbench_ampere_251124.csv")
    deepbench_ml = load_csv_results(f"{base_dir}/ML_deepbench_ampere_251125.csv")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'baseline': '#3498db', 'ml': '#e74c3c', 'improved': '#2ecc71', 'degraded': '#e67e22'}
    
    #############################################
    # 1. Rodinia Bar Chart
    #############################################
    ax1 = plt.subplot(2, 2, 1)
    
    common_rodinia = set(rodinia_baseline.keys()) & set(rodinia_ml.keys())
    rodinia_data = []
    for bench in sorted(common_rodinia):
        if rodinia_baseline[bench] > 0:
            rodinia_data.append({
                'name': bench.split('/')[0].replace('-rodinia-2.0-ft', ''),
                'baseline': rodinia_baseline[bench],
                'ml': rodinia_ml[bench],
                'improvement': ((rodinia_ml[bench] - rodinia_baseline[bench]) / rodinia_baseline[bench]) * 100
            })
    
    if rodinia_data:
        names = [d['name'] for d in rodinia_data]
        baseline_ipcs = [d['baseline'] for d in rodinia_data]
        ml_ipcs = [d['ml'] for d in rodinia_data]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_ipcs, width, label='Baseline (LRR)', color=colors['baseline'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, ml_ipcs, width, label='ML Scheduler', color=colors['ml'], alpha=0.8)
        
        ax1.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
        ax1.set_ylabel('IPC', fontsize=12, fontweight='bold')
        ax1.set_title('Rodinia Benchmark Suite (Training Set)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    #############################################
    # 2. Rodinia Improvement Plot
    #############################################
    ax2 = plt.subplot(2, 2, 2)
    
    if rodinia_data:
        improvements = [d['improvement'] for d in rodinia_data]
        bar_colors = [colors['improved'] if imp > 0 else colors['degraded'] for imp in improvements]
        
        bars = ax2.barh(names, improvements, color=bar_colors, alpha=0.8)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Performance Change (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Rodinia: ML vs Baseline', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            label_x = val + (0.5 if val > 0 else -0.5)
            ax2.text(label_x, i, f'{val:+.1f}%', va='center', fontsize=9, fontweight='bold')
    
    #############################################
    # 3. Deepbench Bar Chart (Top benchmarks)
    #############################################
    ax3 = plt.subplot(2, 2, 3)
    
    common_deepbench = set(deepbench_baseline.keys()) & set(deepbench_ml.keys())
    deepbench_data = []
    for bench in sorted(common_deepbench):
        if deepbench_baseline[bench] > 0:
            improvement = ((deepbench_ml[bench] - deepbench_baseline[bench]) / deepbench_baseline[bench]) * 100
            deepbench_data.append({
                'name': bench.split('/')[0] + '/' + bench.split('/')[1][:20],
                'full_name': bench,
                'baseline': deepbench_baseline[bench],
                'ml': deepbench_ml[bench],
                'improvement': improvement
            })
    
    # Sort by absolute improvement and take top 10
    deepbench_data_sorted = sorted(deepbench_data, key=lambda x: abs(x['improvement']), reverse=True)[:10]
    
    if deepbench_data_sorted:
        names_db = [d['name'] for d in deepbench_data_sorted]
        baseline_ipcs_db = [d['baseline'] for d in deepbench_data_sorted]
        ml_ipcs_db = [d['ml'] for d in deepbench_data_sorted]
        
        x_db = np.arange(len(names_db))
        
        bars1 = ax3.bar(x_db - width/2, baseline_ipcs_db, width, label='Baseline (LRR)', color=colors['baseline'], alpha=0.8)
        bars2 = ax3.bar(x_db + width/2, ml_ipcs_db, width, label='ML Scheduler', color=colors['ml'], alpha=0.8)
        
        ax3.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
        ax3.set_ylabel('IPC', fontsize=12, fontweight='bold')
        ax3.set_title('Deepbench (Top 10 by Change)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_db)
        ax3.set_xticklabels(names_db, rotation=45, ha='right', fontsize=9)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    #############################################
    # 4. Overall Distribution
    #############################################
    ax4 = plt.subplot(2, 2, 4)
    
    all_improvements = [d['improvement'] for d in rodinia_data] + [d['improvement'] for d in deepbench_data]
    
    if all_improvements:
        bins = [-15, -10, -5, -1, 1, 5, 10, 800]
        labels = ['<-10%', '-10 to -5%', '-5 to -1%', '-1 to 1%', '1 to 5%', '5 to 10%', '>10%']
        
        hist_data = []
        for i in range(len(bins)-1):
            count = sum(1 for x in all_improvements if bins[i] <= x < bins[i+1])
            hist_data.append(count)
        
        bar_colors_dist = [colors['degraded'], colors['degraded'], colors['degraded'], 
                          'gray', colors['improved'], colors['improved'], colors['improved']]
        
        bars = ax4.barh(labels, hist_data, color=bar_colors_dist, alpha=0.8)
        ax4.set_xlabel('Number of Benchmarks', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Performance Change Range', fontsize=12, fontweight='bold')
        ax4.set_title('Overall Performance Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, hist_data):
            if val > 0:
                ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), 
                        va='center', fontsize=10, fontweight='bold')
        
        # Add summary statistics
        avg_imp = np.mean(all_improvements)
        median_imp = np.median(all_improvements)
        improved_pct = 100 * sum(1 for x in all_improvements if x > 1) / len(all_improvements)
        
        summary_text = f'Mean: {avg_imp:.2f}%\nMedian: {median_imp:.2f}%\nImproved: {improved_pct:.1f}%'
        ax4.text(0.98, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/jeong/ML_Project_accel-sim/ml_scheduler_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {output_path}")
    
    plt.show()
    
    return rodinia_data, deepbench_data

def print_summary(rodinia_data, deepbench_data):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if rodinia_data:
        rodinia_imps = [d['improvement'] for d in rodinia_data]
        print(f"\n📊 Rodinia ({len(rodinia_data)} benchmarks):")
        print(f"  Average improvement: {np.mean(rodinia_imps):+.2f}%")
        print(f"  Median improvement: {np.median(rodinia_imps):+.2f}%")
        print(f"  Best: {max(rodinia_imps):+.2f}%")
        print(f"  Worst: {min(rodinia_imps):+.2f}%")
        print(f"  Improved: {sum(1 for x in rodinia_imps if x > 0)}/{len(rodinia_imps)}")
    
    if deepbench_data:
        deepbench_imps = [d['improvement'] for d in deepbench_data]
        print(f"\n📊 Deepbench ({len(deepbench_data)} benchmarks):")
        print(f"  Average improvement: {np.mean(deepbench_imps):+.2f}%")
        print(f"  Median improvement: {np.median(deepbench_imps):+.2f}%")
        print(f"  Best: {max(deepbench_imps):+.2f}%")
        print(f"  Worst: {min(deepbench_imps):+.2f}%")
        print(f"  Improved: {sum(1 for x in deepbench_imps if x > 0)}/{len(deepbench_imps)}")

if __name__ == "__main__":
    print("="*80)
    print("ML SCHEDULER PERFORMANCE VISUALIZATION")
    print("="*80)
    
    rodinia_data, deepbench_data = create_comparison_plots()
    print_summary(rodinia_data, deepbench_data)
    
    print(f"\n✨ Visualization complete!")
