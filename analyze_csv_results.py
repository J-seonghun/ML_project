#!/usr/bin/env python3
"""
ML Scheduler Performance Comparison from CSV Results
Compares Baseline vs ML Scheduler across Rodinia and Deepbench
"""

import pandas as pd
import sys

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
            print(f"Warning: No IPC data found in {filepath}")
            return results
        
        # Parse IPC data (format: benchmark,value or APPS,config followed by data)
        for line in lines[ipc_start+2:]:  # Skip header rows
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

def compare_results(baseline_file, ml_file, suite_name):
    """Compare baseline vs ML results for a benchmark suite"""
    
    print(f"\n{'='*100}")
    print(f"Benchmark Suite: {suite_name}")
    print(f"{'='*100}")
    
    baseline = load_csv_results(baseline_file)
    ml = load_csv_results(ml_file)
    
    if not baseline or not ml:
        print(f"⚠️  Could not load data for {suite_name}")
        return []
    
    # Find common benchmarks
    common = set(baseline.keys()) & set(ml.keys())
    
    if not common:
        print(f"⚠️  No common benchmarks found")
        return []
    
    print(f"\n{'Benchmark':<60} {'Baseline IPC':>12} {'ML IPC':>12} {'Improvement':>12}")
    print("-" * 100)
    
    improvements = []
    
    for bench in sorted(common):
        baseline_ipc = baseline[bench]
        ml_ipc = ml[bench]
        
        if baseline_ipc == 0:
            continue
        
        improvement = ((ml_ipc - baseline_ipc) / baseline_ipc) * 100
        improvements.append((bench, baseline_ipc, ml_ipc, improvement))
        
        # Color coding
        if improvement > 10:
            symbol = "🟢"
        elif improvement > 1:
            symbol = "🟡"
        elif improvement > -1:
            symbol = "⚪"
        elif improvement > -10:
            symbol = "🟠"
        else:
            symbol = "🔴"
        
        # Truncate benchmark name
        display_name = bench if len(bench) < 58 else bench[:55] + "..."
        
        print(f"{display_name:<60} {baseline_ipc:>12.4f} {ml_ipc:>12.4f} {symbol} {improvement:>10.2f}%")
    
    # Summary
    if improvements:
        imp_values = [x[3] for x in improvements]
        print("-" * 100)
        print(f"{'SUMMARY':<60}")
        print(f"  Total benchmarks: {len(improvements)}")
        print(f"  Average improvement: {sum(imp_values)/len(imp_values):.2f}%")
        print(f"  Best improvement: {max(imp_values):.2f}% ({[x[0] for x in improvements if x[3] == max(imp_values)][0][:40]})")
        print(f"  Worst degradation: {min(imp_values):.2f}% ({[x[0] for x in improvements if x[3] == min(imp_values)][0][:40]})")
        print(f"  Improved: {sum(1 for x in imp_values if x > 0)} ({100*sum(1 for x in imp_values if x > 0)/len(imp_values):.1f}%)")
        print(f"  Degraded: {sum(1 for x in imp_values if x < 0)} ({100*sum(1 for x in imp_values if x < 0)/len(imp_values):.1f}%)")
    
    return improvements

def main():
    base_dir = "/home/HDD/jeong/accel_sim_traces"
    
    # File mapping
    files = {
        'rodinia_baseline': f"{base_dir}/accel-Ampere_251118.csv",
        'rodinia_ml': f"{base_dir}/ML_rodinia_ampere_251124.csv",
        'deepbench_baseline': f"{base_dir}/deepbench_ampere_251124.csv",
        'deepbench_ml': f"{base_dir}/ML_deepbench_ampere_251125.csv"
    }
    
    print("=" * 100)
    print("ML SCHEDULER PERFORMANCE ANALYSIS")
    print("=" * 100)
    print()
    print(f"Baseline Rodinia: {files['rodinia_baseline']}")
    print(f"ML Rodinia:       {files['rodinia_ml']}")
    print(f"Baseline Deepbench: {files['deepbench_baseline']}")
    print(f"ML Deepbench:       {files['deepbench_ml']}")
    
    # Compare Rodinia
    rodinia_results = compare_results(
        files['rodinia_baseline'],
        files['rodinia_ml'],
        "RODINIA 2.0 (Training Set)"
    )
    
    # Compare Deepbench
    deepbench_results = compare_results(
        files['deepbench_baseline'],
        files['deepbench_ml'],
        "DEEPBENCH (Generalization Test)"
    )
    
    # Overall analysis
    all_results = rodinia_results + deepbench_results
    
    if all_results:
        all_improvements = [x[3] for x in all_results]
        
        print(f"\n{'='*100}")
        print("OVERALL ANALYSIS")
        print(f"{'='*100}")
        print(f"\nTotal benchmarks: {len(all_improvements)}")
        print(f"Overall average improvement: {sum(all_improvements)/len(all_improvements):.2f}%")
        print(f"Best case: {max(all_improvements):.2f}%")
        print(f"Worst case: {min(all_improvements):.2f}%")
        
        # Distribution
        print("\n📊 Performance Distribution:")
        bins = [
            (float('-inf'), -10, "🔴 Severe Degradation (<-10%)"),
            (-10, -5, "🟠 Moderate Degradation (-10% to -5%)"),
            (-5, -1, "🟡 Minor Degradation (-5% to -1%)"),
            (-1, 1, "⚪ Neutral (-1% to +1%)"),
            (1, 5, "🟢 Minor Improvement (+1% to +5%)"),
            (5, 10, "🟢 Moderate Improvement (+5% to +10%)"),
            (10, float('inf'), "🟢 Significant Improvement (>+10%)")
        ]
        
        for low, high, label in bins:
            count = sum(1 for x in all_improvements if low < x <= high)
            percentage = 100 * count / len(all_improvements)
            bar = "█" * int(percentage / 2)
            print(f"  {label:<40}: {bar:<50} {count:>3} ({percentage:>5.1f}%)")
        
        # Key insights
        print(f"\n{'='*100}")
        print("KEY INSIGHTS")
        print(f"{'='*100}")
        
        rodinia_avg = sum([x[3] for x in rodinia_results])/len(rodinia_results) if rodinia_results else 0
        deepbench_avg = sum([x[3] for x in deepbench_results])/len(deepbench_results) if deepbench_results else 0
        
        print(f"\n1. Training vs Generalization:")
        print(f"   - Rodinia (training set) average: {rodinia_avg:+.2f}%")
        print(f"   - Deepbench (test set) average: {deepbench_avg:+.2f}%")
        
        if deepbench_avg < rodinia_avg - 2:
            print(f"   → ⚠️  Overfitting detected: Model performs worse on unseen workloads")
        elif abs(deepbench_avg - rodinia_avg) < 2:
            print(f"   → ✅ Good generalization: Similar performance on training and test sets")
        else:
            print(f"   → ✨ Positive transfer: Model generalizes better than expected")
        
        print(f"\n2. Success Rate:")
        improved = sum(1 for x in all_improvements if x > 1)
        neutral = sum(1 for x in all_improvements if -1 <= x <= 1)
        degraded = sum(1 for x in all_improvements if x < -1)
        print(f"   - Improved: {improved}/{len(all_improvements)} ({100*improved/len(all_improvements):.1f}%)")
        print(f"   - Neutral: {neutral}/{len(all_improvements)} ({100*neutral/len(all_improvements):.1f}%)")
        print(f"   - Degraded: {degraded}/{len(all_improvements)} ({100*degraded/len(all_improvements):.1f}%)")
        
        print(f"\n3. Extreme Cases:")
        # Find best and worst
        best = max(all_results, key=lambda x: x[3])
        worst = min(all_results, key=lambda x: x[3])
        print(f"   Best: {best[0][:60]}")
        print(f"         {best[1]:.4f} → {best[2]:.4f} IPC ({best[3]:+.2f}%)")
        print(f"   Worst: {worst[0][:60]}")
        print(f"         {worst[1]:.4f} → {worst[2]:.4f} IPC ({worst[3]:+.2f}%)")

if __name__ == "__main__":
    main()
