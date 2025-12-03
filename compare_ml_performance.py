#!/usr/bin/env python3
"""
ML Scheduler Performance Comparison Script
Compares baseline (LRR/GTO) vs ML scheduler results
"""

import os
import re
import glob
from collections import defaultdict

def extract_ipc_from_output(filepath):
    """Extract gpu_tot_ipc from GPGPU-Sim output file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(r'gpu_tot_ipc\s*=\s*([\d.]+)', content)
            if match:
                return float(match.group(1))
    except:
        pass
    return None

def find_results(base_dir, benchmark_type):
    """Find all result files for a benchmark type"""
    results = {}
    
    # Pattern: benchmark/workload/RTX3070-SASS/*.o* files
    pattern = f"{base_dir}/{benchmark_type}/*/RTX3070-SASS/*.o*"
    
    for filepath in glob.glob(pattern):
        # Extract workload name
        parts = filepath.split('/')
        workload = parts[-3]  # e.g., "train_half_7_7_832_16_128_5_5_2_2_1_1"
        
        ipc = extract_ipc_from_output(filepath)
        if ipc is not None:
            results[workload] = ipc
    
    return results

def compare_results(baseline_dir, ml_dir, benchmark_types):
    """Compare baseline vs ML scheduler results"""
    
    print("=" * 100)
    print("ML SCHEDULER PERFORMANCE COMPARISON")
    print("=" * 100)
    print()
    
    all_improvements = []
    
    for bench_type in benchmark_types:
        print(f"\n{'='*100}")
        print(f"Benchmark Suite: {bench_type.upper()}")
        print(f"{'='*100}")
        
        baseline_results = find_results(baseline_dir, bench_type)
        ml_results = find_results(ml_dir, bench_type)
        
        # Find common workloads
        common = set(baseline_results.keys()) & set(ml_results.keys())
        
        if not common:
            print(f"⚠️  No common results found for {bench_type}")
            continue
        
        print(f"\n{'Workload':<60} {'Baseline IPC':>12} {'ML IPC':>12} {'Improvement':>12}")
        print("-" * 100)
        
        improvements = []
        
        for workload in sorted(common):
            baseline_ipc = baseline_results[workload]
            ml_ipc = ml_results[workload]
            
            improvement = ((ml_ipc - baseline_ipc) / baseline_ipc) * 100
            improvements.append(improvement)
            all_improvements.append(improvement)
            
            # Color coding
            if improvement > 5:
                symbol = "🟢"
            elif improvement > 0:
                symbol = "🟡"
            elif improvement > -5:
                symbol = "🟠"
            else:
                symbol = "🔴"
            
            # Truncate workload name if too long
            display_name = workload if len(workload) < 58 else workload[:55] + "..."
            
            print(f"{display_name:<60} {baseline_ipc:>12.4f} {ml_ipc:>12.4f} {symbol} {improvement:>10.2f}%")
        
        # Summary statistics for this benchmark
        print("-" * 100)
        print(f"{'SUMMARY':<60} {'':>12} {'':>12} {'':>12}")
        print(f"{'  Total workloads':<60} {'':>12} {len(common):>12} {'':>12}")
        print(f"{'  Average improvement':<60} {'':>12} {'':>12} {sum(improvements)/len(improvements):>10.2f}%")
        print(f"{'  Best improvement':<60} {'':>12} {'':>12} {max(improvements):>10.2f}%")
        print(f"{'  Worst degradation':<60} {'':>12} {'':>12} {min(improvements):>10.2f}%")
        print(f"{'  Improved workloads':<60} {'':>12} {sum(1 for x in improvements if x > 0):>12} {'':>12}")
        print(f"{'  Degraded workloads':<60} {'':>12} {sum(1 for x in improvements if x < 0):>12} {'':>12}")
    
    # Overall summary
    if all_improvements:
        print(f"\n{'='*100}")
        print("OVERALL SUMMARY")
        print(f"{'='*100}")
        print(f"Total workloads analyzed: {len(all_improvements)}")
        print(f"Average improvement: {sum(all_improvements)/len(all_improvements):.2f}%")
        print(f"Best improvement: {max(all_improvements):.2f}%")
        print(f"Worst degradation: {min(all_improvements):.2f}%")
        print(f"Workloads improved: {sum(1 for x in all_improvements if x > 0)} ({100*sum(1 for x in all_improvements if x > 0)/len(all_improvements):.1f}%)")
        print(f"Workloads degraded: {sum(1 for x in all_improvements if x < 0)} ({100*sum(1 for x in all_improvements if x < 0)/len(all_improvements):.1f}%)")
        
        # Distribution
        print("\nPerformance Distribution:")
        bins = [(-float('inf'), -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, float('inf'))]
        bin_labels = ["<-10%", "-10% to -5%", "-5% to 0%", "0% to 5%", "5% to 10%", ">10%"]
        
        for (low, high), label in zip(bins, bin_labels):
            count = sum(1 for x in all_improvements if low < x <= high)
            percentage = 100 * count / len(all_improvements)
            bar = "█" * int(percentage / 2)
            print(f"  {label:>12}: {bar:<50} {count:>3} ({percentage:>5.1f}%)")

if __name__ == "__main__":
    import sys
    
    # Default paths
    baseline_dir = "/home/HDD/jeong/accel_sim_traces/sim_run_12.8"
    ml_dir = "/home/HDD/jeong/accel_sim_traces/sim_run_12.8"
    
    # Benchmark types
    benchmark_types = [
        "backprop-rodinia-2.0-ft",
        "bfs-rodinia-2.0-ft",
        "hotspot-rodinia-2.0-ft",
        "heartwall-rodinia-2.0-ft",
        "lud-rodinia-2.0-ft",
        "nw-rodinia-2.0-ft",
        "nn-rodinia-2.0-ft",
        "pathfinder-rodinia-2.0-ft",
        "srad_v2-rodinia-2.0-ft",
        "streamcluster-rodinia-2.0-ft",
        "conv_bench",
        "gemm_bench",
        "rnn_bench"
    ]
    
    # Allow custom paths via command line
    if len(sys.argv) > 1:
        baseline_dir = sys.argv[1]
    if len(sys.argv) > 2:
        ml_dir = sys.argv[2]
    
    print(f"Baseline directory: {baseline_dir}")
    print(f"ML directory: {ml_dir}")
    print()
    
    compare_results(baseline_dir, ml_dir, benchmark_types)
