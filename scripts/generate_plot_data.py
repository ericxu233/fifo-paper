#!/usr/bin/env python3
"""
Generate filtered CSV data for LaTeX plots from perf_results.csv.
Filters data based on default configuration and computes throughput.
"""

import pandas as pd
import os

# Configuration parameters
FIFO_SIZE = 8
STREAMING_LATENCY = 400
DATASET_SIZE = "MEDIUM_DATASET"
PB = 1

# Benchmark order for plotting
BENCHMARK_ORDER = [
    'ema', 'exp_decay', 'hp_filter', 'momentum_sgd', 'weighted_avg',
    'gaussian_cdf', 'gemm', 'mvt', 'sep', 'fd'
]

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'perf_results.csv')
    output_dir = os.path.join(project_dir, 'artifacts')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Filter for default configuration
    mask = (
        (df['PB'] == PB) &
        (df['fifo_size'] == FIFO_SIZE) &
        (df['streaming_latency'] == STREAMING_LATENCY) &
        (df['size'] == DATASET_SIZE)
    )
    filtered = df[mask].copy()
    
    # Compute throughput (iterations per cycle)
    filtered['throughput'] = filtered['trip_count'] / filtered['execution_cycles']
    
    # Create separate dataframes for each configuration
    # Baseline: LOOKAHEAD=0, FIFO=0
    baseline = filtered[(filtered['LOOKAHEAD'] == 0) & (filtered['FIFO'] == 0)].copy()
    baseline = baseline[['testname', 'throughput', 'execution_cycles', 'trip_count']]
    
    # FIFO Init: LOOKAHEAD=0, FIFO=1
    fifo_init = filtered[(filtered['LOOKAHEAD'] == 0) & (filtered['FIFO'] == 1)].copy()
    fifo_init = fifo_init[['testname', 'throughput', 'execution_cycles', 'trip_count']]
    
    # FIFO Init + Lookahead: LOOKAHEAD=1, FIFO=1
    lookahead = filtered[(filtered['LOOKAHEAD'] == 1) & (filtered['FIFO'] == 1)].copy()
    lookahead = lookahead[['testname', 'throughput', 'execution_cycles', 'trip_count']]
    
    # Sort by benchmark order
    baseline['order'] = baseline['testname'].apply(lambda x: BENCHMARK_ORDER.index(x) if x in BENCHMARK_ORDER else 999)
    fifo_init['order'] = fifo_init['testname'].apply(lambda x: BENCHMARK_ORDER.index(x) if x in BENCHMARK_ORDER else 999)
    lookahead['order'] = lookahead['testname'].apply(lambda x: BENCHMARK_ORDER.index(x) if x in BENCHMARK_ORDER else 999)
    
    baseline = baseline.sort_values('order').drop('order', axis=1)
    fifo_init = fifo_init.sort_values('order').drop('order', axis=1)
    lookahead = lookahead.sort_values('order').drop('order', axis=1)
    
    # Save to CSV files
    baseline.to_csv(os.path.join(output_dir, 'baseline_throughput.csv'), index=False)
    fifo_init.to_csv(os.path.join(output_dir, 'fifo_init_throughput.csv'), index=False)
    lookahead.to_csv(os.path.join(output_dir, 'lookahead_throughput.csv'), index=False)
    
    # Also create a combined file for easy pgfplots usage
    combined = baseline[['testname', 'throughput']].copy()
    combined = combined.rename(columns={'throughput': 'baseline'})
    combined = combined.merge(
        fifo_init[['testname', 'throughput']].rename(columns={'throughput': 'fifo_init'}),
        on='testname'
    )
    combined = combined.merge(
        lookahead[['testname', 'throughput']].rename(columns={'throughput': 'lookahead'}),
        on='testname'
    )
    
    # Compute speedups
    combined['fifo_speedup'] = combined['fifo_init'] / combined['baseline']
    combined['lookahead_speedup'] = combined['lookahead'] / combined['baseline']
    
    combined.to_csv(os.path.join(output_dir, 'throughput_comparison.csv'), index=False)
    
    print(f"Generated plot data in {output_dir}/")
    print(f"  - baseline_throughput.csv")
    print(f"  - fifo_init_throughput.csv") 
    print(f"  - lookahead_throughput.csv")
    print(f"  - throughput_comparison.csv")
    
    # Print summary
    print("\nThroughput Summary:")
    print(combined.to_string(index=False))

if __name__ == '__main__':
    main()
