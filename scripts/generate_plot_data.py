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
    
    # Generate FIFO size impact data
    generate_fifo_size_impact(df, output_dir)
    
    # Generate streaming latency impact data
    generate_streaming_latency_impact(df, output_dir)


def generate_fifo_size_impact(df, output_dir):
    """Generate data showing speedup vs FIFO size for selected benchmarks."""
    
    # Fixed parameters (vary only fifo_size)
    streaming_latency = 400
    dataset_size = "MEDIUM_DATASET"
    pb = 1
    
    # All benchmarks to include
    benchmarks = ['ema', 'exp_decay', 'hp_filter', 'momentum_sgd', 'weighted_avg',
                  'gaussian_cdf', 'gemm', 'mvt', 'sep', 'fd']
    
    # FIFO sizes to include (sorted)
    fifo_sizes = sorted(df['fifo_size'].unique())
    
    results = []
    
    for bench in benchmarks:
        for fifo_size in fifo_sizes:
            # Filter for this configuration
            mask = (
                (df['testname'] == bench) &
                (df['PB'] == pb) &
                (df['fifo_size'] == fifo_size) &
                (df['streaming_latency'] == streaming_latency) &
                (df['size'] == dataset_size)
            )
            filtered = df[mask]
            
            # Get baseline (LOOKAHEAD=0, FIFO=0)
            baseline = filtered[(filtered['LOOKAHEAD'] == 0) & (filtered['FIFO'] == 0)]
            
            # Get lookahead (LOOKAHEAD=1, FIFO=1)
            lookahead = filtered[(filtered['LOOKAHEAD'] == 1) & (filtered['FIFO'] == 1)]
            
            if len(baseline) > 0 and len(lookahead) > 0:
                baseline_cycles = baseline['execution_cycles'].values[0]
                lookahead_cycles = lookahead['execution_cycles'].values[0]
                speedup = baseline_cycles / lookahead_cycles
                
                results.append({
                    'fifo_size': fifo_size,
                    'benchmark': bench,
                    'speedup': speedup
                })
    
    # Convert to dataframe and pivot for pgfplots
    results_df = pd.DataFrame(results)
    
    # Pivot so each benchmark is a column
    pivoted = results_df.pivot(index='fifo_size', columns='benchmark', values='speedup')
    pivoted = pivoted.reset_index()
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'fifo_size_impact.csv')
    pivoted.to_csv(output_file, index=False)
    
    print(f"\nGenerated FIFO size impact data:")
    print(f"  - fifo_size_impact.csv")
    print(pivoted.to_string(index=False))

def generate_streaming_latency_impact(df, output_dir):
    """Generate data showing speedup vs streaming latency for selected benchmarks."""
    
    # Fixed parameters (vary only streaming_latency)
    fifo_size = 8
    dataset_size = "MEDIUM_DATASET"
    pb = 1
    
    # All benchmarks to include
    benchmarks = ['ema', 'exp_decay', 'hp_filter', 'momentum_sgd', 'weighted_avg',
                  'gaussian_cdf', 'gemm', 'mvt', 'sep', 'fd']
    
    # Streaming latencies to include (sorted)
    streaming_latencies = sorted(df['streaming_latency'].unique())
    
    results = []
    
    for bench in benchmarks:
        for streaming_latency in streaming_latencies:
            # Filter for this configuration
            mask = (
                (df['testname'] == bench) &
                (df['PB'] == pb) &
                (df['fifo_size'] == fifo_size) &
                (df['streaming_latency'] == streaming_latency) &
                (df['size'] == dataset_size)
            )
            filtered = df[mask]
            
            # Get baseline (LOOKAHEAD=0, FIFO=0)
            baseline = filtered[(filtered['LOOKAHEAD'] == 0) & (filtered['FIFO'] == 0)]
            
            # Get lookahead (LOOKAHEAD=1, FIFO=1)
            lookahead = filtered[(filtered['LOOKAHEAD'] == 1) & (filtered['FIFO'] == 1)]
            
            if len(baseline) > 0 and len(lookahead) > 0:
                baseline_cycles = baseline['execution_cycles'].values[0]
                lookahead_cycles = lookahead['execution_cycles'].values[0]
                speedup = baseline_cycles / lookahead_cycles
                
                results.append({
                    'streaming_latency': streaming_latency,
                    'benchmark': bench,
                    'speedup': speedup
                })
    
    # Convert to dataframe and pivot for pgfplots
    results_df = pd.DataFrame(results)
    
    # Pivot so each benchmark is a column
    pivoted = results_df.pivot(index='streaming_latency', columns='benchmark', values='speedup')
    pivoted = pivoted.reset_index()
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'streaming_latency_impact.csv')
    pivoted.to_csv(output_file, index=False)
    
    print(f"\nGenerated streaming latency impact data:")
    print(f"  - streaming_latency_impact.csv")
    print(pivoted.to_string(index=False))


def generate_unroll_comparison(output_dir):
    """Generate data comparing baseline, unrolled, lookahead, and unrolled+lookahead."""
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'unroll_results.csv')
    
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Fixed parameters
    fifo_size = 8
    dataset_size = "MEDIUM_DATASET"
    
    # Benchmarks available in unroll_results
    benchmarks = df['testname'].unique().tolist()
    
    results = []
    
    for bench in benchmarks:
        bench_data = df[(df['testname'] == bench) & 
                        (df['fifo_size'] == fifo_size) & 
                        (df['size'] == dataset_size)]
        
        # 1. Baseline: LOOKAHEAD=0, FIFO=0, PB=0, unroll=none
        baseline = bench_data[(bench_data['LOOKAHEAD'] == 0) & 
                              (bench_data['FIFO'] == 0) & 
                              (bench_data['PB'] == 0) & 
                              (bench_data['unroll'] == 'none')]
        
        # 2. Max unrolled without lookahead: FIFO=0, LOOKAHEAD=0, PB=1, max unroll with dfg_size < 64
        unrolled_no_la = bench_data[(bench_data['LOOKAHEAD'] == 0) & 
                                     (bench_data['FIFO'] == 0) & 
                                     (bench_data['PB'] == 0) & 
                                     (bench_data['dfg_size'] < 64) &
                                     (bench_data['unroll'] != 'none')]
        if len(unrolled_no_la) > 0:
            # Get the one with largest dfg_size (most unrolling)
            unrolled_no_la = unrolled_no_la.loc[unrolled_no_la['dfg_size'].idxmax()]
        
        # 3. Lookahead without unrolling: LOOKAHEAD=1, FIFO=1, PB=1, unroll=none
        lookahead = bench_data[(bench_data['LOOKAHEAD'] == 1) & 
                               (bench_data['FIFO'] == 1) & 
                               (bench_data['PB'] == 1) & 
                               (bench_data['unroll'] == 'none')]
        
        # 4. Max unrolled with lookahead: FIFO=1, LOOKAHEAD=1, PB=1, max unroll with dfg_size < 64
        unrolled_la = bench_data[(bench_data['LOOKAHEAD'] == 1) & 
                                  (bench_data['FIFO'] == 1) & 
                                  (bench_data['PB'] == 1) & 
                                  (bench_data['dfg_size'] < 64) &
                                  (bench_data['unroll'] != 'none')]
        if len(unrolled_la) > 0:
            # Get the one with largest dfg_size (most unrolling)
            unrolled_la = unrolled_la.loc[unrolled_la['dfg_size'].idxmax()]
        
        trip_count = 10000  # From unroll_results.csv
        
        result = {'testname': bench}
        
        if len(baseline) > 0:
            result['baseline'] = trip_count / baseline['execution_cycles'].values[0]
        
        if isinstance(unrolled_no_la, pd.Series) and len(unrolled_no_la) > 0:
            result['unrolled'] = trip_count / unrolled_no_la['execution_cycles']
            result['unrolled_factor'] = unrolled_no_la['unroll']
        elif isinstance(unrolled_no_la, pd.DataFrame) and len(unrolled_no_la) > 0:
            result['unrolled'] = trip_count / unrolled_no_la['execution_cycles'].values[0]
            result['unrolled_factor'] = unrolled_no_la['unroll'].values[0]
        
        if len(lookahead) > 0:
            result['lookahead'] = trip_count / lookahead['execution_cycles'].values[0]
        
        if isinstance(unrolled_la, pd.Series) and len(unrolled_la) > 0:
            result['unrolled_lookahead'] = trip_count / unrolled_la['execution_cycles']
            result['unrolled_la_factor'] = unrolled_la['unroll']
        elif isinstance(unrolled_la, pd.DataFrame) and len(unrolled_la) > 0:
            result['unrolled_lookahead'] = trip_count / unrolled_la['execution_cycles'].values[0]
            result['unrolled_la_factor'] = unrolled_la['unroll'].values[0]
        
        results.append(result)
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Define benchmark order
    benchmark_order = ['ema', 'exp_decay', 'hp_filter', 'momentum_sgd', 'weighted_avg',
                       'gaussian_cdf', 'gemm', 'mvt', 'sep']
    results_df['order'] = results_df['testname'].apply(
        lambda x: benchmark_order.index(x) if x in benchmark_order else 999
    )
    results_df = results_df.sort_values('order').drop('order', axis=1)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'unroll_comparison.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\nGenerated unroll comparison data:")
    print(f"  - unroll_comparison.csv")
    print(results_df.to_string(index=False))


def generate_unroll_per_pe_comparison(output_dir):
    """Generate data comparing throughput per PE for baseline, unrolled, lookahead, and unrolled+lookahead."""
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join(project_dir, 'data', 'unroll_results.csv')
    
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Fixed parameters
    fifo_size = 8
    dataset_size = "MEDIUM_DATASET"
    
    # Benchmarks available in unroll_results
    benchmarks = df['testname'].unique().tolist()
    
    results = []
    
    for bench in benchmarks:
        bench_data = df[(df['testname'] == bench) & 
                        (df['fifo_size'] == fifo_size) & 
                        (df['size'] == dataset_size)]
        
        # 1. Baseline: LOOKAHEAD=0, FIFO=0, PB=0, unroll=none
        baseline = bench_data[(bench_data['LOOKAHEAD'] == 0) & 
                              (bench_data['FIFO'] == 0) & 
                              (bench_data['PB'] == 0) & 
                              (bench_data['unroll'] == 'none')]
        
        # 2. Max unrolled without lookahead: FIFO=0, LOOKAHEAD=0, PB=0, max unroll with dfg_size < 64
        unrolled_no_la = bench_data[(bench_data['LOOKAHEAD'] == 0) & 
                                     (bench_data['FIFO'] == 0) & 
                                     (bench_data['PB'] == 0) & 
                                     (bench_data['dfg_size'] < 64) &
                                     (bench_data['unroll'] != 'none')]
        if len(unrolled_no_la) > 0:
            # Get the one with largest dfg_size (most unrolling)
            unrolled_no_la = unrolled_no_la.loc[unrolled_no_la['dfg_size'].idxmax()]
        
        # 3. Lookahead without unrolling: LOOKAHEAD=1, FIFO=1, PB=1, unroll=none
        lookahead = bench_data[(bench_data['LOOKAHEAD'] == 1) & 
                               (bench_data['FIFO'] == 1) & 
                               (bench_data['PB'] == 1) & 
                               (bench_data['unroll'] == 'none')]
        
        # 4. Max unrolled with lookahead: FIFO=1, LOOKAHEAD=1, PB=1, max unroll with dfg_size < 64
        unrolled_la = bench_data[(bench_data['LOOKAHEAD'] == 1) & 
                                  (bench_data['FIFO'] == 1) & 
                                  (bench_data['PB'] == 1) & 
                                  (bench_data['dfg_size'] < 64) &
                                  (bench_data['unroll'] != 'none')]
        if len(unrolled_la) > 0:
            # Get the one with largest dfg_size (most unrolling)
            unrolled_la = unrolled_la.loc[unrolled_la['dfg_size'].idxmax()]
        
        trip_count = 10000  # From unroll_results.csv
        
        result = {'testname': bench}
        
        # Compute throughput per PE (dfg_size represents number of PEs)
        if len(baseline) > 0:
            throughput = trip_count / baseline['execution_cycles'].values[0]
            num_pes = baseline['dfg_size'].values[0]
            result['baseline'] = throughput / num_pes
            result['baseline_pes'] = num_pes
        
        if isinstance(unrolled_no_la, pd.Series) and len(unrolled_no_la) > 0:
            throughput = trip_count / unrolled_no_la['execution_cycles']
            num_pes = unrolled_no_la['dfg_size']
            result['unrolled'] = throughput / num_pes
            result['unrolled_pes'] = num_pes
            result['unrolled_factor'] = unrolled_no_la['unroll']
        elif isinstance(unrolled_no_la, pd.DataFrame) and len(unrolled_no_la) > 0:
            throughput = trip_count / unrolled_no_la['execution_cycles'].values[0]
            num_pes = unrolled_no_la['dfg_size'].values[0]
            result['unrolled'] = throughput / num_pes
            result['unrolled_pes'] = num_pes
            result['unrolled_factor'] = unrolled_no_la['unroll'].values[0]
        
        if len(lookahead) > 0:
            throughput = trip_count / lookahead['execution_cycles'].values[0]
            num_pes = lookahead['dfg_size'].values[0]
            result['lookahead'] = throughput / num_pes
            result['lookahead_pes'] = num_pes
        
        if isinstance(unrolled_la, pd.Series) and len(unrolled_la) > 0:
            throughput = trip_count / unrolled_la['execution_cycles']
            num_pes = unrolled_la['dfg_size']
            result['unrolled_lookahead'] = throughput / num_pes
            result['unrolled_lookahead_pes'] = num_pes
            result['unrolled_la_factor'] = unrolled_la['unroll']
        elif isinstance(unrolled_la, pd.DataFrame) and len(unrolled_la) > 0:
            throughput = trip_count / unrolled_la['execution_cycles'].values[0]
            num_pes = unrolled_la['dfg_size'].values[0]
            result['unrolled_lookahead'] = throughput / num_pes
            result['unrolled_lookahead_pes'] = num_pes
            result['unrolled_la_factor'] = unrolled_la['unroll'].values[0]
        
        results.append(result)
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Define benchmark order
    benchmark_order = ['ema', 'exp_decay', 'hp_filter', 'momentum_sgd', 'weighted_avg',
                       'gaussian_cdf', 'gemm', 'mvt', 'sep']
    results_df['order'] = results_df['testname'].apply(
        lambda x: benchmark_order.index(x) if x in benchmark_order else 999
    )
    results_df = results_df.sort_values('order').drop('order', axis=1)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'unroll_per_pe_comparison.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\nGenerated unroll per-PE comparison data:")
    print(f"  - unroll_per_pe_comparison.csv")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
    
    # Also generate unroll comparison
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(project_dir, 'artifacts')
    generate_unroll_comparison(output_dir)
    generate_unroll_per_pe_comparison(output_dir)
