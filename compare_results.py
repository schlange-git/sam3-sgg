#!/usr/bin/env python
"""
Script to compare evaluation results from multiple experiments.
Usage: python compare_results.py <output_dir1> <output_dir2> [<output_dir3> ...]
"""

import os
import sys
import argparse
from save_eval_results import extract_results_from_log, extract_results_from_metrics, format_value


def print_comparison_table(results_dict):
    """Print a comparison table of results from multiple experiments."""
    
    experiments = list(results_dict.keys())
    
    print("\n" + "=" * 120)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 120)
    
    # Detection Results
    print("\nDetection Results (bbox):")
    print("-" * 120)
    
    # Header
    header = f"{'Metric':<15}"
    for exp in experiments:
        header += f"{exp:<25}"
    print(header)
    print("-" * 120)
    
    # Metrics
    bbox_metrics = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    for metric in bbox_metrics:
        row = f"{metric:<15}"
        for exp in experiments:
            val = results_dict[exp].get('bbox', {}).get(metric, 'N/A')
            row += f"{format_value(val):<25}"
        print(row)
    
    # Scene Graph Results
    print("\n" + "=" * 120)
    print("\nScene Graph Results:")
    print("-" * 120)
    
    # Header
    header = f"{'Metric':<20}"
    for exp in experiments:
        header += f"{exp:<25}"
    print(header)
    print("-" * 120)
    
    # R@K
    for k in [20, 50, 100]:
        row = f"R@{k:<17}"
        for exp in experiments:
            sg = results_dict[exp].get('SG', {})
            val = sg.get(f'R@{k}', sg.get(f'SGRecall@{k}', 'N/A'))
            row += f"{format_value(val):<25}"
        print(row)
    
    # ng-R@K
    for k in [20, 50, 100]:
        row = f"ng-R@{k:<15}"
        for exp in experiments:
            val = results_dict[exp].get('SG', {}).get(f'ng-R@{k}', 'N/A')
            row += f"{format_value(val):<25}"
        print(row)
    
    # zR@K
    for k in [20, 50, 100]:
        row = f"zR@{k:<17}"
        for exp in experiments:
            val = results_dict[exp].get('SG', {}).get(f'zR@{k}', 'N/A')
            row += f"{format_value(val):<25}"
        print(row)
    
    # mR@K
    for k in [20, 50, 100]:
        row = f"mR@{k:<17}"
        for exp in experiments:
            sg = results_dict[exp].get('SG', {})
            val = sg.get(f'mR@{k}', sg.get(f'SGMeanRecall@{k}', 'N/A'))
            row += f"{format_value(val):<25}"
        print(row)
    
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation results from multiple experiments')
    parser.add_argument('output_dirs', nargs='+', type=str, help='Output directories to compare')
    
    args = parser.parse_args()
    
    results_dict = {}
    
    for output_dir in args.output_dirs:
        if not os.path.exists(output_dir):
            print(f"Warning: Directory {output_dir} does not exist, skipping...")
            continue
        
        # Extract experiment name from directory
        exp_name = os.path.basename(output_dir.rstrip('/'))
        
        # Try to extract results
        metrics_path = os.path.join(output_dir, 'metrics.json')
        results = extract_results_from_metrics(metrics_path)
        
        if not results:
            log_path = os.path.join(output_dir, 'log.txt')
            results = extract_results_from_log(log_path)
        
        if results:
            results_dict[exp_name] = results
        else:
            print(f"Warning: Could not extract results from {output_dir}, skipping...")
    
    if not results_dict:
        print("Error: No valid results found in any directory")
        sys.exit(1)
    
    print_comparison_table(results_dict)
    
    # Save comparison to file
    output_file = "results_comparison.txt"
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print_comparison_table(results_dict)
        sys.stdout = original_stdout
    
    print(f"\n✓ Comparison saved to: {output_file}")


if __name__ == '__main__':
    main()
