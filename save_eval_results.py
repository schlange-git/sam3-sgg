#!/usr/bin/env python
"""
Script to extract and save evaluation results to a formatted text file.
Usage: python save_eval_results.py <output_dir> [--name <result_name>]
"""

import os
import sys
import json
import argparse
from collections import OrderedDict
from datetime import datetime


def extract_results_from_metrics(metrics_path):
    """Extract key metrics from metrics.json file."""
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        # Read the last line which contains the final results
        lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('OrderedDict'):
                # Parse the OrderedDict string
                try:
                    # Use eval with OrderedDict in namespace (safe for this specific format)
                    results = eval(line, {"OrderedDict": OrderedDict, "nan": float('nan')})
                    return results
                except:
                    continue
    return None


def extract_results_from_log(log_path):
    """Extract key metrics from log.txt file."""
    if not os.path.exists(log_path):
        return None
    
    results = {'bbox': {}, 'SG': {}}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Extract bbox results
        if 'copypaste: AP,AP50,AP75,APs,APm,APl' in line:
            if i + 1 < len(lines):
                values = lines[i + 1].strip().split(':')[-1].split(',')
                if len(values) >= 6:
                    results['bbox']['AP'] = float(values[0])
                    results['bbox']['AP50'] = float(values[1])
                    results['bbox']['AP75'] = float(values[2])
                    results['bbox']['APs'] = float(values[3])
                    results['bbox']['APm'] = float(values[4])
                    results['bbox']['APl'] = float(values[5])
        
        # Extract SG results
        if 'copypaste: SGMeanRecall@20,SGMeanRecall@50,SGMeanRecall@100,SGRecall@20,SGRecall@50,SGRecall@100' in line:
            if i + 1 < len(lines):
                values = lines[i + 1].strip().split(':')[-1].split(',')
                if len(values) >= 6:
                    try:
                        results['SG']['SGMeanRecall@20'] = float(values[0])
                        results['SG']['SGMeanRecall@50'] = float(values[1])
                        results['SG']['SGMeanRecall@100'] = float(values[2])
                        results['SG']['SGRecall@20'] = float(values[3]) if values[3] != 'nan' else float('nan')
                        results['SG']['SGRecall@50'] = float(values[4]) if values[4] != 'nan' else float('nan')
                        results['SG']['SGRecall@100'] = float(values[5]) if values[5] != 'nan' else float('nan')
                    except:
                        pass
        
        # Extract detailed SG eval results
        if 'SGG eval:     R @' in line:
            parts = line.split(';')
            for part in parts:
                if 'R @ 20:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['R@20'] = float(val)
                    except:
                        results['SG']['R@20'] = float('nan')
                elif 'R @ 50:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['R@50'] = float(val)
                    except:
                        results['SG']['R@50'] = float('nan')
                elif 'R @ 100:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['R@100'] = float(val)
                    except:
                        results['SG']['R@100'] = float('nan')
        
        if 'SGG eval:  ng-R @' in line:
            parts = line.split(';')
            for part in parts:
                if 'ng-R @ 20:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['ng-R@20'] = float(val)
                    except:
                        results['SG']['ng-R@20'] = float('nan')
                elif 'ng-R @ 50:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['ng-R@50'] = float(val)
                    except:
                        results['SG']['ng-R@50'] = float('nan')
                elif 'ng-R @ 100:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['ng-R@100'] = float(val)
                    except:
                        results['SG']['ng-R@100'] = float('nan')
        
        if 'SGG eval:    zR @' in line:
            parts = line.split(';')
            for part in parts:
                if 'zR @ 20:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['zR@20'] = float(val)
                    except:
                        results['SG']['zR@20'] = float('nan')
                elif 'zR @ 50:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['zR@50'] = float(val)
                    except:
                        results['SG']['zR@50'] = float('nan')
                elif 'zR @ 100:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['zR@100'] = float(val)
                    except:
                        results['SG']['zR@100'] = float('nan')
        
        if 'SGG eval:    mR @' in line:
            parts = line.split(';')
            for part in parts:
                if 'mR @ 20:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['mR@20'] = float(val)
                    except:
                        results['SG']['mR@20'] = float('nan')
                elif 'mR @ 50:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['mR@50'] = float(val)
                    except:
                        results['SG']['mR@50'] = float('nan')
                elif 'mR @ 100:' in part:
                    val = part.split(':')[-1].strip()
                    try:
                        results['SG']['mR@100'] = float(val)
                    except:
                        results['SG']['mR@100'] = float('nan')
    
    return results if results['bbox'] or results['SG'] else None


def format_value(val):
    """Format a numeric value for display."""
    if isinstance(val, float):
        if val != val:  # Check for nan
            return "nan"
        return f"{val:.2f}"
    return str(val)


def save_results_to_txt(results, output_path, experiment_name="Evaluation"):
    """Save results to a formatted text file."""
    with open(output_path, 'w') as f:
        # Header
        f.write(f"{'=' * 60}\n")
        f.write(f"{experiment_name} Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        
        # Detection Results
        if 'bbox' in results and results['bbox']:
            f.write("Detection Results (bbox):\n")
            f.write("-" * 60 + "\n")
            bbox = results['bbox']
            if 'AP' in bbox:
                f.write(f"  AP     = {format_value(bbox.get('AP', 'N/A'))}\n")
            if 'AP50' in bbox:
                f.write(f"  AP50   = {format_value(bbox.get('AP50', 'N/A'))}\n")
            if 'AP75' in bbox:
                f.write(f"  AP75   = {format_value(bbox.get('AP75', 'N/A'))}\n")
            if 'APs' in bbox:
                f.write(f"  APs    = {format_value(bbox.get('APs', 'N/A'))}\n")
            if 'APm' in bbox:
                f.write(f"  APm    = {format_value(bbox.get('APm', 'N/A'))}\n")
            if 'APl' in bbox:
                f.write(f"  APl    = {format_value(bbox.get('APl', 'N/A'))}\n")
            f.write("\n")
        
        # Scene Graph Results
        if 'SG' in results and results['SG']:
            f.write("Scene Graph Results:\n")
            f.write("-" * 60 + "\n")
            sg = results['SG']
            
            # Recall
            r20 = format_value(sg.get('R@20', sg.get('SGRecall@20', 'N/A')))
            r50 = format_value(sg.get('R@50', sg.get('SGRecall@50', 'N/A')))
            r100 = format_value(sg.get('R@100', sg.get('SGRecall@100', 'N/A')))
            f.write(f"  R@20/50/100      = {r20}/{r50}/{r100}\n")
            
            # No Graph Constraint Recall
            ngr20 = format_value(sg.get('ng-R@20', 'N/A'))
            ngr50 = format_value(sg.get('ng-R@50', 'N/A'))
            ngr100 = format_value(sg.get('ng-R@100', 'N/A'))
            f.write(f"  ng-R@20/50/100   = {ngr20}/{ngr50}/{ngr100}\n")
            
            # Zero-shot Recall
            zr20 = format_value(sg.get('zR@20', 'N/A'))
            zr50 = format_value(sg.get('zR@50', 'N/A'))
            zr100 = format_value(sg.get('zR@100', 'N/A'))
            f.write(f"  zR@20/50/100     = {zr20}/{zr50}/{zr100}\n")
            
            # Mean Recall
            mr20 = format_value(sg.get('mR@20', sg.get('SGMeanRecall@20', 'N/A')))
            mr50 = format_value(sg.get('mR@50', sg.get('SGMeanRecall@50', 'N/A')))
            mr100 = format_value(sg.get('mR@100', sg.get('SGMeanRecall@100', 'N/A')))
            f.write(f"  mR@20/50/100     = {mr20}/{mr50}/{mr100}\n")
            f.write("\n")
        
        # Footer
        f.write("=" * 60 + "\n")
        f.write(f"Full details saved in metrics.json and log.txt\n")
        f.write("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Extract and save evaluation results')
    parser.add_argument('output_dir', type=str, help='Output directory containing metrics.json and log.txt')
    parser.add_argument('--name', type=str, default='Evaluation', help='Experiment name for the results')
    parser.add_argument('--output', type=str, default='eval_results.txt', help='Output filename')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)
    
    # Try to extract from metrics.json first
    metrics_path = os.path.join(output_dir, 'metrics.json')
    results = extract_results_from_metrics(metrics_path)
    
    # If not found, try log.txt
    if not results:
        log_path = os.path.join(output_dir, 'log.txt')
        results = extract_results_from_log(log_path)
    
    if not results:
        print(f"Error: Could not extract results from {output_dir}")
        sys.exit(1)
    
    # Save results
    output_path = os.path.join(output_dir, args.output)
    save_results_to_txt(results, output_path, args.name)
    
    print(f"✓ Results saved to: {output_path}")
    print("\n" + "=" * 60)
    with open(output_path, 'r') as f:
        print(f.read())


if __name__ == '__main__':
    main()
