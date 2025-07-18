#!/usr/bin/env python3
"""
ALPS Analysis Runner Script
Support running ALPS analysis via configuration file
"""

import yaml
import argparse
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from alps import ALPSAnalyzer

def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_alps_from_config(config_path: str):
    """Run ALPS analysis from configuration file"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract model configuration
    models_config = config.get('models', {})
    base_model = models_config['base_model']
    task_model = models_config['task_model']
    
    # Extract output configuration
    output_config = config.get('output', {})
    output_dir = output_config.get('output_dir', 'ALPS_output')
    show_plot = output_config.get('show_plot', False)
    
    # Extract analysis parameters
    analysis_config = config.get('analysis', {})
    distance_metric = analysis_config.get('distance_metric', 'wasserstein_normalized')
    print_top_n = analysis_config.get('print_top_n', 5)
    
    # Extract device configuration
    device_config = config.get('device', {})
    
    # Extract logging configuration and setup if provided
    logging_config = config.get('logging', {})
    if logging_config:
        import logging
        logging.basicConfig(
            level=getattr(logging, logging_config.get('level', 'INFO')),
            format=logging_config.get('format', '[%(asctime)s] [%(levelname)s] %(message)s'),
            datefmt=logging_config.get('datefmt', '%Y-%m-%d %H:%M:%S')
        )
    
    print(f"Configuration file: {config_path}")
    print(f"Base model: {base_model}")
    print(f"Task model: {task_model}")
    print(f"Output directory: {output_dir}")
    print(f"Distance metric: {distance_metric}")
    print(f"Show plot: {show_plot}")
    print(f"Device config: {device_config}")
    
    # Create ALPS analyzer
    analyzer = ALPSAnalyzer(
        base_model_path=base_model,
        task_model_path=task_model,
        output_dir=output_dir
    )
    
    # Run analysis with configuration parameters
    results = analyzer.run_analysis(distance_metric=distance_metric, show_plot=show_plot)
    
    print(f"\n✅ ALPS analysis completed!")
    print(f"�� Analysis results saved in: {output_dir}")
    print(f"�� Distance metric used: {distance_metric}")
    
    # Print score statistics from results
    if 'sensitivity_scores' in results and results['sensitivity_scores']:
        scores = list(results['sensitivity_scores'].values())
        print(f"�� Score statistics:")
        print(f"   - Min: {min(scores):.6f}")
        print(f"   - Max: {max(scores):.6f}")
        print(f"   - Mean: {sum(scores)/len(scores):.6f}")
        print(f"   - Std: {(sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5:.6f}")
        
        # Show top N most sensitive heads
        top_n = sorted(results['sensitivity_scores'].items(), key=lambda x: x[1], reverse=True)[:print_top_n]
        print(f"�� Top {print_top_n} most sensitive attention heads:")
        for i, (head_name, score) in enumerate(top_n, 1):
            parts = head_name.split('_')
            layer_idx = parts[1]
            head_idx = parts[3]
            print(f"   {i}. Layer {layer_idx}, Head {head_idx}: {score:.6f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run ALPS analysis using configuration file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Check if configuration file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file does not exist: {args.config}")
        print(f"Please create configuration file or specify correct path")
        return
    
    # Run analysis
    try:
        results = run_alps_from_config(args.config)
        print(f"\n�� Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error occurred during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 