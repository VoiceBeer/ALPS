import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from transformers import AutoModel, AutoConfig
import argparse
import os
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ALPSAnalyzer:
    """
    ALPS (Attention Localization and Pruning Strategy) Algorithm Implementation
    
    This algorithm identifies task-sensitive attention heads by comparing weight 
    distribution differences between a base model and task-specific model.
    """
    
    def __init__(self, base_model_path: str, task_model_path: str, output_dir: str = "ALPS_output"):
        """
        Initialize ALPS Analyzer
        
        Args:
            base_model_path: Path to base model (e.g., qwen2.5-7B-Instruct)
            task_model_path: Path to task-specific model (e.g., qwen2.5-Coder-7B-Instruct)
            output_dir: Output directory
        """
        self.base_model_path = base_model_path
        self.task_model_path = task_model_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initializing ALPS Analyzer")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Task model: {task_model_path}")
        
    def load_models(self):
        """Load base model and task-specific model"""
        logger.info("Loading models...")
        
        # Load configurations
        self.base_config = AutoConfig.from_pretrained(self.base_model_path)
        self.task_config = AutoConfig.from_pretrained(self.task_model_path)
        
        # Load models
        self.base_model = AutoModel.from_pretrained(
            self.base_model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.task_model = AutoModel.from_pretrained(
            self.task_model_path,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        logger.info("Models loaded successfully")
        
        # Get model architecture information
        self.num_layers = self.base_config.num_hidden_layers
        self.num_heads = self.base_config.num_attention_heads
        
        logger.info(f"Number of layers: {self.num_layers}")
        logger.info(f"Number of attention heads: {self.num_heads}")
        
    def extract_attention_heads(self, model, model_name: str) -> Dict[str, torch.Tensor]:
        """
        Extract all attention head weights from the model
        
        Args:
            model: Model to analyze
            model_name: Model name (for logging)
            
        Returns:
            Dictionary containing all attention head weights
        """
        logger.info(f"Extracting attention head weights from {model_name}...")
        
        attention_heads = {}
        
        for layer_idx in range(self.num_layers):
            # Get attention layer
            if hasattr(model, 'layers'):  # Qwen model structure
                attn_layer = model.layers[layer_idx].self_attn
            elif hasattr(model, 'h'):  # GPT model structure
                attn_layer = model.h[layer_idx].attn
            else:
                raise ValueError(f"Unsupported model structure: {type(model)}")
            
            # Extract Q, K, V projection weights
            if hasattr(attn_layer, 'q_proj'):
                q_weight = attn_layer.q_proj.weight.data
                k_weight = attn_layer.k_proj.weight.data
                v_weight = attn_layer.v_proj.weight.data
            elif hasattr(attn_layer, 'c_attn'):  # GPT-style structure
                # For merged QKV weights, need to split
                qkv_weight = attn_layer.c_attn.weight.data
                head_dim = qkv_weight.shape[0] // 3
                q_weight = qkv_weight[:head_dim]
                k_weight = qkv_weight[head_dim:2*head_dim]
                v_weight = qkv_weight[2*head_dim:]
            else:
                raise ValueError(f"Cannot find attention projection layers in layer {layer_idx}")
            
            # Reshape weights by attention heads
            head_dim = q_weight.shape[0] // self.num_heads
            
            for head_idx in range(self.num_heads):
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                
                # Extract single head weights
                q_head = q_weight[start_idx:end_idx]
                k_head = k_weight[start_idx:end_idx]
                v_head = v_weight[start_idx:end_idx]
                
                # Concatenate QKV weights as head representation
                head_weight = torch.cat([q_head.flatten(), k_head.flatten(), v_head.flatten()])
                
                attention_heads[f"layer_{layer_idx}_head_{head_idx}"] = head_weight
        
        logger.info(f"{model_name} attention head extraction completed, total {len(attention_heads)} heads")
        return attention_heads
    
    def compute_weight_distribution(self, weights: torch.Tensor) -> np.ndarray:
        """
        Compute softmax distribution of weights
        
        Args:
            weights: Attention head weight tensor
            
        Returns:
            Softmax normalized weight distribution
        """
        # Convert weights to probability distribution
        weights_flat = weights.flatten().float()
        # Apply softmax to get probability distribution
        distribution = F.softmax(weights_flat, dim=0)
        return distribution.cpu().numpy()
    
    def compute_wasserstein_distance(self, dist1: np.ndarray, dist2: np.ndarray, 
                                    normalize: bool = True, use_unit_support: bool = True) -> float:
        """
        Compute Wasserstein distance between two distributions
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            normalize: Whether to normalize the distance by distribution length
            use_unit_support: Whether to use unit interval [0,1] as support
            
        Returns:
            Wasserstein distance
        """
        if use_unit_support:
            # Use unit interval [0,1] as support points
            x = np.linspace(0, 1, len(dist1))
        else:
            # Use indices as support points
            x = np.arange(len(dist1))
        
        distance = wasserstein_distance(x, x, dist1, dist2)
        
        # Normalize by distribution length if requested
        if normalize and not use_unit_support:
            distance = distance / len(dist1)
            
        return distance
    
    def compute_kl_divergence(self, dist1: np.ndarray, dist2: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Compute KL divergence between two distributions as alternative metric
        
        Args:
            dist1: First distribution (base model)
            dist2: Second distribution (task model)
            epsilon: Small value to avoid log(0)
            
        Returns:
            KL divergence D(dist2 || dist1)
        """
        # Add small epsilon to avoid log(0)
        dist1_safe = dist1 + epsilon
        dist2_safe = dist2 + epsilon
        
        # Normalize to ensure they sum to 1
        dist1_safe = dist1_safe / dist1_safe.sum()
        dist2_safe = dist2_safe / dist2_safe.sum()
        
        # Compute KL divergence: D(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(dist2_safe * np.log(dist2_safe / dist1_safe))
        return kl_div
    
    def compute_cosine_similarity(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """
        Compute cosine similarity between two distributions
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Cosine distance (1 - cosine_similarity) for consistency with other metrics
        """
        # Flatten distributions to vectors
        vec1 = dist1.flatten()
        vec2 = dist2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance if one vector is zero
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Return cosine distance (1 - similarity) so higher values indicate more difference
        cosine_distance = 1.0 - cosine_sim
        return cosine_distance
    
    def compute_js_divergence(self, dist1: np.ndarray, dist2: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            epsilon: Small value to avoid log(0)
            
        Returns:
            Jensen-Shannon divergence
        """
        # Add small epsilon and normalize
        dist1_safe = (dist1 + epsilon)
        dist2_safe = (dist2 + epsilon)
        dist1_safe = dist1_safe / dist1_safe.sum()
        dist2_safe = dist2_safe / dist2_safe.sum()
        
        # Compute average distribution
        m = 0.5 * (dist1_safe + dist2_safe)
        
        # Compute JS divergence
        js_div = 0.5 * np.sum(dist1_safe * np.log(dist1_safe / m)) + \
                 0.5 * np.sum(dist2_safe * np.log(dist2_safe / m))
        return js_div
    
    def analyze_task_sensitivity(self, distance_metric: str = "wasserstein_normalized") -> Dict[str, float]:
        """
        Analyze task sensitivity of all attention heads
        
        Args:
            distance_metric: Distance metric to use. Options:
                - "wasserstein": Original Wasserstein distance
                - "wasserstein_normalized": Normalized Wasserstein with unit support
                - "kl_divergence": KL divergence
                - "cosine_similarity": Cosine distance (1 - cosine similarity)
                - "js_divergence": Jensen-Shannon divergence
        
        Returns:
            Dictionary containing task sensitivity scores for each attention head
        """
        logger.info(f"Starting task sensitivity analysis using {distance_metric}...")
        
        # Extract attention heads from both models
        base_heads = self.extract_attention_heads(self.base_model, "base model")
        task_heads = self.extract_attention_heads(self.task_model, "task model")
        
        # Compute task sensitivity score for each head
        sensitivity_scores = {}
        
        for head_name in base_heads.keys():
            if head_name in task_heads:
                # Compute weight distributions
                base_dist = self.compute_weight_distribution(base_heads[head_name])
                task_dist = self.compute_weight_distribution(task_heads[head_name])
                
                # Compute distance based on selected metric
                if distance_metric == "wasserstein":
                    distance = self.compute_wasserstein_distance(base_dist, task_dist, 
                                                               normalize=False, use_unit_support=False)
                elif distance_metric == "wasserstein_normalized":
                    distance = self.compute_wasserstein_distance(base_dist, task_dist, 
                                                               normalize=True, use_unit_support=True)
                elif distance_metric == "kl_divergence":
                    distance = self.compute_kl_divergence(base_dist, task_dist)
                elif distance_metric == "cosine_similarity":
                    distance = self.compute_cosine_similarity(base_dist, task_dist)
                elif distance_metric == "js_divergence":
                    distance = self.compute_js_divergence(base_dist, task_dist)
                else:
                    raise ValueError(f"Unknown distance metric: {distance_metric}")
                
                sensitivity_scores[head_name] = distance
            else:
                logger.warning(f"Corresponding attention head not found in task model: {head_name}")
        
        logger.info(f"Task sensitivity analysis completed, analyzed {len(sensitivity_scores)} attention heads")
        
        # Log statistics about the scores
        if sensitivity_scores:
            scores = list(sensitivity_scores.values())
            logger.info(f"Score statistics - Min: {min(scores):.6f}, Max: {max(scores):.6f}, "
                       f"Mean: {np.mean(scores):.6f}, Std: {np.std(scores):.6f}")
        
        return sensitivity_scores
    
    def create_heatmap(self, sensitivity_scores: Dict[str, float], distance_metric: str = "wasserstein_normalized", show_plot: bool = True):
        """
        Create heatmap of attention head task sensitivity
        
        Args:
            sensitivity_scores: Task sensitivity score dictionary
            distance_metric: Distance metric used for analysis
            show_plot: Whether to display the plot (set False for headless environments)
        """
        logger.info("Creating heatmap...")
        
        # Prepare heatmap data
        heatmap_data = np.zeros((self.num_layers, self.num_heads))
        
        for head_name, score in sensitivity_scores.items():
            # Parse layer and head indices
            parts = head_name.split('_')
            layer_idx = int(parts[1])
            head_idx = int(parts[3])
            heatmap_data[layer_idx, head_idx] = score
        
        # Create heatmap
        plt.figure(figsize=(max(12, self.num_heads), max(8, self.num_layers // 2)))
        
        # Use seaborn to create heatmap
        sns.heatmap(
            heatmap_data,
            annot=False,
            cmap='viridis',
            cbar_kws={'label': f'Task Sensitivity Score ({distance_metric})'},
            xticklabels=[f'Head {i}' for i in range(self.num_heads)],
            yticklabels=[f'Layer {i}' for i in range(self.num_layers)]
        )
        
        plt.title(f'ALPS Task-Sensitive Attention Heads Heatmap\n'
                 f'Base: {os.path.basename(self.base_model_path)} vs '
                 f'Task: {os.path.basename(self.task_model_path)}')
        plt.xlabel('Attention Head Index')
        plt.ylabel('Layer Index')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.output_dir, f'alps_heatmap_{distance_metric}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to: {save_path}")
        
        # Only display the figure if requested
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
        
        return heatmap_data
    
    def save_results(self, sensitivity_scores: Dict[str, float], heatmap_data: np.ndarray, distance_metric: str):
        """
        Save analysis results
        
        Args:
            sensitivity_scores: Sensitivity scores
            heatmap_data: Heatmap data
            distance_metric: Distance metric used for analysis
        """
        logger.info("Saving results...")
        
        # Save sensitivity scores
        scores_path = os.path.join(self.output_dir, f'sensitivity_scores_{distance_metric}.txt')
        with open(scores_path, 'w') as f:
            f.write("# ALPS Task Sensitivity Analysis Results\n")
            f.write(f"# Base Model: {self.base_model_path}\n")
            f.write(f"# Task Model: {self.task_model_path}\n")
            f.write(f"# Distance Metric: {distance_metric}\n")
            f.write("# Format: layer_index head_index sensitivity_score\n\n")
            
            for head_name, score in sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True):
                parts = head_name.split('_')
                layer_idx = parts[1]
                head_idx = parts[3]
                f.write(f"{layer_idx} {head_idx} {score:.6f}\n")
        
        # Save heatmap data as numpy array
        heatmap_path = os.path.join(self.output_dir, f'heatmap_data_{distance_metric}.npy')
        np.save(heatmap_path, heatmap_data)
        
        # Save top 10% most sensitive attention heads
        top_10_percent = int(len(sensitivity_scores) * 0.1)
        top_heads = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)[:top_10_percent]
        
        top_heads_path = os.path.join(self.output_dir, f'top_10_percent_heads_{distance_metric}.txt')
        with open(top_heads_path, 'w') as f:
            f.write("# Top 10% Most Task-Sensitive Attention Heads\n")
            f.write(f"# Distance Metric: {distance_metric}\n")
            f.write("# Format: layer_index head_index sensitivity_score\n\n")
            
            for head_name, score in top_heads:
                parts = head_name.split('_')
                layer_idx = parts[1]
                head_idx = parts[3]
                f.write(f"{layer_idx} {head_idx} {score:.6f}\n")
        
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Sensitivity scores: {scores_path}")
        logger.info(f"Heatmap data: {heatmap_path}")
        logger.info(f"Top 10% sensitive heads: {top_heads_path}")
    
    def run_analysis(self, distance_metric: str = "wasserstein_normalized", show_plot: bool = True):
        """
        Run complete ALPS analysis pipeline
        
        Args:
            distance_metric: Distance metric to use for analysis
            show_plot: Whether to display plots
        
        Returns:
            Analysis results dictionary
        """
        logger.info("Starting ALPS analysis...")
        
        # 1. Load models
        self.load_models()
        
        # 2. Analyze task sensitivity
        sensitivity_scores = self.analyze_task_sensitivity(distance_metric)
        
        # 3. Create heatmap
        heatmap_data = self.create_heatmap(sensitivity_scores, distance_metric, show_plot)
        
        # 4. Save results
        self.save_results(sensitivity_scores, heatmap_data, distance_metric)
        
        # 5. Return analysis results
        results = {
            'sensitivity_scores': sensitivity_scores,
            'heatmap_data': heatmap_data,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'base_model': self.base_model_path,
            'task_model': self.task_model_path,
            'distance_metric': distance_metric
        }
        
        logger.info("ALPS analysis completed!")
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ALPS: Attention Localization and Pruning Strategy')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Base model path (e.g., Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--task_model', type=str, required=True,
                       help='Task-specific model path (e.g., Qwen/Qwen2.5-Coder-7B-Instruct)')
    parser.add_argument('--output_dir', type=str, default='ALPS_output',
                       help='Output directory (default: ALPS_output)')
    parser.add_argument('--distance_metric', type=str, default='wasserstein_normalized',
                       choices=['wasserstein', 'wasserstein_normalized', 'kl_divergence', 'cosine_similarity', 'js_divergence'],
                       help='Distance metric to use (default: wasserstein_normalized)')
    
    args = parser.parse_args()
    
    # Create and run ALPS analyzer
    analyzer = ALPSAnalyzer(
        base_model_path=args.base_model,
        task_model_path=args.task_model,
        output_dir=args.output_dir
    )
    
    # Run analysis
    results = analyzer.run_analysis(distance_metric=args.distance_metric)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ALPS Analysis Summary")
    print(f"{'='*60}")
    print(f"Base model: {args.base_model}")
    print(f"Task model: {args.task_model}")
    print(f"Number of attention heads analyzed: {len(results['sensitivity_scores'])}")
    print(f"Number of layers: {results['num_layers']}")
    print(f"Attention heads per layer: {results['num_heads']}")
    print(f"Distance metric used: {results['distance_metric']}")
    
    # Show top 5 most sensitive attention heads
    top_5 = sorted(results['sensitivity_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 most task-sensitive attention heads:")
    for i, (head_name, score) in enumerate(top_5, 1):
        parts = head_name.split('_')
        layer_idx = parts[1]
        head_idx = parts[3]
        print(f"{i}. Layer {layer_idx}, Head {head_idx}: {score:.6f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 