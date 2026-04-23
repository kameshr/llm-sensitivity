"""
Sensitivity Analysis Module

Provides comprehensive analysis and visualization of parameter sensitivity results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from collections import defaultdict, Counter
import json


class SensitivityAnalyzer:
    """
    Analyzes and visualizes gradient-based sensitivity results.
    """

    def __init__(self, gradient_scanner=None):
        self.gradient_scanner = gradient_scanner
        self.sensitivity_data = None

    def load_sensitivity_data(self, sensitivity_rankings: List[Tuple[str, Tuple, float]]):
        """Load sensitivity rankings for analysis."""
        self.sensitivity_data = sensitivity_rankings

    def analyze_layer_distribution(self, top_k: int = 100) -> Dict[str, Any]:
        """
        Analyze the distribution of sensitive parameters across different layer types.

        Args:
            top_k: Number of top parameters to analyze

        Returns:
            Dictionary with layer distribution analysis
        """
        if not self.sensitivity_data:
            return {}

        top_params = self.sensitivity_data[:top_k]
        layer_counts = defaultdict(int)
        layer_sensitivities = defaultdict(list)
        layer_total_magnitude = defaultdict(float)

        for param_name, element_idx, sensitivity in top_params:
            layer_type = self._classify_parameter_name(param_name)
            layer_counts[layer_type] += 1
            layer_sensitivities[layer_type].append(sensitivity)
            layer_total_magnitude[layer_type] += sensitivity

        # Calculate statistics
        layer_stats = {}
        for layer_type in layer_counts:
            sensitivities = layer_sensitivities[layer_type]
            layer_stats[layer_type] = {
                'count': layer_counts[layer_type],
                'percentage': (layer_counts[layer_type] / top_k) * 100,
                'avg_sensitivity': np.mean(sensitivities),
                'max_sensitivity': np.max(sensitivities),
                'min_sensitivity': np.min(sensitivities),
                'total_magnitude': layer_total_magnitude[layer_type],
                'std_sensitivity': np.std(sensitivities)
            }

        return {
            'top_k': top_k,
            'layer_stats': layer_stats,
            'total_layers': len(layer_stats)
        }

    def analyze_parameter_patterns(self, top_k: int = 50) -> Dict[str, Any]:
        """
        Analyze patterns in the most sensitive parameters.

        Args:
            top_k: Number of top parameters to analyze

        Returns:
            Dictionary with pattern analysis
        """
        if not self.sensitivity_data:
            return {}

        top_params = self.sensitivity_data[:top_k]

        # Analyze by parameter name patterns
        param_name_counts = defaultdict(int)
        param_name_sensitivities = defaultdict(list)

        # Analyze by layer depth (for transformer models)
        layer_depth_counts = defaultdict(int)
        layer_depth_sensitivities = defaultdict(list)

        # Analyze sensitivity magnitude distribution
        sensitivities = [s for _, _, s in top_params]

        for param_name, element_idx, sensitivity in top_params:
            # Group by full parameter name
            param_name_counts[param_name] += 1
            param_name_sensitivities[param_name].append(sensitivity)

            # Extract layer depth for transformers
            layer_depth = self._extract_layer_depth(param_name)
            if layer_depth is not None:
                layer_depth_counts[layer_depth] += 1
                layer_depth_sensitivities[layer_depth].append(sensitivity)

        # Find most vulnerable parameter tensors
        most_vulnerable_tensors = sorted(
            param_name_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'top_k': top_k,
            'sensitivity_distribution': {
                'mean': np.mean(sensitivities),
                'std': np.std(sensitivities),
                'min': np.min(sensitivities),
                'max': np.max(sensitivities),
                'percentiles': {
                    '50': np.percentile(sensitivities, 50),
                    '75': np.percentile(sensitivities, 75),
                    '90': np.percentile(sensitivities, 90),
                    '95': np.percentile(sensitivities, 95),
                    '99': np.percentile(sensitivities, 99)
                }
            },
            'most_vulnerable_tensors': most_vulnerable_tensors,
            'layer_depth_analysis': {
                'counts': dict(layer_depth_counts),
                'avg_sensitivities': {
                    depth: np.mean(sensitivities)
                    for depth, sensitivities in layer_depth_sensitivities.items()
                }
            }
        }

    def compare_with_random_baseline(self, model_params: Dict[str, torch.Tensor],
                                   num_random_samples: int = 1000) -> Dict[str, Any]:
        """
        Compare top sensitive parameters with random baseline.

        Args:
            model_params: Dictionary of model parameters
            num_random_samples: Number of random parameters to sample

        Returns:
            Comparison results
        """
        if not self.sensitivity_data:
            return {}

        # Get sensitivities of top parameters
        top_sensitivities = [s for _, _, s in self.sensitivity_data[:num_random_samples]]

        # Sample random parameters
        random_sensitivities = []
        all_param_names = list(model_params.keys())

        for _ in range(num_random_samples):
            # Choose random parameter
            param_name = np.random.choice(all_param_names)
            param_tensor = model_params[param_name]

            # Choose random element
            if param_tensor.numel() > 1:
                flat_idx = np.random.randint(0, param_tensor.numel())
                element_idx = np.unravel_index(flat_idx, param_tensor.shape)
            else:
                element_idx = (0,)

            # This would require actual gradient values, so we'll use placeholder
            # In practice, you'd compute gradients for these random elements
            random_sensitivity = 0.0  # Placeholder
            random_sensitivities.append(random_sensitivity)

        return {
            'top_sensitivity_stats': {
                'mean': np.mean(top_sensitivities),
                'std': np.std(top_sensitivities),
                'max': np.max(top_sensitivities)
            },
            'random_baseline_stats': {
                'mean': np.mean(random_sensitivities),
                'std': np.std(random_sensitivities),
                'max': np.max(random_sensitivities)
            },
            'enhancement_factor': np.mean(top_sensitivities) / (np.mean(random_sensitivities) + 1e-10)
        }

    def generate_sensitivity_heatmap(self, param_name: str, save_path: str = None) -> Optional[plt.Figure]:
        """
        Generate a heatmap showing sensitivity distribution for a specific parameter tensor.

        Args:
            param_name: Name of the parameter tensor
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        if not self.gradient_scanner or not self.gradient_scanner.running_max:
            print("No gradient scanner data available")
            return None

        if param_name not in self.gradient_scanner.running_max:
            print(f"Parameter {param_name} not found in gradient data")
            return None

        sensitivity_tensor = self.gradient_scanner.running_max[param_name]

        # For 2D tensors, create heatmap directly
        if len(sensitivity_tensor.shape) == 2:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            sns.heatmap(sensitivity_tensor.cpu().numpy(), ax=ax, cmap='viridis')
            ax.set_title(f'Sensitivity Heatmap: {param_name}')
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index')

        # For 1D tensors, create bar plot
        elif len(sensitivity_tensor.shape) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            ax.plot(sensitivity_tensor.cpu().numpy())
            ax.set_title(f'Sensitivity Distribution: {param_name}')
            ax.set_xlabel('Parameter Index')
            ax.set_ylabel('Gradient Magnitude')

        # For higher-dimensional tensors, flatten and show distribution
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            flat_data = sensitivity_tensor.flatten().cpu().numpy()
            ax.hist(flat_data, bins=50, alpha=0.7)
            ax.set_title(f'Sensitivity Distribution: {param_name}')
            ax.set_xlabel('Gradient Magnitude')
            ax.set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")

        return fig

    def create_sensitivity_dashboard(self, top_k: int = 100, save_dir: str = None) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive dashboard of sensitivity analysis plots.

        Args:
            top_k: Number of top parameters to analyze
            save_dir: Directory to save plots

        Returns:
            Dictionary of matplotlib figures
        """
        if not self.sensitivity_data:
            print("No sensitivity data available")
            return {}

        figures = {}

        # 1. Top-k sensitivity distribution
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        top_sensitivities = [s for _, _, s in self.sensitivity_data[:top_k]]
        ax1.plot(range(1, len(top_sensitivities) + 1), top_sensitivities, 'b-', linewidth=2)
        ax1.set_xlabel('Parameter Rank')
        ax1.set_ylabel('Gradient Magnitude')
        ax1.set_title(f'Top-{top_k} Parameter Sensitivity Distribution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        figures['sensitivity_distribution'] = fig1

        # 2. Layer type distribution
        layer_analysis = self.analyze_layer_distribution(top_k)
        if layer_analysis:
            layer_stats = layer_analysis['layer_stats']
            layer_names = list(layer_stats.keys())
            layer_counts = [layer_stats[name]['count'] for name in layer_names]

            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))

            # Pie chart
            ax2a.pie(layer_counts, labels=layer_names, autopct='%1.1f%%')
            ax2a.set_title('Distribution by Layer Type')

            # Bar chart
            ax2b.bar(layer_names, layer_counts)
            ax2b.set_xlabel('Layer Type')
            ax2b.set_ylabel('Number of Parameters')
            ax2b.set_title('Parameter Count by Layer Type')
            ax2b.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            figures['layer_distribution'] = fig2

        # 3. Sensitivity magnitude histogram
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        all_sensitivities = [s for _, _, s in self.sensitivity_data[:min(1000, len(self.sensitivity_data))]]
        ax3.hist(all_sensitivities, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Gradient Magnitude')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Gradient Magnitudes')
        ax3.set_yscale('log')
        figures['magnitude_histogram'] = fig3

        # Save figures if directory provided
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_dir}")

        return figures

    def export_results(self, filepath: str, top_k: int = 1000):
        """
        Export sensitivity analysis results to JSON file.

        Args:
            filepath: Path to save JSON file
            top_k: Number of top parameters to export
        """
        if not self.sensitivity_data:
            print("No sensitivity data to export")
            return

        export_data = {
            'metadata': {
                'total_parameters': len(self.sensitivity_data),
                'top_k_exported': min(top_k, len(self.sensitivity_data))
            },
            'top_parameters': [],
            'layer_analysis': self.analyze_layer_distribution(top_k),
            'pattern_analysis': self.analyze_parameter_patterns(top_k)
        }

        # Export top parameters
        for rank, (param_name, element_idx, sensitivity) in enumerate(self.sensitivity_data[:top_k], 1):
            export_data['top_parameters'].append({
                'rank': rank,
                'parameter_name': param_name,
                'element_index': element_idx,
                'sensitivity': float(sensitivity),
                'layer_type': self._classify_parameter_name(param_name)
            })

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Results exported to {filepath}")

    def _classify_parameter_name(self, param_name: str) -> str:
        """Classify parameter by layer type based on name."""
        name_lower = param_name.lower()

        if 'embed' in name_lower or 'wte' in name_lower or 'wpe' in name_lower:
            return 'embedding'
        elif 'ln' in name_lower or 'norm' in name_lower:
            return 'layer_norm'
        elif 'attn' in name_lower or 'attention' in name_lower:
            return 'attention'
        elif 'mlp' in name_lower or 'fc' in name_lower or 'dense' in name_lower:
            return 'mlp'
        elif 'lm_head' in name_lower or 'output' in name_lower:
            return 'output_head'
        else:
            return 'other'

    def _extract_layer_depth(self, param_name: str) -> Optional[int]:
        """Extract layer depth from parameter name (for transformer models)."""
        import re
        # Look for patterns like 'h.0.', 'layers.5.', 'layer.10.', etc.
        patterns = [r'\.h\.(\d+)\.', r'\.layers\.(\d+)\.', r'\.layer\.(\d+)\.']

        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))

        return None

    def print_detailed_analysis(self, top_k: int = 50):
        """Print a comprehensive text-based analysis report."""
        if not self.sensitivity_data:
            print("No sensitivity data available for analysis")
            return

        print(f"\n{'='*80}")
        print(f"DETAILED SENSITIVITY ANALYSIS REPORT")
        print(f"{'='*80}")

        # Overall statistics
        all_sensitivities = [s for _, _, s in self.sensitivity_data]
        print(f"Total parameters analyzed: {len(all_sensitivities):,}")
        print(f"Sensitivity range: {min(all_sensitivities):.3e} to {max(all_sensitivities):.3e}")
        print(f"Mean sensitivity: {np.mean(all_sensitivities):.3e}")
        print(f"Median sensitivity: {np.median(all_sensitivities):.3e}")

        # Layer distribution analysis
        layer_analysis = self.analyze_layer_distribution(top_k)
        if layer_analysis:
            print(f"\nLayer Distribution Analysis (Top-{top_k}):")
            print("-" * 50)
            layer_stats = layer_analysis['layer_stats']
            for layer_type, stats in layer_stats.items():
                print(f"{layer_type.capitalize()}:")
                print(f"  Count: {stats['count']} ({stats['percentage']:.1f}%)")
                print(f"  Avg sensitivity: {stats['avg_sensitivity']:.3e}")
                print(f"  Max sensitivity: {stats['max_sensitivity']:.3e}")

        # Pattern analysis
        pattern_analysis = self.analyze_parameter_patterns(top_k)
        if pattern_analysis:
            print(f"\nMost Vulnerable Parameter Tensors:")
            print("-" * 50)
            for param_name, count in pattern_analysis['most_vulnerable_tensors']:
                percentage = (count / top_k) * 100
                print(f"  {param_name}: {count} elements ({percentage:.1f}%)")


if __name__ == "__main__":
    # Test the sensitivity analyzer
    print("Testing sensitivity analyzer...")

    # Create mock sensitivity data
    mock_sensitivity_data = [
        ('transformer.wte.weight', (100, 50), 5.235),
        ('transformer.wte.weight', (200, 50), 4.481),
        ('transformer.h.0.attn.c_attn.weight', (10, 20), 3.247),
        ('transformer.h.1.mlp.c_fc.weight', (15, 30), 2.898),
        ('transformer.h.0.ln_1.weight', (50,), 2.669),
        ('transformer.h.2.attn.c_proj.weight', (25, 40), 2.517),
    ]

    # Initialize analyzer
    analyzer = SensitivityAnalyzer()
    analyzer.load_sensitivity_data(mock_sensitivity_data)

    # Test analysis functions
    layer_dist = analyzer.analyze_layer_distribution(top_k=6)
    print("Layer distribution analysis completed")

    pattern_analysis = analyzer.analyze_parameter_patterns(top_k=6)
    print("Pattern analysis completed")

    # Test detailed report
    analyzer.print_detailed_analysis(top_k=6)

    print("\nSensitivity analyzer test completed!")