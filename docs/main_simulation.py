#!/usr/bin/env python3
"""
Main Simulation Script for Gradient-Based Sensitivity Analysis in Large Language Models

This script implements the complete pipeline described in the paper:
1. Load and configure a transformer model
2. Stream text corpus through gradient scanning
3. Identify most sensitive parameters
4. Perform bit-flip corruption experiments
5. Evaluate and classify output degradation
6. Generate comprehensive analysis reports

Usage:
    python main_simulation.py --model gpt2 --steps 1000 --experiment
"""

import argparse
import os
import sys
import time
from datetime import datetime
import json
import torch

# Import our custom modules
from model_config import ModelConfig, get_recommended_config
from data_processing import DataLoader
from gradient_scanner import GradientScanner
from bit_flip_simulator import BitFlipSimulator
from output_evaluation import OutputEvaluator
from sensitivity_analysis import SensitivityAnalyzer


class GradientBasedSensitivitySimulation:
    """
    Main simulation class that orchestrates the complete sensitivity analysis pipeline.
    """

    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"results_{self.timestamp}"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self.model_config = None
        self.model = None
        self.tokenizer = None
        self.data_loader = None
        self.gradient_scanner = None
        self.bit_flip_simulator = None
        self.output_evaluator = None
        self.sensitivity_analyzer = None

        print(f"Gradient-Based Sensitivity Analysis Simulation")
        print(f"Output directory: {self.output_dir}")
        print(f"Configuration: {args}")

    def setup_model(self):
        """Initialize and configure the model."""
        print(f"\n{'='*60}")
        print(f"STEP 1: MODEL SETUP")
        print(f"{'='*60}")

        self.model_config = ModelConfig(self.args.model, device=self.args.device)
        self.model, self.tokenizer = self.model_config.load_model(self.args.model_variant)

        # Print model summary
        self.model_config.print_model_summary()

        # Save model info
        model_info = self.model_config.get_model_info()
        with open(f"{self.output_dir}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2, default=str)

    def setup_data(self):
        """Initialize data processing pipeline."""
        print(f"\n{'='*60}")
        print(f"STEP 2: DATA SETUP")
        print(f"{'='*60}")

        self.data_loader = DataLoader(
            dataset_name=self.args.dataset,
            subset=self.args.dataset_subset,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size
        )
        self.data_loader.set_tokenizer(self.tokenizer)

        print(f"Dataset: {self.args.dataset}")
        print(f"Sequence length: {self.args.seq_len}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Maximum steps: {self.args.max_steps}")

    def run_gradient_scan(self):
        """Perform gradient scanning to identify sensitive parameters."""
        print(f"\n{'='*60}")
        print(f"STEP 3: GRADIENT SCANNING")
        print(f"{'='*60}")

        self.gradient_scanner = GradientScanner(self.model, device=self.args.device)

        # Run the gradient scan
        start_time = time.time()
        scan_results = self.gradient_scanner.scan_dataset(
            self.data_loader, max_steps=self.args.max_steps
        )
        scan_time = time.time() - start_time

        print(f"\nGradient scan completed in {scan_time:.2f} seconds")
        print(f"Average loss: {scan_results['avg_loss']:.4f}")
        print(f"Average max gradient: {scan_results['avg_max_gradient']:.6f}")

        # Save gradient scan results
        self.gradient_scanner.save_results(f"{self.output_dir}/gradient_scan_results.pt")

        # Generate sensitivity report
        self.gradient_scanner.print_sensitivity_report(
            top_k=self.args.top_k,
            exclude_embeddings=True
        )

    def analyze_sensitivity(self):
        """Perform comprehensive sensitivity analysis."""
        print(f"\n{'='*60}")
        print(f"STEP 4: SENSITIVITY ANALYSIS")
        print(f"{'='*60}")

        # Get sensitivity rankings
        all_rankings = self.gradient_scanner.get_sensitivity_rankings()
        top_sensitive = self.gradient_scanner.get_top_k_sensitive(
            k=self.args.top_k, exclude_embeddings=False, exclude_layernorm=False
        )
        top_filtered = self.gradient_scanner.get_top_k_sensitive(
            k=self.args.top_k, exclude_embeddings=True, exclude_layernorm=True
        )

        # Initialize sensitivity analyzer
        self.sensitivity_analyzer = SensitivityAnalyzer(self.gradient_scanner)
        self.sensitivity_analyzer.load_sensitivity_data(all_rankings)

        # Generate detailed analysis
        self.sensitivity_analyzer.print_detailed_analysis(top_k=self.args.top_k)

        # Create visualization dashboard
        try:
            figures = self.sensitivity_analyzer.create_sensitivity_dashboard(
                top_k=self.args.top_k,
                save_dir=f"{self.output_dir}/plots"
            )
            print(f"Sensitivity plots saved to {self.output_dir}/plots")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

        # Export analysis results
        self.sensitivity_analyzer.export_results(
            f"{self.output_dir}/sensitivity_analysis.json",
            top_k=1000
        )

        return top_sensitive, top_filtered

    def run_bit_flip_experiment(self, sensitive_params):
        """Run bit-flip corruption experiments."""
        if not self.args.experiment:
            print("Skipping bit-flip experiments (use --experiment flag to enable)")
            return None

        print(f"\n{'='*60}")
        print(f"STEP 5: BIT-FLIP EXPERIMENTS")
        print(f"{'='*60}")

        # Initialize bit-flip simulator
        self.bit_flip_simulator = BitFlipSimulator(self.model, device=self.args.device)

        # Define test prompts (from the paper)
        test_prompts = [
            "The weather today is",
            "The patient should take",
            "The bank transfer amount is",
            "The recommended dose for a child is",
            "The evacuation order status is"
        ]

        print(f"Running bit-flip experiment with {len(test_prompts)} prompts")
        print(f"Testing top-{self.args.num_corruptions} most sensitive parameters")
        print(f"Trials per prompt: {self.args.trials_per_prompt}")

        # Run comprehensive experiment
        start_time = time.time()
        experiment_results = self.bit_flip_simulator.run_bit_flip_experiment(
            sensitive_params=sensitive_params,
            tokenizer=self.tokenizer,
            test_prompts=test_prompts,
            num_corruptions=self.args.num_corruptions,
            trials_per_prompt=self.args.trials_per_prompt,
            bit_types=['sign', 'exponent', 'mantissa']
        )
        experiment_time = time.time() - start_time

        print(f"Bit-flip experiments completed in {experiment_time:.2f} seconds")

        # Save experiment results
        with open(f"{self.output_dir}/bit_flip_results.json", 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)

        return experiment_results

    def evaluate_results(self, experiment_results):
        """Evaluate and classify experiment results."""
        if experiment_results is None:
            return None

        print(f"\n{'='*60}")
        print(f"STEP 6: RESULT EVALUATION")
        print(f"{'='*60}")

        # Initialize output evaluator
        self.output_evaluator = OutputEvaluator()

        # Evaluate experiment results
        evaluation_results = self.output_evaluator.evaluate_experiment_results(experiment_results)

        # Generate evaluation report
        report_text = self.output_evaluator.create_evaluation_report(
            evaluation_results,
            save_path=f"{self.output_dir}/evaluation_report.txt"
        )
        print(report_text)

        # Compare with paper results
        paper_comparison = self.output_evaluator.compare_with_paper_results(evaluation_results)
        print(f"\nComparison with paper results:")
        print(f"Overall similarity: {paper_comparison['overall_similarity']:.3f}")

        # Save evaluation results
        with open(f"{self.output_dir}/evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        with open(f"{self.output_dir}/paper_comparison.json", 'w') as f:
            json.dump(paper_comparison, f, indent=2, default=str)

        return evaluation_results

    def generate_summary(self, experiment_results, evaluation_results):
        """Generate final summary report."""
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")

        summary = {
            'simulation_info': {
                'timestamp': self.timestamp,
                'model': self.args.model,
                'model_variant': self.args.model_variant,
                'dataset': self.args.dataset,
                'max_steps': self.args.max_steps,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'experiment_conducted': self.args.experiment
            },
            'gradient_scan_summary': {
                'total_steps': self.gradient_scanner.step_count if self.gradient_scanner else 0,
                'top_sensitivity': self.gradient_scanner.get_top_k_sensitive(k=1)[0] if self.gradient_scanner else None
            }
        }

        if evaluation_results:
            summary['experiment_summary'] = {
                'total_trials': evaluation_results['total_trials'],
                'classification_percentages': evaluation_results['overall_classification']['percentages'],
                'avg_f1_score': evaluation_results['f1_statistics']['mean']
            }

        # Print key findings
        print(f"Model: {summary['simulation_info']['model']}")
        print(f"Total parameters: {summary['simulation_info']['total_parameters']:,}")
        print(f"Gradient scan steps: {summary['gradient_scan_summary']['total_steps']}")

        if summary['gradient_scan_summary']['top_sensitivity']:
            top_param = summary['gradient_scan_summary']['top_sensitivity']
            print(f"Most sensitive parameter: {top_param[0]} at {top_param[1]} (sensitivity: {top_param[2]:.3e})")

        if evaluation_results:
            percentages = summary['experiment_summary']['classification_percentages']
            print(f"\nBit-flip experiment results ({summary['experiment_summary']['total_trials']} trials):")
            print(f"  Preserved: {percentages['preserved']:.1f}%")
            print(f"  Changed: {percentages['changed']:.1f}%")
            print(f"  Gibberish: {percentages['gibberish']:.1f}%")
            print(f"  Average F1 Score: {summary['experiment_summary']['avg_f1_score']:.4f}")

        print(f"\nAll results saved to: {self.output_dir}/")

        # Save summary
        with open(f"{self.output_dir}/simulation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def run_full_simulation(self):
        """Execute the complete simulation pipeline."""
        try:
            # Step 1: Setup model
            self.setup_model()

            # Step 2: Setup data
            self.setup_data()

            # Step 3: Run gradient scanning
            self.run_gradient_scan()

            # Step 4: Analyze sensitivity
            top_sensitive, top_filtered = self.analyze_sensitivity()

            # Step 5: Run bit-flip experiments (optional)
            experiment_results = self.run_bit_flip_experiment(top_sensitive)

            # Step 6: Evaluate results
            evaluation_results = self.evaluate_results(experiment_results)

            # Step 7: Generate final summary
            self.generate_summary(experiment_results, evaluation_results)

            print(f"\n{'='*60}")
            print(f"SIMULATION COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")

        except KeyboardInterrupt:
            print(f"\nSimulation interrupted by user")
            print(f"Partial results saved to: {self.output_dir}/")
        except Exception as e:
            print(f"\nSimulation failed with error: {e}")
            import traceback
            traceback.print_exc()
            print(f"Partial results may be available in: {self.output_dir}/")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gradient-Based Sensitivity Analysis Simulation')

    # Model configuration
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name (gpt2, gpt-neo, etc.)')
    parser.add_argument('--model-variant', type=str, default=None,
                       help='Specific model variant (e.g., gpt2-medium)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')

    # Data configuration
    parser.add_argument('--dataset', type=str, default='wikitext',
                       help='Dataset name')
    parser.add_argument('--dataset-subset', type=str, default='wikitext-103-raw-v1',
                       help='Dataset subset')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')

    # Gradient scanning
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum gradient scanning steps')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Number of top parameters to analyze')

    # Bit-flip experiments
    parser.add_argument('--experiment', action='store_true',
                       help='Run bit-flip corruption experiments')
    parser.add_argument('--num-corruptions', type=int, default=3,
                       help='Number of top parameters to corrupt')
    parser.add_argument('--trials-per-prompt', type=int, default=5,
                       help='Number of trials per prompt per bit type')

    # Configuration presets
    parser.add_argument('--config', type=str, choices=['small', 'medium', 'large'],
                       help='Use predefined configuration (overrides individual settings)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Apply configuration preset if specified
    if args.config:
        config = get_recommended_config(args.config)
        for key, value in config.items():
            if not hasattr(args, key.replace('_', '-')) or getattr(args, key.replace('_', '-')) is None:
                setattr(args, key.replace('_', '-'), value)

    # Set default model variant if not specified
    if args.model_variant is None:
        if args.model == 'gpt2':
            args.model_variant = 'gpt2'
        elif args.model == 'gpt-neo':
            args.model_variant = 'EleutherAI/gpt-neo-125M'
        else:
            args.model_variant = args.model

    # Create and run simulation
    simulation = GradientBasedSensitivitySimulation(args)
    simulation.run_full_simulation()


if __name__ == "__main__":
    main()