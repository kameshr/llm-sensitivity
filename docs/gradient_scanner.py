"""
Gradient Scanning Module for Sensitivity Analysis

Implements Algorithm B.1: Gradient Scan from the paper.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
import numpy as np
from collections import defaultdict
from data_processing import DataLoader


class GradientScanner:
    """
    Implements gradient-based sensitivity analysis for neural language models.

    Tracks maximum absolute gradients per parameter element across many batches
    to identify the most sensitivity-critical model weights.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
        self.running_max = {name: torch.zeros_like(param, dtype=torch.float32)
                           for name, param in self.param_dict.items()}
        self.step_count = 0

    def reset_tracking(self):
        """Reset the gradient tracking to start fresh."""
        self.running_max = {name: torch.zeros_like(param, dtype=torch.float32)
                           for name, param in self.param_dict.items()}
        self.step_count = 0

    def compute_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.

        Args:
            inputs: Input token sequences [B x L]
            labels: Target token sequences [B x L]

        Returns:
            Cross-entropy loss scalar
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Forward pass through model
        outputs = self.model(inputs, labels=labels)

        # Extract loss (for HuggingFace models, loss is automatically computed)
        if hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            # Manual cross-entropy computation
            logits = outputs.logits
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss

    def scan_batch(self, inputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Process a single batch and update gradient maximums.

        Args:
            inputs: Input token sequences [B x L]
            labels: Target token sequences [B x L]

        Returns:
            Dictionary with batch statistics
        """
        # Zero gradients
        self.model.zero_grad(set_to_none=True)

        # Compute loss
        loss = self.compute_loss(inputs, labels)

        # Backward pass
        loss.backward()

        # Update running maximums
        total_params = 0
        max_grad = 0.0

        for name, param in self.param_dict.items():
            if param.grad is not None:
                grad_abs = param.grad.detach().abs()
                self.running_max[name] = torch.maximum(self.running_max[name], grad_abs)

                total_params += param.numel()
                max_grad = max(max_grad, grad_abs.max().item())

        self.step_count += 1

        return {
            'loss': loss.item(),
            'max_gradient': max_grad,
            'total_params': total_params,
            'step': self.step_count
        }

    def scan_dataset(self, data_loader: DataLoader, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Perform gradient scan across multiple batches from dataset.

        Args:
            data_loader: DataLoader instance
            max_steps: Maximum number of batches to process

        Returns:
            Dictionary with scan results and statistics
        """
        print(f"Starting gradient scan for {max_steps} steps...")

        batch_stats = []

        try:
            for step, (inputs, labels) in enumerate(data_loader.get_batch_iterator()):
                if step >= max_steps:
                    break

                stats = self.scan_batch(inputs, labels)
                batch_stats.append(stats)

                if (step + 1) % 100 == 0:
                    print(f"Step {step + 1}/{max_steps}, Loss: {stats['loss']:.4f}, Max Grad: {stats['max_gradient']:.6f}")

        except KeyboardInterrupt:
            print(f"Scan interrupted at step {step + 1}")

        # Compute summary statistics
        avg_loss = np.mean([s['loss'] for s in batch_stats])
        avg_max_grad = np.mean([s['max_gradient'] for s in batch_stats])

        return {
            'running_max': self.running_max,
            'batch_stats': batch_stats,
            'total_steps': self.step_count,
            'avg_loss': avg_loss,
            'avg_max_gradient': avg_max_grad
        }

    def get_sensitivity_rankings(self, exclude_embeddings: bool = False,
                               exclude_layernorm: bool = False) -> List[Tuple[str, Tuple, float]]:
        """
        Get ranked list of most sensitive parameters.

        Args:
            exclude_embeddings: Whether to exclude embedding layers
            exclude_layernorm: Whether to exclude layer normalization weights

        Returns:
            List of (param_name, element_index, gradient_magnitude) tuples sorted by magnitude
        """
        sensitivities = []

        for name, grad_max in self.running_max.items():
            # Apply exclusion filters
            if exclude_embeddings and ('embed' in name.lower() or 'wte' in name.lower()):
                continue
            if exclude_layernorm and ('norm' in name.lower() or 'ln' in name.lower()):
                continue

            # Flatten tensor and get indices
            flat_grad = grad_max.flatten()
            flat_indices = torch.nonzero(flat_grad, as_tuple=False).squeeze()

            if flat_indices.numel() == 0:
                continue

            # Get original tensor shape for index mapping
            original_shape = grad_max.shape

            for flat_idx in flat_indices:
                # Convert flat index back to multi-dimensional index
                if len(original_shape) == 1:
                    element_idx = (flat_idx.item(),)
                else:
                    element_idx = np.unravel_index(flat_idx.item(), original_shape)

                magnitude = flat_grad[flat_idx].item()
                sensitivities.append((name, element_idx, magnitude))

        # Sort by magnitude (descending)
        sensitivities.sort(key=lambda x: x[2], reverse=True)
        return sensitivities

    def get_top_k_sensitive(self, k: int = 10, **kwargs) -> List[Tuple[str, Tuple, float]]:
        """Get top-k most sensitive parameters."""
        rankings = self.get_sensitivity_rankings(**kwargs)
        return rankings[:k]

    def print_sensitivity_report(self, top_k: int = 10, exclude_embeddings: bool = False):
        """Print a formatted report of the most sensitive parameters."""
        print(f"\n{'='*80}")
        print(f"GRADIENT-BASED SENSITIVITY ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"Total steps processed: {self.step_count}")
        print(f"Total parameters tracked: {sum(p.numel() for p in self.param_dict.values()):,}")

        # All parameters
        print(f"\nTop-{top_k} Most Sensitive Parameters (All):")
        print(f"{'Rank':<5} {'Parameter Name':<50} {'Element':<20} {'|∇L|':<12}")
        print("-" * 90)

        top_all = self.get_top_k_sensitive(top_k, exclude_embeddings=False, exclude_layernorm=False)
        for rank, (name, element, magnitude) in enumerate(top_all, 1):
            element_str = str(element) if len(str(element)) < 18 else str(element)[:15] + "..."
            print(f"{rank:<5} {name:<50} {element_str:<20} {magnitude:<12.3e}")

        if exclude_embeddings:
            print(f"\nTop-{top_k} Most Sensitive Parameters (Excluding Embeddings & LayerNorm):")
            print(f"{'Rank':<5} {'Parameter Name':<50} {'Element':<20} {'|∇L|':<12}")
            print("-" * 90)

            top_filtered = self.get_top_k_sensitive(top_k, exclude_embeddings=True, exclude_layernorm=True)
            for rank, (name, element, magnitude) in enumerate(top_filtered, 1):
                element_str = str(element) if len(str(element)) < 18 else str(element)[:15] + "..."
                print(f"{rank:<5} {name:<50} {element_str:<20} {magnitude:<12.3e}")

    def save_results(self, filepath: str):
        """Save gradient scanning results to file."""
        results = {
            'running_max': {name: tensor.cpu().numpy() for name, tensor in self.running_max.items()},
            'step_count': self.step_count,
            'param_shapes': {name: list(param.shape) for name, param in self.param_dict.items()}
        }
        torch.save(results, filepath)
        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str):
        """Load gradient scanning results from file."""
        results = torch.load(filepath, map_location='cpu')
        self.running_max = {name: torch.tensor(arr) for name, arr in results['running_max'].items()}
        self.step_count = results['step_count']
        print(f"Results loaded from {filepath}")


if __name__ == "__main__":
    # Test the gradient scanner with a small model
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Testing gradient scanner...")

    # Load a small model for testing
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize scanner
    scanner = GradientScanner(model)

    # Create sample data
    data_loader = DataLoader(seq_len=128, batch_size=2)
    data_loader.set_tokenizer(tokenizer)

    # Run short scan
    results = scanner.scan_dataset(data_loader, max_steps=5)

    # Print results
    scanner.print_sensitivity_report(top_k=5, exclude_embeddings=True)

    print("\nGradient scanner test completed!")