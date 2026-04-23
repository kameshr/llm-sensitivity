"""
Model Configuration Module for Gradient-Based Sensitivity Analysis

Handles loading and configuration of various transformer models for sensitivity analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    GPTNeoForCausalLM, GPTNeoTokenizer,
    DistilBertModel, DistilBertTokenizer
)


class ModelConfig:
    """Configuration class for managing different model setups."""

    SUPPORTED_MODELS = {
        'gpt2': {
            'model_class': GPT2LMHeadModel,
            'tokenizer_class': GPT2Tokenizer,
            'variants': ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
            'task': 'causal_lm'
        },
        'gpt-neo': {
            'model_class': GPTNeoForCausalLM,
            'tokenizer_class': GPTNeoTokenizer,
            'variants': ['EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B'],
            'task': 'causal_lm'
        },
        'distilbert': {
            'model_class': DistilBertModel,
            'tokenizer_class': DistilBertTokenizer,
            'variants': ['distilbert-base-uncased'],
            'task': 'masked_lm'
        }
    }

    def __init__(self, model_name: str = 'gpt2', device: str = 'auto'):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for computation."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device

    def load_model(self, model_variant: str = None, **kwargs) -> Tuple[nn.Module, AutoTokenizer]:
        """
        Load a model and tokenizer for sensitivity analysis.

        Args:
            model_variant: Specific model variant to load (e.g., 'gpt2', 'gpt2-medium')
            **kwargs: Additional arguments for model loading

        Returns:
            (model, tokenizer): Loaded model and tokenizer
        """
        if model_variant is None:
            # Use the base model name
            if self.model_name in ['gpt2', 'gpt-neo', 'distilbert']:
                model_variant = self.SUPPORTED_MODELS[self.model_name]['variants'][0]
            else:
                model_variant = self.model_name

        print(f"Loading model: {model_variant}")
        print(f"Device: {self.device}")

        try:
            # Load using AutoModel classes for flexibility
            self.model = AutoModelForCausalLM.from_pretrained(
                model_variant,
                torch_dtype=torch.float32,  # Use float32 for gradient computation
                device_map=None,  # We'll move to device manually
                **kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_variant)

            # Set up tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Move model to device
            self.model = self.model.to(self.device)

            # Ensure model is in training mode for gradient computation
            self.model.train()

            print(f"Model loaded successfully: {model_variant}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

            return self.model, self.tokenizer

        except Exception as e:
            print(f"Error loading model {model_variant}: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}

        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Analyze parameter distribution by layer type
        layer_analysis = {}
        for name, param in self.model.named_parameters():
            layer_type = self._classify_parameter(name)
            if layer_type not in layer_analysis:
                layer_analysis[layer_type] = {'count': 0, 'params': 0}
            layer_analysis[layer_type]['count'] += 1
            layer_analysis[layer_type]['params'] += param.numel()

        return {
            'model_name': self.model_name,
            'device': self.device,
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'model_config': self.model.config.to_dict() if hasattr(self.model, 'config') else {},
            'layer_analysis': layer_analysis
        }

    def _classify_parameter(self, param_name: str) -> str:
        """Classify a parameter by its layer type."""
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

    def print_model_summary(self):
        """Print a detailed summary of the model architecture."""
        if self.model is None:
            print("No model loaded")
            return

        info = self.get_model_info()

        print(f"\n{'='*60}")
        print(f"MODEL SUMMARY: {info['model_name']}")
        print(f"{'='*60}")
        print(f"Device: {info['device']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")

        print(f"\nParameter Distribution by Layer Type:")
        print(f"{'Layer Type':<15} {'Count':<8} {'Parameters':<12} {'Percentage':<10}")
        print("-" * 50)

        total_params = info['total_parameters']
        for layer_type, stats in info['layer_analysis'].items():
            percentage = (stats['params'] / total_params) * 100
            print(f"{layer_type:<15} {stats['count']:<8} {stats['params']:<12,} {percentage:<10.2f}%")

        if hasattr(self.model, 'config'):
            config = self.model.config
            print(f"\nModel Configuration:")
            relevant_config = {
                'vocab_size': getattr(config, 'vocab_size', 'N/A'),
                'hidden_size': getattr(config, 'hidden_size', 'N/A'),
                'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'N/A')),
                'num_attention_heads': getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'N/A')),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 'N/A'))
            }

            for key, value in relevant_config.items():
                print(f"  {key}: {value}")

    def get_layer_parameters(self, layer_type: str = None) -> Dict[str, torch.Tensor]:
        """
        Get parameters filtered by layer type.

        Args:
            layer_type: Type of layer to filter ('embedding', 'attention', 'mlp', etc.)

        Returns:
            Dictionary of parameter names and tensors
        """
        if self.model is None:
            return {}

        filtered_params = {}
        for name, param in self.model.named_parameters():
            if layer_type is None or self._classify_parameter(name) == layer_type:
                filtered_params[name] = param

        return filtered_params


def get_recommended_config(model_size: str = 'small') -> Dict[str, Any]:
    """Get recommended configuration based on desired model size."""
    configs = {
        'small': {
            'model_name': 'gpt2',
            'model_variant': 'gpt2',
            'seq_len': 512,
            'batch_size': 16,
            'max_steps': 1000
        },
        'medium': {
            'model_name': 'gpt2',
            'model_variant': 'gpt2-medium',
            'seq_len': 1024,
            'batch_size': 8,
            'max_steps': 500
        },
        'large': {
            'model_name': 'gpt-neo',
            'model_variant': 'EleutherAI/gpt-neo-125M',
            'seq_len': 1024,
            'batch_size': 4,
            'max_steps': 200
        }
    }

    return configs.get(model_size, configs['small'])


if __name__ == "__main__":
    # Test the model configuration
    print("Testing model configuration...")

    config = ModelConfig('gpt2')
    model, tokenizer = config.load_model('gpt2')

    config.print_model_summary()

    # Test parameter classification
    print(f"\nEmbedding parameters:")
    embed_params = config.get_layer_parameters('embedding')
    for name in list(embed_params.keys())[:3]:
        print(f"  {name}: {embed_params[name].shape}")

    print(f"\nAttention parameters:")
    attn_params = config.get_layer_parameters('attention')
    for name in list(attn_params.keys())[:3]:
        print(f"  {name}: {attn_params[name].shape}")

    print("Model configuration test completed!")