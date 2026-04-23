"""
Bit-Flip Simulation Module for Sensitivity Analysis

Implements controlled bit-flip operations on sensitive parameters to test model robustness.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import struct
import copy


class BitFlipSimulator:
    """
    Simulates bit flips in model parameters to test sensitivity and robustness.

    Supports flipping different parts of IEEE 754 float32 representation:
    - Sign bit (1 bit)
    - Exponent bits (8 bits)
    - Mantissa bits (23 bits)
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.original_state = None
        self.corrupted_params = {}

    def save_original_state(self):
        """Save the original model state for restoration."""
        self.original_state = copy.deepcopy(self.model.state_dict())

    def restore_original_state(self):
        """Restore the model to its original uncorrupted state."""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state)
            self.corrupted_params = {}

    def float32_to_bits(self, value: float) -> str:
        """Convert float32 to 32-bit binary representation."""
        packed = struct.pack('>f', value)
        integer_repr = struct.unpack('>I', packed)[0]
        return format(integer_repr, '032b')

    def bits_to_float32(self, bits: str) -> float:
        """Convert 32-bit binary representation back to float32."""
        integer_repr = int(bits, 2)
        packed = struct.pack('>I', integer_repr)
        return struct.unpack('>f', packed)[0]

    def flip_bit_in_float(self, value: float, bit_position: int) -> float:
        """
        Flip a specific bit in a float32 value.

        Args:
            value: Original float32 value
            bit_position: Position of bit to flip (0-31, where 0 is LSB)

        Returns:
            Float with specified bit flipped
        """
        bits = self.float32_to_bits(value)
        bit_array = list(bits)

        # Flip the bit (convert to list index: MSB is index 0)
        array_index = 31 - bit_position
        bit_array[array_index] = '1' if bit_array[array_index] == '0' else '0'

        corrupted_bits = ''.join(bit_array)
        return self.bits_to_float32(corrupted_bits)

    def flip_random_bit_type(self, value: float, bit_type: str) -> float:
        """
        Flip a random bit in a specific section of the float32 representation.

        Args:
            value: Original float32 value
            bit_type: 'sign', 'exponent', or 'mantissa'

        Returns:
            Float with a random bit flipped in the specified section
        """
        if bit_type == 'sign':
            # Sign bit is at position 31 (MSB)
            return self.flip_bit_in_float(value, 31)
        elif bit_type == 'exponent':
            # Exponent bits are positions 30-23
            bit_position = np.random.randint(23, 31)
            return self.flip_bit_in_float(value, bit_position)
        elif bit_type == 'mantissa':
            # Mantissa bits are positions 22-0
            bit_position = np.random.randint(0, 23)
            return self.flip_bit_in_float(value, bit_position)
        else:
            raise ValueError(f"Unknown bit type: {bit_type}. Use 'sign', 'exponent', or 'mantissa'")

    def corrupt_parameter_element(self, param_name: str, element_index: Tuple[int, ...],
                                 bit_type: str = 'random') -> Dict[str, Any]:
        """
        Corrupt a specific element of a parameter tensor.

        Args:
            param_name: Name of the parameter to corrupt
            element_index: Multi-dimensional index of the element to corrupt
            bit_type: Type of bit to flip ('sign', 'exponent', 'mantissa', or 'random')

        Returns:
            Dictionary with corruption details
        """
        # Get the parameter
        param_dict = dict(self.model.named_parameters())
        if param_name not in param_dict:
            raise ValueError(f"Parameter {param_name} not found in model")

        param = param_dict[param_name]

        # Save original value
        original_value = param.data[element_index].item()

        # Choose random bit type if specified
        if bit_type == 'random':
            bit_type = np.random.choice(['sign', 'exponent', 'mantissa'])

        # Corrupt the value
        corrupted_value = self.flip_random_bit_type(original_value, bit_type)

        # Update the parameter
        with torch.no_grad():
            param.data[element_index] = corrupted_value

        # Record the corruption
        corruption_key = f"{param_name}[{element_index}]"
        self.corrupted_params[corruption_key] = {
            'param_name': param_name,
            'element_index': element_index,
            'original_value': original_value,
            'corrupted_value': corrupted_value,
            'bit_type': bit_type,
            'magnitude_change': abs(corrupted_value - original_value),
            'relative_change': abs(corrupted_value - original_value) / (abs(original_value) + 1e-10)
        }

        return self.corrupted_params[corruption_key]

    def corrupt_top_sensitive_parameters(self, sensitive_params: List[Tuple[str, Tuple, float]],
                                       num_corruptions: int = 1,
                                       bit_type: str = 'random') -> List[Dict[str, Any]]:
        """
        Corrupt the top-k most sensitive parameters.

        Args:
            sensitive_params: List of (param_name, element_index, sensitivity) tuples
            num_corruptions: Number of parameters to corrupt
            bit_type: Type of bit to flip

        Returns:
            List of corruption details
        """
        if self.original_state is None:
            self.save_original_state()

        corruptions = []
        num_corruptions = min(num_corruptions, len(sensitive_params))

        for i in range(num_corruptions):
            param_name, element_index, sensitivity = sensitive_params[i]
            corruption_details = self.corrupt_parameter_element(param_name, element_index, bit_type)
            corruption_details['sensitivity_rank'] = i + 1
            corruption_details['original_sensitivity'] = sensitivity
            corruptions.append(corruption_details)

        return corruptions

    def generate_text_comparison(self, tokenizer, prompt: str, max_length: int = 100,
                               num_samples: int = 1) -> Dict[str, Any]:
        """
        Generate text with both clean and corrupted models for comparison.

        Args:
            tokenizer: Tokenizer for encoding/decoding text
            prompt: Input prompt for generation
            max_length: Maximum length of generated text
            num_samples: Number of samples to generate

        Returns:
            Dictionary with clean and corrupted outputs
        """
        results = {
            'prompt': prompt,
            'clean_outputs': [],
            'corrupted_outputs': [],
            'corruptions_applied': list(self.corrupted_params.values())
        }

        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Generate with corrupted model
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                corrupted_output = self.model.generate(
                    inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
                corrupted_text = tokenizer.decode(corrupted_output[0], skip_special_tokens=True)
                results['corrupted_outputs'].append(corrupted_text)

        # Restore original model and generate clean outputs
        self.restore_original_state()

        with torch.no_grad():
            for _ in range(num_samples):
                clean_output = self.model.generate(
                    inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
                clean_text = tokenizer.decode(clean_output[0], skip_special_tokens=True)
                results['clean_outputs'].append(clean_text)

        self.model.train()
        return results

    def run_bit_flip_experiment(self, sensitive_params: List[Tuple[str, Tuple, float]],
                              tokenizer, test_prompts: List[str],
                              num_corruptions: int = 3, trials_per_prompt: int = 5,
                              bit_types: List[str] = ['sign', 'exponent', 'mantissa']) -> Dict[str, Any]:
        """
        Run a comprehensive bit-flip experiment.

        Args:
            sensitive_params: List of sensitive parameters
            tokenizer: Tokenizer for text generation
            test_prompts: List of test prompts
            num_corruptions: Number of top parameters to corrupt
            trials_per_prompt: Number of trials per prompt per bit type
            bit_types: Types of bits to test

        Returns:
            Comprehensive experiment results
        """
        experiment_results = {
            'test_prompts': test_prompts,
            'num_corruptions': num_corruptions,
            'trials_per_prompt': trials_per_prompt,
            'bit_types': bit_types,
            'results': []
        }

        total_trials = len(test_prompts) * trials_per_prompt * len(bit_types)
        trial_count = 0

        print(f"Running bit-flip experiment with {total_trials} total trials...")

        for prompt_idx, prompt in enumerate(test_prompts):
            for bit_type in bit_types:
                for trial in range(trials_per_prompt):
                    trial_count += 1
                    print(f"Trial {trial_count}/{total_trials}: Prompt {prompt_idx+1}, {bit_type}, Trial {trial+1}")

                    # Apply corruptions
                    corruptions = self.corrupt_top_sensitive_parameters(
                        sensitive_params, num_corruptions, bit_type
                    )

                    # Generate comparison
                    comparison = self.generate_text_comparison(
                        tokenizer, prompt, max_length=100, num_samples=1
                    )

                    # Store results
                    trial_result = {
                        'prompt_index': prompt_idx,
                        'prompt': prompt,
                        'bit_type': bit_type,
                        'trial': trial,
                        'corruptions': corruptions,
                        'clean_output': comparison['clean_outputs'][0],
                        'corrupted_output': comparison['corrupted_outputs'][0]
                    }

                    experiment_results['results'].append(trial_result)

                    # Restore for next trial
                    self.restore_original_state()

        print("Bit-flip experiment completed!")
        return experiment_results

    def print_corruption_summary(self):
        """Print a summary of current corruptions."""
        if not self.corrupted_params:
            print("No corruptions currently applied.")
            return

        print(f"\n{'='*60}")
        print(f"CURRENT CORRUPTIONS SUMMARY")
        print(f"{'='*60}")
        print(f"Total corrupted parameters: {len(self.corrupted_params)}")

        for key, corruption in self.corrupted_params.items():
            print(f"\nParameter: {corruption['param_name']}")
            print(f"Element: {corruption['element_index']}")
            print(f"Bit type: {corruption['bit_type']}")
            print(f"Original: {corruption['original_value']:.6e}")
            print(f"Corrupted: {corruption['corrupted_value']:.6e}")
            print(f"Magnitude change: {corruption['magnitude_change']:.6e}")
            print(f"Relative change: {corruption['relative_change']:.4f}")


def analyze_bit_flip_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the results of a bit-flip experiment.

    Args:
        results: Results from run_bit_flip_experiment

    Returns:
        Analysis summary
    """
    total_trials = len(results['results'])

    # Count outcomes by bit type
    outcomes_by_bit_type = {}
    for bit_type in results['bit_types']:
        outcomes_by_bit_type[bit_type] = {
            'total': 0,
            'outputs': []
        }

    for trial in results['results']:
        bit_type = trial['bit_type']
        outcomes_by_bit_type[bit_type]['total'] += 1
        outcomes_by_bit_type[bit_type]['outputs'].append({
            'clean': trial['clean_output'],
            'corrupted': trial['corrupted_output'],
            'prompt': trial['prompt']
        })

    analysis = {
        'total_trials': total_trials,
        'outcomes_by_bit_type': outcomes_by_bit_type,
        'summary': f"Analyzed {total_trials} bit-flip trials across {len(results['bit_types'])} bit types"
    }

    return analysis


if __name__ == "__main__":
    # Test the bit flip simulator
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Testing bit-flip simulator...")

    # Load a small model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize simulator
    simulator = BitFlipSimulator(model)

    # Create mock sensitive parameters
    mock_sensitive = [
        ('transformer.wte.weight', (100, 50), 2.5),
        ('transformer.h.0.attn.c_attn.weight', (10, 20), 1.8),
        ('transformer.h.0.mlp.c_fc.weight', (5, 15), 1.2)
    ]

    # Test basic corruption
    simulator.save_original_state()
    corruption = simulator.corrupt_parameter_element(
        'transformer.wte.weight', (100, 50), 'exponent'
    )

    print(f"Test corruption applied:")
    print(f"Original: {corruption['original_value']:.6e}")
    print(f"Corrupted: {corruption['corrupted_value']:.6e}")
    print(f"Bit type: {corruption['bit_type']}")

    # Test text generation
    test_prompts = ["The weather today is", "The patient should take"]
    comparison = simulator.generate_text_comparison(
        tokenizer, test_prompts[0], max_length=50
    )

    print(f"\nText generation test:")
    print(f"Prompt: {comparison['prompt']}")
    print(f"Clean: {comparison['clean_outputs'][0][:100]}...")
    print(f"Corrupted: {comparison['corrupted_outputs'][0][:100]}...")

    print("\nBit-flip simulator test completed!")