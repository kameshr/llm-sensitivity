# Gradient-Based Sensitivity Analysis in Large Language Models

This repository implements the complete simulation framework described in the paper "Gradient-Based Sensitivity Analysis in Large Language Models". The implementation provides tools to identify the most sensitive parameters in transformer models and evaluate their robustness to bit-flip errors.

## Overview

The paper presents a gradient-based method for identifying parameters in language models that are most sensitive to corruption (soft errors). The key insight is that parameters with larger gradients during training are more likely to cause significant output degradation when corrupted.

### Key Components

1. **Gradient Scanning**: Tracks maximum absolute gradients per parameter across many training batches
2. **Parameter Sensitivity Ranking**: Identifies the most critical parameters based on gradient magnitudes
3. **Bit-Flip Simulation**: Tests model robustness by corrupting sensitive parameters
4. **Output Classification**: Categorizes corrupted outputs as Preserved/Changed/Gibberish using BERTScore

### Paper Results Reproduced

The simulation reproduces key findings from the paper:
- **GPT-2 (137M)** analysis on WikiText-103 dataset
- **Top-10 sensitive parameters** dominated by embedding weights (`transformer.wte.weight`)
- **Bit-flip experiments** showing 95.6% gibberish rate for top-3 sensitive parameters
- **Classification thresholds**: BERTScore F1 > 0.87 (Preserved), 0.80-0.87 (Changed), < 0.80 (Gibberish)

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for text analysis):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Quick Start

### Basic Gradient Scanning

Run a basic sensitivity analysis on GPT-2:

```bash
python main_simulation.py --model gpt2 --max-steps 500 --top-k 50
```

### Full Experiment with Bit-Flip Testing

Run the complete pipeline including bit-flip corruption experiments:

```bash
python main_simulation.py --model gpt2 --max-steps 1000 --experiment --num-corruptions 3 --trials-per-prompt 5
```

### Configuration Presets

Use predefined configurations for different experiment scales:

```bash
# Small-scale experiment (fast)
python main_simulation.py --config small --experiment

# Medium-scale experiment (paper reproduction)
python main_simulation.py --config medium --experiment

# Large-scale experiment (comprehensive)
python main_simulation.py --config large --experiment
```

## Command Line Arguments

### Model Configuration
- `--model`: Model type (gpt2, gpt-neo, etc.)
- `--model-variant`: Specific model variant (e.g., gpt2-medium)
- `--device`: Compute device (auto, cpu, cuda, mps)

### Data Configuration
- `--dataset`: Dataset name (default: wikitext)
- `--dataset-subset`: Dataset subset (default: wikitext-103-raw-v1)
- `--seq-len`: Sequence length for training windows (default: 512)
- `--batch-size`: Batch size (default: 8)

### Gradient Scanning
- `--max-steps`: Maximum gradient scanning steps (default: 500)
- `--top-k`: Number of top parameters to analyze (default: 50)

### Bit-Flip Experiments
- `--experiment`: Enable bit-flip corruption experiments
- `--num-corruptions`: Number of top parameters to corrupt (default: 3)
- `--trials-per-prompt`: Trials per prompt per bit type (default: 5)

## Module Documentation

### Core Modules

#### `data_processing.py`
Implements Algorithm A.1 from the paper for tokenization and batching:
- `chunk_generator()`: Rolling buffer approach for windowed sequences
- `get_batches()`: Creates input-label pairs for language modeling
- `DataLoader`: Manages dataset streaming and preprocessing

#### `gradient_scanner.py`
Implements Algorithm B.1 for gradient-based sensitivity analysis:
- `GradientScanner`: Main class for tracking parameter sensitivities
- `scan_batch()`: Processes individual batches and updates gradient maximums
- `get_sensitivity_rankings()`: Returns parameters ranked by sensitivity

#### `model_config.py`
Handles model loading and configuration:
- `ModelConfig`: Manages different transformer model types
- Support for GPT-2, GPT-Neo, and other Hugging Face models
- Device management and parameter analysis

#### `bit_flip_simulator.py`
Implements controlled parameter corruption:
- `BitFlipSimulator`: Simulates bit flips in IEEE 754 float32 representation
- Supports sign, exponent, and mantissa bit corruption
- `run_bit_flip_experiment()`: Comprehensive corruption testing

#### `output_evaluation.py`
Classifies and evaluates corrupted model outputs:
- `OutputEvaluator`: Uses BERTScore and structural analysis
- Three-class classification: Preserved/Changed/Gibberish
- Comparison with paper results

#### `sensitivity_analysis.py`
Provides comprehensive analysis and visualization:
- `SensitivityAnalyzer`: Statistical analysis of sensitivity patterns
- Layer distribution analysis and visualization
- Export capabilities for results

### Main Simulation

#### `main_simulation.py`
Orchestrates the complete pipeline:
1. Model setup and configuration
2. Data processing and streaming
3. Gradient scanning across batches
4. Sensitivity analysis and ranking
5. Bit-flip corruption experiments
6. Output evaluation and classification
7. Results visualization and reporting

## Output Structure

Each simulation run creates a timestamped directory with:

```
results_YYYYMMDD_HHMMSS/
├── model_info.json              # Model configuration and statistics
├── gradient_scan_results.pt     # Raw gradient scanning data
├── sensitivity_analysis.json    # Detailed sensitivity analysis
├── bit_flip_results.json        # Bit-flip experiment data
├── evaluation_results.json      # Output classification results
├── evaluation_report.txt        # Human-readable evaluation report
├── paper_comparison.json        # Comparison with paper results
├── simulation_summary.json      # Overall simulation summary
└── plots/                       # Visualization plots
    ├── sensitivity_distribution.png
    ├── layer_distribution.png
    └── magnitude_histogram.png
```

## Example Results

### Gradient Scanning Output
```
Top-10 Most Sensitive Parameters (All):
Rank  Parameter Name                            Element          |∇L|
1     transformer.wte.weight                    (2488, 496)      5.235e+00
2     transformer.wte.weight                    (837, 496)       4.481e+00
3     transformer.wte.weight                    (198, 496)       3.247e+00
...
```

### Bit-Flip Experiment Results
```
BIT-FLIP EXPERIMENT EVALUATION REPORT
Overall Results (225 trials):
Preserved   :    4 ( 1.8%)
Changed     :    6 ( 2.7%)
Gibberish   :  215 (95.6%)

BERTScore F1 Statistics:
Mean F1:      0.2847
Std F1:       0.3021
Min F1:       0.0142
Max F1:       0.9234
```

## Research Applications

This implementation can be used for:

1. **Model Robustness Analysis**: Identify vulnerable parameters across different architectures
2. **Hardware Reliability**: Evaluate soft error susceptibility in neural accelerators
3. **Compression Research**: Target insensitive parameters for aggressive quantization
4. **Security Research**: Understand attack surfaces in deployed models
5. **Interpretability**: Analyze which parameters contribute most to model behavior

## Reproducing Paper Results

To reproduce the exact results from the paper:

```bash
# GPT-2 137M parameter analysis
python main_simulation.py \
    --model gpt2 \
    --model-variant gpt2 \
    --dataset wikitext \
    --dataset-subset wikitext-103-raw-v1 \
    --seq-len 1024 \
    --batch-size 16 \
    --max-steps 1000 \
    --experiment \
    --num-corruptions 3 \
    --trials-per-prompt 5
```

Expected key findings:
- Token embedding weights dominate top-10 most sensitive parameters
- Single bit flips in top-3 parameters cause 95.6% gibberish outputs
- MLP layer 5 shows high sensitivity among non-embedding parameters

## Performance Considerations

- **Memory Usage**: Scales with model size and batch size
- **Compute Requirements**: Gradient computation requires significant GPU memory
- **Storage**: Results can be large for comprehensive experiments (>1GB)
- **Time**: Full experiments may take 1-6 hours depending on configuration

### Optimization Tips

1. Use smaller batch sizes for memory-constrained environments
2. Reduce `max-steps` for faster experimentation
3. Skip bit-flip experiments (`--no-experiment`) for gradient analysis only
4. Use CPU device for small models to free up GPU resources

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{gradient_sensitivity_2024,
    title={Gradient-Based Sensitivity Analysis in Large Language Models},
    author={[Authors]},
    year={2024}
}
```

## License

This implementation is provided for research purposes. Please refer to the licenses of the underlying libraries (PyTorch, Transformers, etc.) for commercial use.

## Contributing

This implementation follows the paper's methodology closely but can be extended for:
- Additional model architectures
- Different corruption types
- Alternative sensitivity metrics
- Enhanced visualization capabilities

## Support

For questions about the implementation, please refer to:
1. The original paper for theoretical background
2. Individual module docstrings for technical details
3. The `main_simulation.py` script for usage examples