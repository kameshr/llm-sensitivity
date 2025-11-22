# llm-sensitivity
Analyzing error robustness in LLMs

## Notebooks

- `Final.ipynb` – Colab-oriented end-to-end bit-flip sensitivity experiment on GPT-2 over WikiText-103, scanning for top-K high-gradient weights, flipping FP32 bits, and logging quality metrics for clean vs corrupted outputs.
- `Grad_Sensitivity.ipynb` – Repository-friendly version of the bit-flip sensitivity experiment that saves per-trial and aggregated results under `bitflip_outputs/`, using GPT-2, WikiText-103, and multiple textual similarity metrics.
- `Top-k Grad.ipynb` – Computes the global top-K parameter coordinates with largest gradient magnitudes for GPT-2 on WikiText-103, identifying the most sensitive weights for later error-injection studies.
- `gpt2_prompt_runner.ipynb` – Lightweight GPT-2 playground that installs Hugging Face Transformers, loads a text-generation pipeline, and runs it on arbitrary prompts.
