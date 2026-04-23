# llm-sensitivity

Research code for **gradient-based sensitivity analysis of large language models** under bit-level soft errors. Implements the paper *"Gradient-Based Sensitivity Analysis in Large Language Models"* (see [`docs/main.tex`](docs/main.tex)): stream WikiText-103 through a causal LM, identify the top-K scalars with the largest `|∂L/∂θ|`, flip single bits in their FP32 representation, and score the resulting clean-vs-corrupt generations with edit distance / BLEU / METEOR / BERTScore / ROUGE.

## Quick start

```bash
# Local (defaults reproduce the paper: GPT-2, SEQ_LEN=1024, BATCH_SIZE=16, MAX_STEPS=1100)
python3 sensitivity_pipeline.py --top-k 3 --v-select all --n-trials-per-class 5

# Colab (add --install-deps on first run)
python3 sensitivity_pipeline.py --install-deps --top-k 3 --v-select all
```

Outputs land in `bitflip_outputs/`: `topk_sensitive.csv`, `bitflip_per_trial.csv`, `bitflip_aggregated.csv`, `classification_summary.txt`. On Colab the directory is also mirrored to `/content/drive/MyDrive/bitflip_outputs/`.

## Repository layout

### Root (current reference implementation)

- **`sensitivity_pipeline.py`** — Single unified, paper-aligned pipeline. Consolidates every notebook and every `docs/*.py` module into one runnable script. Implements: WikiText-103 windowing (Algorithm A.1), CPU-accumulated gradient scan with `running_max[name] = max(running_max[name], |grad|)` (Algorithm B.1), global top-K with optional embedding / layer-norm exclusion, systematic bit-class flips (sign / exponent / mantissa) with `finally:`-guaranteed weight restoration, guarded corrupt decoding (`NanInfClampProcessor` + `MaxConsecutiveRepeatProcessor` + `no_repeat_ngram_size=3`), full metric suite, Preserved/Changed/Gibberish classification on the paper's BERTScore-F1 thresholds (`>0.87` / `0.80–0.87` / `<0.80`), and per-trial + aggregated CSV output with Colab Drive mirroring. Exposes both a `Config` dataclass and an argparse CLI. **Use this for new experiments.**
- **`sensitivity_pipeline.ipynb`** — Cell-by-cell Jupyter / Colab rendering of `sensitivity_pipeline.py`, generated mechanically from the script. Code is byte-identical aside from two required notebook adaptations: `_SCRIPT_DIR` has a `try/except NameError` fallback to `os.getcwd()` (because `__file__` is undefined in a kernel), and the `if __name__ == "__main__": run(parse_args())` tail is replaced with a direct `cfg = Config(install_deps=True); run(cfg)` cell (because `argparse` would try to parse the kernel's `sys.argv` otherwise). Open in Colab and run all cells; set `install_deps=True` on first run, then `False` on reruns. **Keep this in sync with `sensitivity_pipeline.py` — re-regenerate from the `.py` rather than hand-editing cells.**
- **`CLAUDE.md`** — Instructions for Claude Code sessions in this repo. Full paper context, file-by-file docs for `docs/`, stage-by-stage walkthrough of `sensitivity_pipeline.py`, and a list of invariants (coords as Python ints, `flip_bit` requires float32, always restore in `finally:`, mandatory logits guards, best-effort Drive mirror).
- **`README.md`** — This file.
- **`LICENSE`** — Repository license.
- **`.gitignore`** — Ignores `push-code.sh` and `.github_token`.
- **`.claude/`** — Claude Code project configuration. `settings.json` registers a `Stop` hook (`.claude/hooks/docs-sync-reminder.sh`) that reads `git status --porcelain` at the end of every session; if anything other than `CLAUDE.md` / `README.md` has changed, it emits a blocking reminder to update both docs before stopping. A per-session sentinel in `/tmp/claude-docs-reminder-<session_id>` keeps it from firing more than once per session. `settings.local.json` is personal-scope and holds per-user permission allows.

### `archive/` — legacy notebooks and extracted scripts (superseded by `sensitivity_pipeline.py`, kept for reference)

- **`archive/Final.ipynb`** — Historical Colab-oriented end-to-end pipeline. Largest and most complete notebook; the code in `sensitivity_pipeline.py` is derived from its cells. Writes CSVs to `/content/drive/MyDrive/bitflip_outputs`.
- **`archive/Grad_Sensitivity.ipynb`** — Repo-friendly variant of `Final.ipynb`. Writes per-trial and aggregated CSVs to a local `bitflip_outputs/` directory (same location the unified script uses).
- **`archive/Top-k Grad.ipynb`** — Gradient scan only; reports the top-K most sensitive scalars without running the bit-flip trials.
- **`archive/gpt2_prompt_runner.ipynb`** — Minimal GPT-2 generation playground for probing prompts.
- **`archive/Final_extracted.py`** — Flat dump of every code cell from `Final.ipynb`. **Not runnable as a script** — concatenates multiple experimental blocks that redefine the same names. Useful for grepping function definitions without opening the notebook. Treat as read-only reference; edit the notebook or `sensitivity_pipeline.py` instead.

### Prior-literature references (PDFs with no extension)

- **`Cross-Entropy Loss Sensitivity`**, **`Related Codes`**, **`Soft Error Propagation in DNNs`** — Literature references consulted while developing the framework. Consult when adding new sensitivity or soft-error methods.

### `docs/` sub-folder

Companion to the research paper: LaTeX source, candidate prompts, and a modular paper-aligned Python reference implementation that mirrors each algorithm box in the paper as its own module.

- **`docs/main.tex`** — The paper. Section 1 formalizes per-parameter sensitivity as a first-order Taylor approximation, $S_i(\delta) \approx \delta \cdot g_i$. Section 2 describes the corpus → windows → batches → forward → loss → backward → running-max procedure. Section 3 reports the headline GPT-2 results: top-10 overall dominated by `transformer.wte.weight` (most-sensitive scalar at `(2488, 496)`, `|∇L| ≈ 5.24`); top-10 excluding embeddings / layer-norm dominated by `transformer.h.5.mlp.c_fc.weight` column 1866; 225-trial bit-flip classification split of 95.6% Gibberish / 2.67% Changed / 1.78% Preserved. Appendices A and B contain Algorithms A.1 (tokenization / batching) and B.1 (gradient scan) with reference Python snippets; Appendix C lists top-10 sensitive weights for Deepseek-R1-Distill-Qwen-1.5B.
- **`docs/README.md`** — Human-readable project overview mirroring `main.tex`. Lists expected paper results, CLI examples for `docs/main_simulation.py`, and the output directory layout for the modular implementation.
- **`docs/requirements.txt`** — Minimal dependency pin for the `docs/*.py` modules: `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`, `seaborn`, `pandas`, `nltk`, `bert-score`, `scipy`. `sensitivity_pipeline.py` additionally needs `evaluate`, `sacrebleu`, `rouge-score`, `tabulate`.
- **`docs/candidate-prompts.txt`** — Reference list of candidate test prompts considered for the paper. The paper selected five: *"The weather today is"*, *"The patient should take"*, *"The bank transfer amount is"*, *"The recommended dose for a child is"*, *"The evacuation order status is"*.
- **`docs/model_config.py`** — `ModelConfig` class. Loads HuggingFace causal LMs (GPT-2, GPT-Neo, DistilBERT), auto-detects device (cuda / mps / cpu), classifies parameters into `embedding / layer_norm / attention / mlp / output_head / other`, prints per-layer stats, and exposes `get_recommended_config("small"/"medium"/"large")` presets.
- **`docs/data_processing.py`** — Algorithm A.1 as a module. `chunk_generator()`, `get_batches()`, and a `DataLoader` wrapper around `datasets.load_dataset` with a `set_tokenizer` / `get_batch_iterator` API.
- **`docs/gradient_scanner.py`** — Algorithm B.1 as a class. `GradientScanner` tracks `running_max` per parameter, exposes `scan_batch` / `scan_dataset`, `get_sensitivity_rankings(exclude_embeddings, exclude_layernorm)`, `get_top_k_sensitive`, `print_sensitivity_report`, and `save_results` / `load_results` via `torch.save`.
- **`docs/bit_flip_simulator.py`** — Reference bit-flip module. Uses `struct.pack/unpack` for the bit-flip (paper-equivalent to the numpy-uint32 view in `sensitivity_pipeline.py`). Provides `save_original_state` / `restore_original_state` via `state_dict` snapshot, per-element `corrupt_parameter_element`, and a `run_bit_flip_experiment` driver. **Note:** this module's text-generation helper uses `do_sample=True`, while the paper and `sensitivity_pipeline.py` use greedy decoding — expect different outputs.
- **`docs/output_evaluation.py`** — `OutputEvaluator`. BERTScore-F1 wrapper (with a Jaccard/bigram fallback when `bert_score` is unavailable), paper classification (Preserved/Changed/Gibberish), per-text structural analysis (repetition ratio, comma density, average word length, etc.), experiment-level aggregation, and `compare_with_paper_results` which checks percentages against the paper's 1.78/2.67/95.6 split.
- **`docs/sensitivity_analysis.py`** — `SensitivityAnalyzer`. Layer-distribution stats, parameter-pattern stats, per-tensor sensitivity heatmaps (matplotlib/seaborn), and a three-panel dashboard: top-K sensitivity curve, layer-type pie+bar, magnitude histogram. Exports results as JSON.
- **`docs/main_simulation.py`** — End-to-end driver tying the seven `docs/*.py` modules together. Produces a timestamped `results_YYYYMMDD_HHMMSS/` directory with `model_info.json`, `gradient_scan_results.pt`, `sensitivity_analysis.json`, `bit_flip_results.json`, `evaluation_results.json`, `evaluation_report.txt`, `paper_comparison.json`, `simulation_summary.json`, and a `plots/` sub-directory. Superseded by `sensitivity_pipeline.py` for new work, but useful when you need the per-layer visual dashboard.

## Which entry point to use

| Goal | Use |
| --- | --- |
| Reproduce the paper end-to-end, get CSVs | **`sensitivity_pipeline.py`** (repo root) |
| Run the same pipeline in Colab | **`sensitivity_pipeline.ipynb`** (open in Colab; set `install_deps=True` on first run) |
| Read the theory and algorithm proofs | `docs/main.tex` |
| Browse the paper-aligned modular reference | `docs/main_simulation.py` + `docs/*.py` |
| Generate per-layer sensitivity plots / heatmaps | `docs/main_simulation.py` (has matplotlib dashboard) |
| Gradient scan only, no bit-flip trials | `archive/Top-k Grad.ipynb` or `sensitivity_pipeline.py` with `--v-select 0` |
| Quick GPT-2 prompt probing | `archive/gpt2_prompt_runner.ipynb` |

## Paper highlights

- Gradient magnitude $|g_i|$ is a first-order proxy for parameter-level soft-error sensitivity: $S_i(\delta) \approx \delta \cdot g_i$.
- On GPT-2 small over ~18M WikiText-103 tokens, the top-10 sensitive scalars are all `transformer.wte.weight`; once embeddings and layer-norm are excluded, `transformer.h.5.mlp.c_fc.weight` column 1866 dominates.
- Single-bit flips in the top-3 sensitive parameters produce gibberish output 95.6% of the time, intelligibly-changed output 2.67% of the time, and preserved meaning only 1.78% of the time (225 trials, BERTScore-F1 classification with thresholds 0.87 / 0.80).

## License

See [`LICENSE`](LICENSE).
