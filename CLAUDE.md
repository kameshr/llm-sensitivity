# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research code for analyzing gradient-based parameter sensitivity in causal language models (GPT-2 small by default) under bit-level soft errors. The pipeline implements the paper *"Gradient-Based Sensitivity Analysis in Large Language Models"* (see `docs/main.tex`) end-to-end:

1. **Gradient scan** over all trainable parameters on windowed WikiText-103 batches to identify the global top-K scalars with the largest `|∂L/∂θ|`.
2. **Bit-flip injection** on those top-K weights, sampling bits systematically from sign / exponent / mantissa classes of the FP32 representation.
3. **Clean vs. corrupt text generation** from a fixed `TEST_PROMPTS` list, scored with edit distance / BLEU / METEOR / BERTScore / ROUGE (optionally BLEURT), classified Preserved / Changed / Gibberish per the paper's BERTScore-F1 thresholds, and written to per-trial + aggregated CSVs.

## Paper context (`docs/main.tex`)

The paper formalizes per-parameter sensitivity as a first-order Taylor approximation of the loss delta under perturbation:

- $S_i(\delta) = \mathcal{L}(f(x; \theta + \delta e_i), y) - \mathcal{L}(f(x; \theta), y) \approx \delta \cdot g_i$ where $g_i = \partial\mathcal{L}/\partial\theta_i$.
- The gradient magnitude $|g_i|$ is therefore a first-order proxy for sensitivity to a single-bit perturbation at index $i$.

The paper's procedure:

- Stream a tokenized corpus into rolling (L+1)-windows (Algorithm A.1 in `main.tex`).
- Split each window into input `I = [x_n ... x_{n+L-1}]` and labels `Y = [x_{n+1} ... x_{n+L}]`, stack into batches `I, Y ∈ ℕ^{B×L}`.
- Forward + cross-entropy loss + backward, then `running_max[name] = max(running_max[name], |grad|)` (Algorithm B.1). Accumulated across `MAX_STEPS` batches.
- The global top-K `running_max` entries are the candidates for bit-flip testing.

Reference headline results (GPT-2 small, WikiText-103, SEQ_LEN=1024, BATCH_SIZE=16, MAX_STEPS=1100, ~18M tokens):

- **Top-10 overall** is dominated by `transformer.wte.weight` (token-embedding). The single most-sensitive scalar sits at `(2488, 496)` with `|∇L| ≈ 5.24e0`.
- **Top-10 excluding embeddings and layer-norm** is dominated by `transformer.h.5.mlp.c_fc.weight` with column 1866 appearing repeatedly; the most-sensitive element is `(393, 1866)` with `|∇L| ≈ 1.34e0`.
- **Bit-flip outcomes** on the top-3 parameters (5 trials per {sign, exponent, mantissa}, 5 prompts, `max_new_tokens=100`): out of 225 outputs, 95.6% Gibberish, 2.67% Changed, 1.78% Preserved. Classification uses BERTScore-F1: `>0.87` preserved, `0.80–0.87` changed, `<0.80` gibberish.
- A secondary appendix lists top-10 sensitive weights for Deepseek-R1-Distill-Qwen-1.5B; MLP gate_proj and lm_head dominate.

The paper's five canonical test prompts (match these when reproducing):

```
"The weather today is"
"The patient should take"
"The bank transfer amount is"
"The recommended dose for a child is"
"The evacuation order status is"
```

## Current unified implementation (`sensitivity_pipeline.py` + `sensitivity_pipeline.ipynb`)

`sensitivity_pipeline.py` at the repo root is the single current reference implementation. It consolidates all notebooks and the `docs/*.py` modules into one runnable script. Use this file for new experiments; the notebooks and `docs/*.py` stay for historical context and paper-report alignment.

`sensitivity_pipeline.ipynb` is the Colab-executable twin of `sensitivity_pipeline.py`, generated mechanically from the script. **It is not hand-edited** — if you change the `.py`, re-generate the `.ipynb` instead of patching cells. The only deliberate deltas between the two are: (1) `_SCRIPT_DIR` is wrapped in `try/except NameError` falling back to `os.getcwd()` (because `__file__` is undefined in a kernel); (2) the `if __name__ == "__main__": run(parse_args())` tail is replaced with a direct `cfg = Config(install_deps=True); run(cfg)` cell (because `argparse` would consume the kernel's `sys.argv`). Every function body, class, and constant is otherwise byte-identical; `# === N) Title ===` separators become `## N) Title` markdown cells and the module docstring becomes the opening markdown cell.

Run locally:

```bash
python3 sensitivity_pipeline.py --model-id gpt2 --max-steps 1100 --top-k 50 \
    --v-select 1,2,3 --n-trials-per-class 5 --max-new-tokens 100
```

Run on Colab — open `sensitivity_pipeline.ipynb`, Runtime → Run all. The final cell is equivalent to:

```python
from sensitivity_pipeline import Config, run
cfg = Config(max_steps=1100, top_k=3, v_select="all", install_deps=True)
result = run(cfg)
```

### Pipeline stages (implemented in `sensitivity_pipeline.py`)

1. **Config + CLI** — `Config` dataclass with every knob; `parse_args()` maps the CLI to `Config`.
2. **Data** — `chunk_generator()` / `get_batches()` implement Algorithm A.1. Tokens are streamed into a rolling buffer, (L+1)-windows are cut, split into `(input, target)` pairs, and stacked into `[B, L]` batches on the target device.
3. **Gradient scan** — `gradient_scan()` implements Algorithm B.1. `running_max` is accumulated on **CPU** (`torch.zeros_like(p, device="cpu")`), and each batch's `|grad|` is moved to CPU before the `torch.maximum` reduction. This is the only way to fit the full-parameter running max for larger models.
4. **Top-K selection** — `global_topk()` takes a per-tensor `torch.topk`, then sorts globally to produce a list of `(|grad|, param_name, coord_tuple)` entries. Coordinates are immediately converted to tuples of Python ints (see invariants below). A `exclude_embeddings` / `exclude_layernorm` variant reproduces the paper's secondary table.
5. **V_SELECT** — `normalize_v_select()` expands `"all"`, a single int, or a list of ints into the 1-indexed ranks to actually flip.
6. **Bit-flip** — `flip_bit(val_tensor, bit)` performs the XOR via `numpy.uint32` view (`val.detach().cpu().numpy().copy().view(np.uint32) ^= 1<<bit`, then view back to `float32` and back to the original device). Raises on non-float32. `BIT_CLASSES = {"sign":[31], "exponent":[23..30], "mantissa":[0..22]}`.
7. **Guarded corrupt decoding** — `_build_logits_processors()` builds `NanInfClampProcessor` (replaces NaN/Inf, clamps to `[-80, 80]`, sets `had_nan` flag) and `MaxConsecutiveRepeatProcessor` (bans a token after 6 consecutive repeats). The corrupt decode path wraps these in a `LogitsProcessorList` plus `no_repeat_ngram_size=3`. **Required** — unguarded corrupt decoding routinely emits NaN logits and degenerate loops.
8. **Generation** — `Generator_` wraps both paths. The clean path uses plain greedy decoding; a clean cache `{prompt: text}` is built once at the start of the trial loop. Top-k and top-p sampling are available but the paper uses greedy.
9. **Trial loop** — for each rank in `V_SELECT` × each bit-class × `N_TRIALS_PER_CLASS`: save `orig_val`, XOR a randomly chosen bit in that class, generate corrupt continuations for every prompt, score against the cached clean continuation, classify, append a row, and **always restore the original value in a `finally:` block** before the next trial.
10. **Scoring** — `MetricSuite.score_pair()` computes edit-distance (local Levenshtein DP, no external dep), BLEU (sacrebleu), METEOR / BERTScore-F1 / ROUGE-{1,2,L} (`evaluate` library), optionally BLEURT. Each metric is in its own try/except so one failure doesn't poison a trial.
11. **Classification** — `classify_output(f1, cfg)` applies the paper thresholds: F1 > 0.87 → `preserved`, 0.80 < F1 ≤ 0.87 → `changed`, else → `gibberish`, and `unknown` if BERTScore was disabled/failed.
12. **Output** — `write_outputs()` emits:
    - `bitflip_outputs/topk_sensitive.csv` — global top-K ranking.
    - `bitflip_outputs/bitflip_per_trial.csv` — one row per (rank, tensor, coord, bit_class, bit_index, trial, prompt).
    - `bitflip_outputs/bitflip_aggregated.csv` — mean/median/std per `(rank, tensor, coord, bit_class, prompt)` for every metric present.
    - `bitflip_outputs/classification_summary.txt` — paper-style Preserved/Changed/Gibberish table with BERTScore-F1 summary stats.
    - All four files are mirrored to `/content/drive/MyDrive/bitflip_outputs/` when `google.colab` is importable; the copy is wrapped in try/except so local runs don't fail.

### Key knobs (`Config` defaults match the paper)

| Field | Default | What it controls |
| --- | --- | --- |
| `model_id` | `"gpt2"` | Any HuggingFace causal LM. The paper also reports Deepseek-R1-Distill-Qwen-1.5B. |
| `seq_len`, `batch_size`, `max_steps` | 1024, 16, 1100 | Gradient scan cost and token coverage (~18M tokens ≈ 18% of WikiText-103 at defaults). |
| `top_k` | 50 | How many global top-gradient scalars to retain as flip candidates. |
| `v_select` | `"all"` | Which ranks to actually flip: `"all"`, an int, or a list of ints. |
| `n_trials_per_class` | 5 | Flips per bit-class per selected rank. Total rows = `len(v_select) × 3 × n_trials_per_class × len(test_prompts)`. |
| `max_new_tokens` | 100 | Decode length for clean & corrupt generation. |
| `decoding_strategy` | `"greedy"` | Paper uses greedy; `"top-k"` and `"top-p"` are available for ablations. |
| `preserved_threshold`, `changed_threshold` | 0.87, 0.80 | Paper BERTScore-F1 classification boundaries. |
| `enable_bleurt`, `enable_llm_judge` | off | BLEURT triggers an extra `pip install bleurt` on first load. |
| `install_deps` | False | When True, pip-installs the runtime dep set on startup (Colab convenience). |

### Invariants to preserve when editing

- **Coordinates are tuples of Python ints**, not tensors. `global_topk()` already converts via `tuple(int(c) for c in coord)`. Keep it that way before using as dict keys or writing to CSV.
- **`flip_bit` requires float32.** The function raises on other dtypes; don't silently cast. If you add support for bfloat16 or float16, write a separate function — the paper's bit semantics assume IEEE-754 float32.
- **Always restore the original value in a `finally:` block** after every trial. A missed restore silently corrupts subsequent trials. The trial loop in `run_bit_flip_trials` enforces this.
- **Corrupt decode must use the clamp + repeat-guard processors.** Without them, corruption in high-sensitivity weights routinely produces NaN logits and stalls generation in a tight loop.
- **The Colab Drive mirror is best-effort.** Keep the try/except wrapper around `from google.colab import drive`; don't hard-require `google.colab`.
- **Deps install in `ensure_runtime_deps()` must be gated on `install_deps=True`.** Don't run pip on every local invocation.

## `docs/` sub-folder

`docs/` is the companion folder for the paper. It holds the LaTeX source, candidate prompts, and the paper-aligned modular Python reference implementation.

| Path | What it is |
| --- | --- |
| `docs/main.tex` | The paper itself. Formalizes the sensitivity framework (Section 1), the corpus → windows → batches → forward → backward → running-max procedure (Section 2), the GPT-2 experimental results including the top-10 tables and the 225-trial classification counts (Section 3), and appendices A/B/C with the tokenization / gradient scan algorithms and Deepseek-R1-Distill-Qwen-1.5B secondary results. Update this file if you rename a documented function or change the classification thresholds. |
| `docs/README.md` | Human-readable project overview mirroring `main.tex`. Lists expected paper results, CLI examples for `main_simulation.py`, and the output directory layout. |
| `docs/requirements.txt` | Dependency pin: `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`, `seaborn`, `pandas`, `nltk`, `bert-score`, `scipy`. Kept as the minimal install set for the `docs/*.py` modules (not for `sensitivity_pipeline.py`, which additionally uses `evaluate`, `sacrebleu`, `rouge-score`, `tabulate`). |
| `docs/candidate-prompts.txt` | Reference list of candidate test prompts considered for the paper. The paper selected the five listed under "Paper context" above. |
| `docs/model_config.py` | `ModelConfig` — loads HF causal LMs (GPT-2, GPT-Neo, DistilBERT), classifies parameters into `embedding / layer_norm / attention / mlp / output_head / other`, prints per-layer stats, and exposes `get_recommended_config("small"/"medium"/"large")` presets. |
| `docs/data_processing.py` | Algorithm A.1 as a module: `chunk_generator()`, `get_batches()`, and a `DataLoader` wrapper around `datasets.load_dataset` with a `set_tokenizer` / `get_batch_iterator` API. |
| `docs/gradient_scanner.py` | Algorithm B.1 as a class: `GradientScanner` tracks `running_max` per parameter, exposes `scan_batch` / `scan_dataset`, `get_sensitivity_rankings(exclude_embeddings, exclude_layernorm)`, `get_top_k_sensitive`, `print_sensitivity_report`, and `save_results` / `load_results` (`torch.save`). |
| `docs/bit_flip_simulator.py` | Reference bit-flip module. Uses `struct.pack/unpack` for the bit-flip (paper-equivalent to `sensitivity_pipeline.py`'s numpy-uint32 view). Provides `save_original_state` / `restore_original_state` via `state_dict` snapshot, per-element `corrupt_parameter_element`, and a `run_bit_flip_experiment` driver. **Note:** this module uses `do_sample=True` sampling in its text-generation helper, while the paper and `sensitivity_pipeline.py` use greedy decoding — expect different outputs. |
| `docs/output_evaluation.py` | `OutputEvaluator` — BERTScore-F1 wrapper (with a Jaccard/bigram fallback when `bert_score` is unavailable), paper classification (Preserved/Changed/Gibberish), per-text structural analysis (repetition ratio, comma density, avg word length, etc.), experiment-level aggregation, and `compare_with_paper_results` which checks percentages against the paper's 1.78/2.67/95.6 split. |
| `docs/sensitivity_analysis.py` | `SensitivityAnalyzer` — layer-distribution stats, parameter-pattern stats, per-tensor sensitivity heatmaps (matplotlib/seaborn), and a three-panel dashboard: top-K sensitivity curve, layer-type pie+bar, magnitude histogram. Exports results as JSON. |
| `docs/main_simulation.py` | End-to-end driver tying the seven `docs/*.py` modules together. Produces a timestamped `results_YYYYMMDD_HHMMSS/` directory with `model_info.json`, `gradient_scan_results.pt`, `sensitivity_analysis.json`, `bit_flip_results.json`, `evaluation_results.json`, `evaluation_report.txt`, `paper_comparison.json`, `simulation_summary.json`, and a `plots/` sub-directory. Superseded by `sensitivity_pipeline.py` for new work, but useful when you need the per-layer visual dashboard. |

## Legacy notebooks and extracted scripts

These live under `archive/` for reference. Prefer `sensitivity_pipeline.py` for new experiments.

- `archive/Final.ipynb` — historical Colab-oriented end-to-end pipeline. Largest and most complete notebook; `sensitivity_pipeline.py` was built from its cells.
- `archive/Grad_Sensitivity.ipynb` — repo-friendly variant of `Final.ipynb`; writes outputs under a local `bitflip_outputs/` directory (same name that `sensitivity_pipeline.py` uses).
- `archive/Top-k Grad.ipynb` — gradient scan only; reports top-K most sensitive scalars without running bit-flip trials.
- `archive/gpt2_prompt_runner.ipynb` — minimal GPT-2 generation playground for probing prompts.
- `archive/Final_extracted.py` — flat dump of every code cell from `Final.ipynb`. **Not runnable as a script** — it concatenates multiple experimental blocks that redefine the same names. Useful for grepping function definitions without opening the notebook. Treat as read-only reference; edit the notebook or `sensitivity_pipeline.py` instead.
- `Cross-Entropy Loss Sensitivity`, `Related Codes`, `Soft Error Propagation in DNNs` — prior literature PDFs / references at the repo root (no extension, so `git` tracks them as binaries). Consult when adding new sensitivity or soft-error methods.

## Output layout

Running `python3 sensitivity_pipeline.py` writes to `bitflip_outputs/` (the current working directory):

```
bitflip_outputs/
├── topk_sensitive.csv           # rank, tensor, coord, abs_grad
├── bitflip_per_trial.csv        # one row per prompt-trial
├── bitflip_aggregated.csv       # mean/median/std per (rank, tensor, coord, bit_class, prompt)
└── classification_summary.txt   # Preserved/Changed/Gibberish counts + BERTScore F1 stats
```

`docs/main_simulation.py` instead writes to a timestamped `results_YYYYMMDD_HHMMSS/` directory with the JSON/PT-based schema documented in `docs/README.md`. Don't mix the two output formats in the same run.

## Conventions to preserve

- **Notebooks install their own dependencies** via `subprocess.run([sys.executable, "-m", "pip", "install", ...])` inside their first cell. `sensitivity_pipeline.py` centralizes that behavior in `ensure_runtime_deps(install=cfg.install_deps)` — don't scatter pip calls through the module.
- **Dependencies:** `sensitivity_pipeline.py` needs `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `sacrebleu`, `evaluate`, `rouge-score`, `bert-score`, plus `nltk` if the `docs/*.py` modules are imported. `docs/requirements.txt` is the minimal set for `docs/*.py` only.
- **No package, no test suite, no build step.** This is a research repo. Both entry points (`sensitivity_pipeline.py` and `docs/main_simulation.py`) are self-contained — no `setup.py`, no `pyproject.toml`, no `conftest.py`.
- **Paper LaTeX is authoritative for results shape.** Keep `docs/main.tex` in sync with any change to classification thresholds, algorithm defaults, or published top-10 tables.
- **Google Drive save block stays wrapped in try/except.** `sensitivity_pipeline.py` and all notebooks mirror outputs to `/content/drive/MyDrive/bitflip_outputs` when running in Colab; don't hard-require `google.colab`.
- **Keep `CLAUDE.md` and `README.md` in sync with repo changes.** Whenever a file is added, moved, renamed, or deleted (especially entry points, `docs/*.py`, `archive/*`, output directories, or invariants in `sensitivity_pipeline.py`), update both files in the same change. This is enforced by a `Stop` hook defined in `.claude/settings.json` and implemented at `.claude/hooks/docs-sync-reminder.sh`: it runs `git status --porcelain` at session-end, filters out `CLAUDE.md` / `README.md`, and if anything else remains, emits `{decision:"block", reason:...}` with the list of changes so Claude is forced to re-check both docs before stopping. A `/tmp/claude-docs-reminder-<session_id>` sentinel keeps it from firing more than once per session. Don't bypass the hook — if it fires, update the docs and let it pass on the next stop. Personal-scope permission allows live in the (gitignored by convention) `.claude/settings.local.json`.
