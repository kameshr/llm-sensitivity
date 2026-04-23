# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research code for analyzing error robustness of GPT-2 under bit-level soft errors. All experiments combine:

1. A **gradient scan** over GPT-2 parameters on WikiText-103 windows to identify the global top-K scalars with the largest `|∂L/∂θ|`.
2. A **bit-flip injection** on those top-K weights, sampling bits from sign / exponent / mantissa classes of the FP32 representation.
3. **Clean vs. corrupt text generation** from a fixed `TEST_PROMPTS` list, scored with edit distance, BLEU, METEOR, BERTScore, ROUGE (optionally BLEURT), then written to per-trial and aggregated CSVs.

There is no package, no test suite, no build step — this is a notebook-first research repo that targets Google Colab.

## Running experiments

Primary entry points are Jupyter notebooks, each self-contained (each cell re-installs deps, re-loads the model, re-imports everything):

- `Final.ipynb` — Colab-oriented end-to-end pipeline (gradient scan → bit-flip → metric scoring → CSVs to `/content/drive/MyDrive/bitflip_outputs`).
- `Grad_Sensitivity.ipynb` — Repo-friendly variant that writes under a local `bitflip_outputs/` directory.
- `Top-k Grad.ipynb` — Gradient scan only; reports the top-K most sensitive scalars.
- `gpt2_prompt_runner.ipynb` — Minimal GPT-2 text-generation playground.
- `Final_extracted.py` — A flat dump of every code cell from `Final.ipynb` (useful for grepping function definitions without opening the notebook). Not runnable as a script — it concatenates multiple experimental blocks that redefine the same names. When fixing a function, edit both the `.py` and the corresponding notebook cell, or treat the `.py` as read-only reference.

Typical run: open a notebook in Colab (CUDA expected; CPU fallback exists via `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`), execute cells top-to-bottom. Outputs land in `bitflip_per_trial.csv` and `bitflip_aggregated.csv` locally, with a second copy to Google Drive when the Colab mount succeeds.

## Experiment knobs

Each pipeline cell defines its own config block near the top. The load-bearing ones that change runtime / cost:

- `MODEL_ID` — always `"gpt2"` (GPT-2 small, 117M).
- `SEQ_LEN`, `BATCH_SIZE`, `MAX_STEPS` — control the gradient scan cost (windows of `SEQ_LEN` tokens, `MAX_STEPS` mini-batches).
- `TOP_K` — how many global top-gradient scalars to retain as flip candidates.
- `V_SELECT` — which ranks within the top-K to actually flip: `"all"`, a single int, or a list of ints. Normalized by `normalize_v_select`.
- `N_TRIALS_PER_CLASS` — flips per bit-class (sign / exponent / mantissa) per selected rank.
- `MAX_NEW_TOKENS` — decode length for `generate_tail_clean` / `generate_tail_corrupt`.
- `ENABLE_BLEU / METEOR / BERTSCORE / ROUGE / BLEURT / LLM_JUDGE` — metric toggles. BLEURT and the LLM judge are off by default; BLEURT triggers an extra `pip install bleurt` when enabled.

## Pipeline shape (shared across notebooks)

1. `chunk_generator()` streams WikiText-103 docs, tokenizes, and yields `(input, target)` windows of length `SEQ_LEN`. `get_batch(gen, bs)` stacks them into device tensors.
2. Gradient scan: for each mini-batch, `model(inp, labels=inp).loss.backward()`, then `running_max[name] = max(running_max[name], |grad|)` accumulated on **CPU** to avoid GPU OOM for the full parameter set. After the scan, per-tensor top-K is collected into a global sorted list `topk_entries`.
3. `flip_bit(val_tensor, bit)` performs the actual corruption: moves to CPU, views `float32` as `uint32`, XORs the bit, views back. Restricted to FP32. `BIT_CLASSES = {"sign":[31], "exponent":[23..30], "mantissa":[0..22]}`.
4. Each trial: save the original value, flip a bit, generate continuations for every `TEST_PROMPTS` entry, score against the clean cache, **always restore the original value in a `finally` block** before the next trial. Preserve this invariant when editing flip loops — a missed restore silently corrupts subsequent trials.
5. Generation: clean path uses plain greedy decoding; corrupt path wraps decoding in a `LogitsProcessorList` with `NanInfClampProcessor` (replaces NaN/Inf, clamps to `[-80, 80]`, records a `had_nan` flag) and `MaxConsecutiveRepeatProcessor` (bans a token after 6 consecutive repeats) plus `no_repeat_ngram_size=3`. This is required — unguarded corrupt decoding routinely emits NaN logits and degenerate loops.
6. Scoring via `score_pair(clean, corrupt)` returns a dict merged into each result row. `edit_distance` is a local Levenshtein DP implementation, not an external dep.
7. Result rows become a DataFrame with fixed columns (`rank, tensor, coord, bit_class, bit_index, trial, prompt, clean, corrupt, corrupt_logits_had_nan, <metrics>`). Aggregation uses `groupby(["rank","tensor","coord","bit_class","prompt"]).agg(mean/median/std)`.

## Conventions to preserve

- **Coordinates are tuples of Python ints**, not tensors. Convert with `tuple(map(int, coord))` before using as dict keys or writing to CSV.
- **Bit-flip requires `float32`**. `flip_bit` raises on other dtypes; don't silently cast.
- The Colab Drive save block is wrapped in `try/except` so non-Colab runs don't fail — keep that shape; don't hard-require `google.colab`.
- Notebooks install their own dependencies via `subprocess.run([sys.executable, "-m", "pip", "install", ...])` inside the first cell. There's no `requirements.txt` or env file; if you need to add a dep, add it to that install block in each affected notebook.

## Docs

`docs/Final_functions_report.tex` (and its compiled PDF) is a human-readable reference for the functions and classes in `Final.ipynb`. Useful when you need an overview without opening the notebook; update it if you rename or materially change a documented function.
