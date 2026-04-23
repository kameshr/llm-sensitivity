#!/usr/bin/env python3
"""
Unified sensitivity pipeline for GPT-2 under bit-level soft errors.

This script is the single current reference implementation of the paper
"Gradient-Based Sensitivity Analysis in Large Language Models"
(see docs/main.tex). It consolidates all prior notebooks
(Final.ipynb, Grad_Sensitivity.ipynb, Top-k Grad.ipynb) and the
modular scripts in docs/ (model_config.py, data_processing.py,
gradient_scanner.py, bit_flip_simulator.py, output_evaluation.py,
sensitivity_analysis.py, main_simulation.py) into one cohesive flow.

Pipeline stages
---------------
  1. Load a causal LM (default GPT-2 small, 117M) and tokenizer.
  2. Stream WikiText-103 into windows of SEQ_LEN+1 tokens, yielding
     (input, target) pairs for next-token prediction (Alg. A.1).
  3. Gradient scan: for MAX_STEPS mini-batches, backprop cross-entropy
     loss and accumulate running_max[name] = max(running_max[name],
     |grad|) on CPU to avoid GPU OOM (Alg. B.1).
  4. Global top-K: collect per-tensor topk, then sort to a global list.
  5. Bit-flip trials: for each rank in V_SELECT and each of sign /
     exponent / mantissa bit classes, run N_TRIALS_PER_CLASS trials.
     Each trial picks a random bit in that class, flips it in the
     float32 view, generates greedy continuations for TEST_PROMPTS
     (clean cached once, corrupt wrapped with NaN/Inf clamp +
     consecutive-repeat guard + no_repeat_ngram_size=3), scores
     clean vs corrupt, and ALWAYS restores the original value in a
     finally: block before the next trial.
  6. Scoring: edit distance, BLEU, METEOR, BERTScore-F1, ROUGE-{1,2,L}
     (BLEURT optional). Classification per paper thresholds on
     BERTScore-F1: >0.87 Preserved, 0.80-0.87 Changed, <0.80 Gibberish.
  7. Output: per-trial CSV (bitflip_per_trial.csv), aggregated CSV
     (bitflip_aggregated.csv, mean/median/std per rank x tensor x coord
     x bit_class x prompt), top-K ranking CSV (topk_sensitive.csv),
     and a human-readable classification summary. All artefacts land in
     OUTPUT_DIR (local) and are mirrored to Google Drive when
     `google.colab` is importable.

Run
---
  Local:   python sensitivity_pipeline.py
  Colab:   open Final.ipynb or paste this file into a cell.

All knobs are at the top (CONFIG) or accepted via argparse. Invariants
marked (*) must be preserved when editing:
  - flip_bit requires float32 (*)
  - original weight must be restored in finally: after every trial (*)
  - coords stored as tuple(map(int, coord)) before CSV/dict use (*)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 0) Optional dependency install block (Colab-friendly)
# ============================================================
def _pip_install(pkgs: Sequence[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs],
        check=True,
    )


def ensure_runtime_deps(install: bool = False) -> None:
    """Best-effort dependency install for Colab. No-op when install=False."""
    if not install:
        return
    _pip_install([
        "datasets>=2.18.0", "fsspec>=2023.6.0",
        "transformers>=4.30.0",
        "pandas>=2.0.0", "sacrebleu>=2.4.0",
        "evaluate>=0.4.2", "rouge-score>=0.1.2",
        "bert-score>=0.3.13", "tabulate>=0.9.0",
        "nltk>=3.8",
    ])


# Resolve the default output directory to a sibling of this script, so the
# pipeline always writes to <repo>/bitflip_outputs/ regardless of CWD.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "bitflip_outputs")


# ============================================================
# 1) Config
# ============================================================
@dataclass
class Config:
    # model / data
    model_id: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_subset: str = "wikitext-103-raw-v1"
    split: str = "train"
    device: str = "auto"                 # "auto" | "cpu" | "cuda" | "mps"
    seed: int = 123

    # gradient scan
    seq_len: int = 1024
    batch_size: int = 16
    max_steps: int = 1100
    top_k: int = 50

    # bit-flip trials
    v_select: Any = "all"                # "all" | int | list[int]
    n_trials_per_class: int = 5
    max_new_tokens: int = 100

    # decoding (paper uses greedy)
    decoding_strategy: str = "greedy"    # "greedy" | "top-k" | "top-p"
    top_k_sampling: int = 10
    top_p_sampling: float = 0.9
    temperature: float = 1.0

    # metrics
    enable_bleu: bool = True
    enable_meteor: bool = True
    enable_bertscore: bool = True
    enable_rouge: bool = True
    enable_bleurt: bool = False

    # classification thresholds (paper)
    preserved_threshold: float = 0.87    # BERTScore F1 >  → preserved
    changed_threshold: float = 0.80      # BERTScore F1 in (0.80, 0.87] → changed; else gibberish

    # IO
    output_dir: str = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    save_to_gdrive: bool = True
    install_deps: bool = False
    log_every: int = 100

    # prompts
    test_prompts: List[str] = field(default_factory=lambda: [
        "The weather today is",
        "The patient should take",
        "The bank transfer amount is",
        "The recommended dose for a child is",
        "The evacuation order status is",
    ])


# Paper bit layout for IEEE-754 float32.
BIT_CLASSES: Dict[str, List[int]] = {
    "sign":     [31],
    "exponent": list(range(23, 31)),
    "mantissa": list(range(0, 23)),
}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================
# 2) Data: tokenization and batching (Alg. A.1)
# ============================================================
def chunk_generator(wiki, tok, seq_len: int) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Stream tokens from an HF dataset into fixed (L+1)-long windows."""
    cache: List[int] = []
    for doc in wiki:
        cache.extend(tok(doc["text"]).input_ids)
        while len(cache) >= seq_len + 1:
            win, cache = cache[:seq_len + 1], cache[seq_len + 1:]
            yield win[:-1], win[1:]


def get_batches(gen, batch_size: int, device: str) -> Generator[torch.Tensor, None, None]:
    """Stack windows into [B x L] batches on the target device."""
    buf: List[List[int]] = []
    for x, _ in gen:
        buf.append(x)
        if len(buf) == batch_size:
            yield torch.tensor(buf, device=device)
            buf = []


# ============================================================
# 3) Gradient scan (Alg. B.1)
# ============================================================
def gradient_scan(
    model: nn.Module,
    tok,
    wiki,
    cfg: Config,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, nn.Parameter]]:
    """Accumulate running_max[name] = max over batches of |grad| on CPU."""
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    running_max = {n: torch.zeros_like(p, device="cpu") for n, p in param_dict.items()}

    gen = chunk_generator(wiki, tok, cfg.seq_len)
    batches = get_batches(gen, cfg.batch_size, device)

    print(f"[scan] model={cfg.model_id} device={device} "
          f"SEQ_LEN={cfg.seq_len} BATCH_SIZE={cfg.batch_size} MAX_STEPS={cfg.max_steps}")
    for step, inp in enumerate(batches, 1):
        model.zero_grad(set_to_none=True)
        loss = model(inp, labels=inp).loss
        loss.backward()
        for name, p in param_dict.items():
            if p.grad is None:
                continue
            running_max[name] = torch.maximum(
                running_max[name],
                p.grad.detach().abs().to("cpu"),
            )
        if step % cfg.log_every == 0 or step == 1:
            print(f"[scan] step {step}/{cfg.max_steps}  loss={loss.item():.4f}")
        if step >= cfg.max_steps:
            break

    return running_max, param_dict


def global_topk(
    running_max: Dict[str, torch.Tensor],
    top_k: int,
    exclude_embeddings: bool = False,
    exclude_layernorm: bool = False,
) -> List[Tuple[float, str, Tuple[int, ...]]]:
    """Return top_k entries as (|grad|, param_name, coord_tuple_of_python_ints)."""
    candidates: List[Tuple[float, str, Tuple[int, ...]]] = []
    for name, rm in running_max.items():
        lname = name.lower()
        if exclude_embeddings and ("wte" in lname or "wpe" in lname or "embed" in lname):
            continue
        if exclude_layernorm and ("ln" in lname or "norm" in lname):
            continue
        k_local = min(top_k, rm.numel())
        if k_local == 0:
            continue
        vals, idxs = torch.topk(rm.view(-1), k_local)
        for v, flat_idx in zip(vals, idxs):
            coord = torch.unravel_index(flat_idx, rm.shape)
            # coords MUST be python ints (not tensors) before use as dict keys / CSV cells
            candidates.append((v.item(), name, tuple(int(c) for c in coord)))
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[:top_k]


def normalize_v_select(sel: Any, k: int) -> List[int]:
    """Resolve V_SELECT into a 1-indexed list of ranks within [1, k]."""
    if sel == "all":
        return list(range(1, k + 1))
    if isinstance(sel, int):
        return [sel]
    if isinstance(sel, (list, tuple)):
        return [int(x) for x in sel]
    raise ValueError("V_SELECT must be 'all', int, or list[int]")


# ============================================================
# 4) Bit-flip primitive (FP32 only) — paper-critical invariant
# ============================================================
def flip_bit(val_tensor: torch.Tensor, bit: int) -> torch.Tensor:
    """Flip a single bit of a float32 tensor via uint32 view XOR.

    Raises on non-float32 to preserve the paper's bit semantics.
    """
    if val_tensor.dtype != torch.float32:
        raise TypeError("flip_bit expects a float32 tensor")
    if bit < 0 or bit > 31:
        raise ValueError("bit index must be in [0, 31]")
    device = val_tensor.device
    np_view = val_tensor.detach().cpu().numpy().copy().view(np.uint32)
    np_view ^= np.uint32(1) << np.uint32(bit)
    flipped = torch.from_numpy(np_view.view(np.float32)).to(device)
    return flipped.view_as(val_tensor)


# ============================================================
# 5) Logits guards for corrupt decoding
#     Unguarded corrupt decoding routinely emits NaN/Inf logits
#     and degenerate loops. These two processors are mandatory.
# ============================================================
def _build_logits_processors():
    # Import here so the module is importable without `transformers`.
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

    class NanInfClampProcessor(LogitsProcessor):
        def __init__(self, clamp_min=-80.0, clamp_max=80.0, flag_dict=None):
            self.clamp_min = clamp_min
            self.clamp_max = clamp_max
            self.flag_dict = flag_dict

        def __call__(self, input_ids, scores):
            if self.flag_dict is not None and (torch.isnan(scores).any() or torch.isinf(scores).any()):
                self.flag_dict["had_nan"] = True
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.clamp(scores, self.clamp_min, self.clamp_max)

    class MaxConsecutiveRepeatProcessor(LogitsProcessor):
        def __init__(self, max_consecutive=6):
            self.max_consecutive = max_consecutive

        def __call__(self, input_ids, scores):
            if input_ids.size(0) != 1 or input_ids.size(1) == 0:
                return scores
            seq, last = input_ids[0], input_ids[0, -1].item()
            run = 0
            for t in range(seq.size(0) - 1, -1, -1):
                if seq[t].item() == last:
                    run += 1
                else:
                    break
            if run >= self.max_consecutive:
                scores[:, last] = -1e9
            return scores

    return NanInfClampProcessor, MaxConsecutiveRepeatProcessor, LogitsProcessorList


# ============================================================
# 6) Generation helpers
# ============================================================
class Generator_:
    """Tiny wrapper around `model.generate` for clean / corrupt paths."""

    def __init__(self, model, tok, cfg: Config, device: str):
        self.model = model
        self.tok = tok
        self.cfg = cfg
        self.device = device
        (self.NanInfClampProcessor,
         self.MaxConsecutiveRepeatProcessor,
         self.LogitsProcessorList) = _build_logits_processors()

    def _sampling_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"temperature": self.cfg.temperature}
        if self.cfg.decoding_strategy == "top-k":
            kw["top_k"] = self.cfg.top_k_sampling
        elif self.cfg.decoding_strategy == "top-p":
            kw["top_p"] = self.cfg.top_p_sampling
        return kw

    @torch.no_grad()
    def clean(self, prompt: str) -> str:
        enc = self.tok(prompt, return_tensors="pt", return_attention_mask=True)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out = self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            do_sample=(self.cfg.decoding_strategy != "greedy"),
            max_new_tokens=self.cfg.max_new_tokens,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
            **self._sampling_kwargs(),
        )
        return self.tok.decode(out[0, ids.size(1):], skip_special_tokens=True)

    @torch.no_grad()
    def corrupt(self, prompt: str) -> Tuple[str, bool]:
        enc = self.tok(prompt, return_tensors="pt", return_attention_mask=True)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        diag = {"had_nan": False}
        procs = self.LogitsProcessorList([
            self.NanInfClampProcessor(clamp_min=-80.0, clamp_max=80.0, flag_dict=diag),
            self.MaxConsecutiveRepeatProcessor(max_consecutive=6),
        ])
        out = self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            do_sample=(self.cfg.decoding_strategy != "greedy"),
            max_new_tokens=self.cfg.max_new_tokens,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
            logits_processor=procs,
            no_repeat_ngram_size=3,
            **self._sampling_kwargs(),
        )
        return self.tok.decode(out[0, ids.size(1):], skip_special_tokens=True), diag["had_nan"]


# ============================================================
# 7) Metrics
# ============================================================
def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (local DP, no external dep)."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


class MetricSuite:
    """Lazy-loaded metric bundle. Wraps each compute in try/except so one
    failing metric (e.g., BERTScore model download) doesn't poison a trial."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.meteor = None
        self.bertscore = None
        self.rouge = None
        self.bleurt = None
        self._load()

    def _load(self) -> None:
        try:
            import evaluate
        except ImportError:
            print("[metrics] `evaluate` not installed — metrics disabled.")
            return
        if self.cfg.enable_meteor:
            try: self.meteor = evaluate.load("meteor")
            except Exception as e: print(f"[metrics] METEOR load failed: {e}")
        if self.cfg.enable_bertscore:
            try: self.bertscore = evaluate.load("bertscore")
            except Exception as e: print(f"[metrics] BERTScore load failed: {e}")
        if self.cfg.enable_rouge:
            try: self.rouge = evaluate.load("rouge")
            except Exception as e: print(f"[metrics] ROUGE load failed: {e}")
        if self.cfg.enable_bleurt:
            try:
                _pip_install(["bleurt"])
                self.bleurt = evaluate.load("bleurt")
            except Exception as e: print(f"[metrics] BLEURT load failed: {e}")

    def score_pair(self, clean: str, corrupt: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        ed = edit_distance(clean, corrupt)
        scores["EditDist"] = float(ed)
        scores["EditDist_Norm"] = float(ed / max(1, len(clean)))

        if self.cfg.enable_bleu:
            try:
                import sacrebleu
                scores["BLEU"] = float(sacrebleu.corpus_bleu([corrupt], [[clean]]).score)
            except Exception as e:
                print(f"[metrics] BLEU failed: {e}")

        if self.meteor is not None:
            try:
                scores["METEOR"] = float(self.meteor.compute(predictions=[corrupt], references=[clean])["meteor"])
            except Exception as e:
                print(f"[metrics] METEOR failed: {e}")

        if self.bertscore is not None:
            try:
                bs = self.bertscore.compute(predictions=[corrupt], references=[clean], lang="en")
                scores["BERTScore_F1"] = float(bs["f1"][0])
            except Exception as e:
                print(f"[metrics] BERTScore failed: {e}")

        if self.rouge is not None:
            try:
                r = self.rouge.compute(predictions=[corrupt], references=[clean], use_stemmer=True)
                scores["ROUGE1_F1"] = float(r["rouge1"])
                scores["ROUGE2_F1"] = float(r["rouge2"])
                scores["ROUGEL_F1"] = float(r["rougeL"])
            except Exception as e:
                print(f"[metrics] ROUGE failed: {e}")

        if self.bleurt is not None:
            try:
                scores["BLEURT"] = float(self.bleurt.compute(predictions=[corrupt], references=[clean])["scores"][0])
            except Exception as e:
                print(f"[metrics] BLEURT failed: {e}")

        return scores


def classify_output(f1: Optional[float], cfg: Config) -> str:
    """Paper-specified 3-way classification on BERTScore F1."""
    if f1 is None:
        return "unknown"
    if f1 > cfg.preserved_threshold:
        return "preserved"
    if f1 > cfg.changed_threshold:
        return "changed"
    return "gibberish"


# ============================================================
# 8) Bit-flip trial loop
# ============================================================
def run_bit_flip_trials(
    model: nn.Module,
    tok,
    cfg: Config,
    device: str,
    topk_entries: List[Tuple[float, str, Tuple[int, ...]]],
    param_dict: Dict[str, nn.Parameter],
    metrics: MetricSuite,
) -> "pd.DataFrame":
    import pandas as pd

    ranks_to_test = normalize_v_select(cfg.v_select, len(topk_entries))
    gen = Generator_(model, tok, cfg, device)

    # Cache clean generations once — they never change.
    model.eval()
    clean_cache: Dict[str, str] = {p: gen.clean(p) for p in cfg.test_prompts}

    coords_list = [(name, coord) for _, name, coord in topk_entries]
    rows: List[Dict[str, Any]] = []

    total = len(ranks_to_test) * len(BIT_CLASSES) * cfg.n_trials_per_class * len(cfg.test_prompts)
    done = 0
    print(f"[flip] running {total} prompt-trials "
          f"({len(ranks_to_test)} ranks x {len(BIT_CLASSES)} bit-classes "
          f"x {cfg.n_trials_per_class} trials x {len(cfg.test_prompts)} prompts)")

    t0 = time.time()
    for rank in ranks_to_test:
        tensor_name, coord = coords_list[rank - 1]
        W = param_dict[tensor_name]
        orig_val = W.data[coord].clone()

        for bit_class, pool in BIT_CLASSES.items():
            for trial in range(1, cfg.n_trials_per_class + 1):
                bit = random.choice(pool)
                W.data[coord] = flip_bit(orig_val, bit)
                try:
                    for prompt in cfg.test_prompts:
                        clean_out = clean_cache[prompt]
                        corrupt_out, had_nan = gen.corrupt(prompt)
                        scores = metrics.score_pair(clean_out, corrupt_out)
                        f1 = scores.get("BERTScore_F1")
                        rows.append({
                            "rank": rank,
                            "tensor": tensor_name,
                            "coord": coord,                # already python-ints tuple
                            "bit_class": bit_class,
                            "bit_index": bit,
                            "trial": trial,
                            "prompt": prompt,
                            "clean": clean_out,
                            "corrupt": corrupt_out,
                            "corrupt_logits_had_nan": had_nan,
                            "classification": classify_output(f1, cfg),
                            **scores,
                        })
                        done += 1
                        if done % 50 == 0 or done == total:
                            print(f"[flip] {done}/{total}  "
                                  f"elapsed={time.time() - t0:.1f}s  "
                                  f"last_rank={rank} class={bit_class} trial={trial}")
                finally:
                    # Invariant: always restore before the next trial, even on error.
                    W.data[coord] = orig_val

    return pd.DataFrame(rows)


# ============================================================
# 9) Output: CSVs + summaries
# ============================================================
def _metric_cols_present(df) -> List[str]:
    candidates = ["EditDist", "EditDist_Norm", "BLEU", "METEOR",
                  "BERTScore_F1", "ROUGE1_F1", "ROUGE2_F1", "ROUGEL_F1", "BLEURT"]
    return [c for c in candidates if c in df.columns]


def write_outputs(
    df,
    topk_entries: List[Tuple[float, str, Tuple[int, ...]]],
    cfg: Config,
) -> Dict[str, str]:
    import pandas as pd
    os.makedirs(cfg.output_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    # 9a) top-K ranking
    topk_df = pd.DataFrame([
        {"rank": i + 1, "tensor": name, "coord": coord, "abs_grad": v}
        for i, (v, name, coord) in enumerate(topk_entries)
    ])
    paths["topk"] = os.path.join(cfg.output_dir, "topk_sensitive.csv")
    topk_df.to_csv(paths["topk"], index=False)

    if df.empty:
        print("[out] no trial rows produced — skipping trial CSVs.")
        return paths

    # 9b) per-trial
    paths["per_trial"] = os.path.join(cfg.output_dir, "bitflip_per_trial.csv")
    df.to_csv(paths["per_trial"], index=False)

    # 9c) aggregated (mean/median/std per rank x tensor x coord x bit_class x prompt)
    metric_cols = _metric_cols_present(df)
    if metric_cols:
        summary = df.groupby(
            ["rank", "tensor", "coord", "bit_class", "prompt"], as_index=False
        ).agg({m: ["mean", "median", "std"] for m in metric_cols})
        if isinstance(summary.columns, pd.MultiIndex):
            summary.columns = ["_".join([str(c) for c in col if c]) for col in summary.columns]
        paths["aggregated"] = os.path.join(cfg.output_dir, "bitflip_aggregated.csv")
        summary.to_csv(paths["aggregated"], index=False)

    # 9d) classification summary (paper-style)
    if "classification" in df.columns:
        counts = df["classification"].value_counts().to_dict()
        total = int(df.shape[0])
        lines = ["=" * 60, "BIT-FLIP CLASSIFICATION SUMMARY", "=" * 60,
                 f"Total prompt-trials: {total}"]
        for label in ("preserved", "changed", "gibberish", "unknown"):
            c = int(counts.get(label, 0))
            pct = (c / total) * 100 if total else 0.0
            lines.append(f"  {label:<10}: {c:>5} ({pct:5.2f}%)")
        if "BERTScore_F1" in df.columns:
            s = df["BERTScore_F1"].dropna()
            if not s.empty:
                lines += [
                    "", "BERTScore F1 stats:",
                    f"  mean={s.mean():.4f}  std={s.std():.4f}  "
                    f"median={s.median():.4f}  min={s.min():.4f}  max={s.max():.4f}",
                ]
        report = "\n".join(lines)
        paths["summary"] = os.path.join(cfg.output_dir, "classification_summary.txt")
        with open(paths["summary"], "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print("\n" + report)

    # 9e) mirror to Google Drive when running on Colab
    if cfg.save_to_gdrive:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive", force_remount=False)
            gdir = "/content/drive/MyDrive/bitflip_outputs"
            os.makedirs(gdir, exist_ok=True)
            for key, p in list(paths.items()):
                dst = os.path.join(gdir, os.path.basename(p))
                try:
                    import shutil
                    shutil.copyfile(p, dst)
                    print(f"[out] mirrored {key} → {dst}")
                except Exception as e:
                    print(f"[out] Drive copy of {key} failed: {e}")
        except Exception as e:
            # Silently ignore outside Colab — this is expected in local runs.
            print(f"[out] Drive mirror skipped: {e}")

    return paths


# ============================================================
# 10) Entry point
# ============================================================
def load_model_and_tokenizer(cfg: Config, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float32).to(device)
    return model, tok


def run(cfg: Config) -> Dict[str, Any]:
    ensure_runtime_deps(install=cfg.install_deps)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = resolve_device(cfg.device)
    model, tok = load_model_and_tokenizer(cfg, device)

    from datasets import load_dataset
    wiki = load_dataset(cfg.dataset_name, cfg.dataset_subset, split=cfg.split, trust_remote_code=True)

    # gradient scan first requires train-mode forward + backward
    model.train()
    running_max, param_dict = gradient_scan(model, tok, wiki, cfg, device)

    # report both variants (paper Tables 1 and 2)
    topk_all = global_topk(running_max, cfg.top_k)
    topk_filtered = global_topk(running_max, cfg.top_k,
                                exclude_embeddings=True, exclude_layernorm=True)

    print(f"\nGlobal Top-{cfg.top_k} |∂L/∂θ| scalars (ALL tensors):")
    for r, (v, n, c) in enumerate(topk_all[:10], 1):
        print(f"  #{r}: {n}{c}  |grad|={v:.3e}")
    print(f"\nGlobal Top-{cfg.top_k} |∂L/∂θ| scalars (NO embeddings, NO layer-norm):")
    for r, (v, n, c) in enumerate(topk_filtered[:10], 1):
        print(f"  #{r}: {n}{c}  |grad|={v:.3e}")

    metrics = MetricSuite(cfg)
    model.eval()  # generation runs after the scan; switch to eval
    df = run_bit_flip_trials(model, tok, cfg, device, topk_all, param_dict, metrics)
    paths = write_outputs(df, topk_all, cfg)

    return {"config": cfg, "topk": topk_all, "topk_filtered": topk_filtered,
            "trials": df, "paths": paths}


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Gradient-sensitivity + bit-flip pipeline for LLMs")
    p.add_argument("--model-id", default="gpt2")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1100)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--v-select", default="all",
                   help='"all", an int, or comma-separated ints (e.g. "1,2,3")')
    p.add_argument("--n-trials-per-class", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--decoding", default="greedy", choices=["greedy", "top-k", "top-p"])
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-gdrive", action="store_true", help="disable Google Drive mirror")
    p.add_argument("--install-deps", action="store_true", help="pip install runtime deps on start (Colab)")
    p.add_argument("--no-bleu", action="store_true")
    p.add_argument("--no-meteor", action="store_true")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--bleurt", action="store_true", help="enable BLEURT (triggers extra install)")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    v_select: Any
    if args.v_select == "all":
        v_select = "all"
    elif "," in args.v_select:
        v_select = [int(x) for x in args.v_select.split(",") if x.strip()]
    else:
        v_select = int(args.v_select)

    return Config(
        model_id=args.model_id,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        top_k=args.top_k,
        v_select=v_select,
        n_trials_per_class=args.n_trials_per_class,
        max_new_tokens=args.max_new_tokens,
        decoding_strategy=args.decoding,
        output_dir=args.output_dir,
        save_to_gdrive=not args.no_gdrive,
        install_deps=args.install_deps,
        enable_bleu=not args.no_bleu,
        enable_meteor=not args.no_meteor,
        enable_bertscore=not args.no_bertscore,
        enable_rouge=not args.no_rouge,
        enable_bleurt=args.bleurt,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(parse_args())
