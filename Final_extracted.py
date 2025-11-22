
# ---- code cell ----
# ============================================================
# 0) One-off setup (installs) — run in Colab/Notebook
# ============================================================
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--upgrade",
     "datasets>=2.18.0", "fsspec>=2023.6.0",
     "pandas>=2.0.0", "sacrebleu>=2.4.0",
     "evaluate>=0.4.2", "rouge-score>=0.1.2",
     "bert-score>=0.3.13", "tabulate>=0.9.0"],
    check=True
)

# ============================================================
# 1) Imports & Config
# ============================================================
import math, random, os, torch, torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import sacrebleu
import evaluate
from tabulate import tabulate

# ---- Experiment knobs ----
SEED = 123
random.seed(SEED); torch.manual_seed(SEED)

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID           = "gpt2"
SPLIT              = "train"
SEQ_LEN            = 1024
BATCH_SIZE         = 4
MAX_STEPS          = 4
TOP_K              = 50
V_SELECT           = "1"
N_TRIALS_PER_CLASS = 1
MAX_NEW_TOKENS     = 5
MAX_COL_WIDTH      = 100

TEST_PROMPTS = [
    "The weather today is",
    "The patient should take",
    "The bank transfer amount is",
    "The recommended dose for a child is",
    "The evacuation order status is",
]

ENABLE_BLEU       = True
ENABLE_METEOR     = True
ENABLE_BERTSCORE  = True
ENABLE_ROUGE      = True
ENABLE_BLEURT     = False
ENABLE_LLM_JUDGE  = False

BERTSCORE_KW = dict(lang="en")

# ============================================================
# 2) Model & Tokenizer
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ============================================================
# 3) WikiText windows for gradient scan
# ============================================================
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split=SPLIT)

def chunk_generator():
    cache = []
    for doc in wiki:
        cache.extend(tok(doc["text"]).input_ids)
        while len(cache) >= SEQ_LEN + 1:
            win, cache = cache[:SEQ_LEN + 1], cache[SEQ_LEN + 1:]
            yield win[:-1], win[1:]

def get_batch(gen, bs=BATCH_SIZE):
    buf = []
    for x, _ in gen:
        buf.append(x)
        if len(buf) == bs:
            yield torch.tensor(buf, device=DEVICE)
            buf = []

# ============================================================
# 4) Scan ALL tensors → Global top-K |∂L/∂θ|
# ============================================================
param_dict  = {n: p for n, p in model.named_parameters() if p.requires_grad}
running_max = {n: torch.zeros_like(p, device="cpu") for n, p in param_dict.items()}

for step, inp in enumerate(get_batch(chunk_generator()), 1):
    model.zero_grad(set_to_none=True)
    model(inp, labels=inp).loss.backward()
    for name, p in param_dict.items():
        running_max[name] = torch.maximum(
            running_max[name],
            p.grad.detach().abs().to("cpu")
        )
    if step >= MAX_STEPS:
        break

candidates = []
for name, rm in running_max.items():
    k_local = min(TOP_K, rm.numel())
    if k_local == 0:
        continue
    vals, idxs = torch.topk(rm.view(-1), k_local)
    for v, flat_idx in zip(vals, idxs):
        coord = torch.unravel_index(flat_idx, rm.shape)
        candidates.append((v.item(), name, coord))

candidates.sort(key=lambda t: t[0], reverse=True)
topk_entries = candidates[:TOP_K]
coords_list  = [(name, coord) for _, name, coord in topk_entries]

print(f"\nGlobal Top-{TOP_K} |∂L/∂θ| scalars:")
for rank, (val, name, coord) in enumerate(topk_entries, 1):
    print(f"  #{rank}: {name}{tuple(map(int,coord))}  |grad|={val:.3e}")

def normalize_v_select(sel, k):
    if sel == "all": return list(range(1, k+1))
    if isinstance(sel, int): return [sel]
    if isinstance(sel, (list, tuple)): return list(sel)
    raise ValueError("V_SELECT must be 'all', int, or list[int]")

ranks_to_test = normalize_v_select(V_SELECT, TOP_K)
print(f"\nTesting ranks: {ranks_to_test}")

# ============================================================
# 5) Bit-flip helpers (FP32)
# ============================================================

def flip_bit(val_tensor: torch.Tensor, bit: int):
    """Flip the specified bit of a float32 tensor element-by-element."""
    if val_tensor.dtype != torch.float32:
        raise TypeError("flip_bit expects a float32 tensor")
    if bit < 0 or bit > 31:
        raise ValueError("bit index must be in [0, 31]")
    device = val_tensor.device
    # Work on CPU for bit manipulation then move back to the original device
    np_view = val_tensor.detach().cpu().numpy().copy().view(np.uint32)
    np_view ^= (1 << bit)
    flipped = torch.from_numpy(np_view.view(np.float32)).to(device)
    return flipped.view_as(val_tensor)
BIT_CLASSES = {
    "sign"     : [31],
    "exponent" : list(range(23, 31)),
    "mantissa" : list(range(0, 23)),
}

# ============================================================
# 6) Metrics loaders & wrappers
# ============================================================
meteor_metric    = evaluate.load("meteor") if ENABLE_METEOR else None
bertscore_metric = evaluate.load("bertscore") if ENABLE_BERTSCORE else None
rouge_metric     = evaluate.load("rouge") if ENABLE_ROUGE else None
bleurt_metric    = None
if ENABLE_BLEURT:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bleurt"], check=True)
    bleurt_metric = evaluate.load("bleurt")

def edit_distance(a: str, b: str):
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[m]

def score_pair(clean: str, corrupt: str):
    scores = {}
    ed = edit_distance(clean, corrupt)
    scores["EditDist"] = float(ed)
    scores["EditDist_Norm"] = float(ed / max(1, len(clean)))

    if ENABLE_BLEU:
        scores["BLEU"] = sacrebleu.corpus_bleu([corrupt], [[clean]]).score
    if ENABLE_METEOR and meteor_metric is not None:
        scores["METEOR"] = float(meteor_metric.compute(predictions=[corrupt], references=[clean])["meteor"])
    if ENABLE_BERTSCORE and bertscore_metric is not None:
        bs = bertscore_metric.compute(predictions=[corrupt], references=[clean], **BERTSCORE_KW)
        scores["BERTScore_F1"] = float(bs["f1"][0])
    if ENABLE_ROUGE and rouge_metric is not None:
        r = rouge_metric.compute(predictions=[corrupt], references=[clean], use_stemmer=True)
        scores["ROUGE1_F1"] = float(r["rouge1"])
        scores["ROUGE2_F1"] = float(r["rouge2"])
        scores["ROUGEL_F1"] = float(r["rougeL"])
    if ENABLE_BLEURT and bleurt_metric is not None:
        scores["BLEURT"] = float(bleurt_metric.compute(predictions=[corrupt], references=[clean])["scores"][0])
    return scores

def llm_judge_score(prompt: str, clean: str, corrupt: str, rubric: str = None):
    if not ENABLE_LLM_JUDGE:
        return {}
    return {}

# ============================================================
# 7) Generation helpers
# ============================================================
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
        for t in range(seq.size(0)-1, -1, -1):
            if seq[t].item() == last:
                run += 1
            else:
                break
        if run >= self.max_consecutive:
            scores[:, last] = -1e9
        return scores

@torch.no_grad()
def generate_tail_clean(prompt: str, max_new_tokens=MAX_NEW_TOKENS):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(
        ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True)

@torch.no_grad()
def generate_tail_corrupt(prompt: str, max_new_tokens=MAX_NEW_TOKENS):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    diag = {"had_nan": False}
    procs = LogitsProcessorList([
        NanInfClampProcessor(clamp_min=-80.0, clamp_max=80.0, flag_dict=diag),
        MaxConsecutiveRepeatProcessor(max_consecutive=6),
    ])
    out = model.generate(
        ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        logits_processor=procs,
        no_repeat_ngram_size=3
    )
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True), diag["had_nan"]

# ============================================================
# 8) Build CLEAN cache (no processors) and run flips
#    — systematic over bit_class, random within-class bits
# ============================================================
CLEAN_CACHE = {p: generate_tail_clean(p, MAX_NEW_TOKENS) for p in TEST_PROMPTS}

rows = []
for rank in ranks_to_test:
    tensor_name, coord = coords_list[rank-1]
    W        = param_dict[tensor_name]
    orig_val = W.data[coord].clone()

    # systematic over each bit_class
    for bit_class, pool in BIT_CLASSES.items():
        for trial in range(1, N_TRIALS_PER_CLASS + 1):
            # pick a random bit within this class
            bit = random.choice(pool)

            # flip that bit
            W.data[coord] = flip_bit(orig_val, bit)
            try:
                for prompt in TEST_PROMPTS:
                    clean_out, = (CLEAN_CACHE[prompt],)
                    corrupt_out, had_nan = generate_tail_corrupt(prompt, MAX_NEW_TOKENS)

                    scores = score_pair(clean_out, corrupt_out)
                    scores.update(llm_judge_score(prompt, clean_out, corrupt_out))

                    rows.append({
                        "rank": rank,
                        "tensor": tensor_name,
                        "coord": tuple(map(int, coord)),
                        "bit_class": bit_class,
                        "bit_index": bit,
                        "trial": trial,
                        "prompt": prompt,
                        "clean": clean_out,
                        "corrupt": corrupt_out,
                        "corrupt_logits_had_nan": had_nan,
                        **scores
                    })
            finally:
                # restore before next trial
                W.data[coord] = orig_val

df = pd.DataFrame(rows)

# ============================================================
# 9) Pretty tables with borders + CSVs
# ============================================================
metric_cols = [c for c in [
    "EditDist","EditDist_Norm","BLEU","METEOR","BERTScore_F1",
    "ROUGE1_F1","ROUGE2_F1","ROUGEL_F1","BLEURT"
] if c in df.columns]
base_cols   = ["rank","tensor","coord","bit_class","bit_index","trial","prompt","clean","corrupt","corrupt_logits_had_nan"]
display_cols= base_cols + metric_cols

def _truncate(s, w=MAX_COL_WIDTH):
    if not isinstance(s, str): return s
    return s if len(s) <= w else s[:w-1] + "…"

if not df.empty:
    df_disp = df[display_cols].copy()
    for col in ["prompt","clean","corrupt"]:
        df_disp[col] = df_disp[col].map(lambda x: _truncate(x, MAX_COL_WIDTH))
    print("\n" + "="*16 + " Per-trial results (preview) " + "="*16)
    print(tabulate(df_disp.head(30), headers="keys", tablefmt="grid", showindex=False))

    df_local_path = "bitflip_per_trial.csv"
    df.to_csv(df_local_path, index=False)
    print(f"\nSaved full per-trial results → {os.path.abspath(df_local_path)}")

    agg_map = {m: ["mean", "median", "std"] for m in metric_cols}
    if metric_cols:
        summary = df.groupby(
            ["rank","tensor","coord","bit_class","prompt"], as_index=False
        ).agg(agg_map)
        if isinstance(summary.columns, pd.MultiIndex):
            summary.columns = ["_".join([str(c) for c in col if c]) for col in summary.columns]
        else:
            summary.columns = summary.columns.astype(str)
        summary["prompt"] = summary["prompt"].map(lambda x: _truncate(x, MAX_COL_WIDTH))

        print("\n" + "="*14 + " Aggregated over trials " + "="*14)
        print(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))

        summary_local_path = "bitflip_aggregated.csv"
        summary.to_csv(summary_local_path, index=False)
        print(f"\nSaved aggregated results → {os.path.abspath(summary_local_path)}")
    else:
        print("\nNo metric columns to aggregate — skipping aggregated table.")
else:
    print("\nNo rows produced — check TOP_K, V_SELECT, or N_TRIALS_PER_CLASS settings.")

# ============================================================
# 10) ALSO save CSVs to Google Drive (MyDrive/bitflip_outputs)
# ============================================================
try:
    from google.colab import drive
    drive.mount("/content/drive")
    GDRIVE_DIR = "/content/drive/MyDrive/bitflip_outputs"
    os.makedirs(GDRIVE_DIR, exist_ok=True)

    if not df.empty:
        df.to_csv(os.path.join(GDRIVE_DIR, "bitflip_per_trial.csv"), index=False)
        print(f"\nCopied per-trial CSV to Google Drive → {GDRIVE_DIR}/bitflip_per_trial.csv")

    if 'summary' in locals():
        summary.to_csv(os.path.join(GDRIVE_DIR, "bitflip_aggregated.csv"), index=False)
        print(f"Copied aggregated CSV to Google Drive → {GDRIVE_DIR}/bitflip_aggregated.csv")
except Exception as e:
    print("\n[Warning] Could not save to Google Drive. If not using Colab, ignore this.")
    print("Error:", e)

# ---- code cell ----
# ============================================================
# 0.  One-off upgrade so WikiText-103 loads LOCALLY
# ============================================================
import subprocess, sys, math, random, torch, torch.nn.functional as F
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--upgrade", "datasets>=2.18.0", "fsspec>=2023.6.0"],
    check=True
)

# ============================================================
# 1.  Imports & config
# ============================================================
from datasets     import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "gpt2"           # GPT-2-small (117 M)
SEQ_LEN     = 1_024
BATCH_SIZE  = 8
MAX_STEPS   = 2                # mini-batches scanned
TOP_K       = 10               # global top-K scalars to print

# ============================================================
# 2.  Model & tokenizer
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ============================================================
# 3.  Load WikiText-103 (train split) and helpers
# ============================================================
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

def chunk_generator():
    cache = []
    for doc in wiki:
        cache.extend(tok(doc["text"]).input_ids)
        while len(cache) >= SEQ_LEN + 1:
            win, cache = cache[:SEQ_LEN + 1], cache[SEQ_LEN + 1:]
            yield win[:-1], win[1:]

def get_batch(gen, bs=BATCH_SIZE):
    buf = []
    for x, _ in gen:
        buf.append(x)
        if len(buf) == bs:
            yield torch.tensor(buf, device=DEVICE)
            buf = []

# ============================================================
# 4.  Scan MAX_STEPS batches → global top-K |∂L/∂θ|
# ============================================================
param_dict  = {n: p for n, p in model.named_parameters() if p.requires_grad}
running_max = {n: torch.zeros_like(p, device="cpu") for n, p in param_dict.items()}

for step, inp in enumerate(get_batch(chunk_generator()), 1):
    model.zero_grad(set_to_none=True)
    model(inp, labels=inp).loss.backward()
    for name, p in param_dict.items():
        running_max[name] = torch.maximum(
            running_max[name],
            p.grad.detach().abs().to("cpu")
        )
    if step >= MAX_STEPS:
        break

# gather local top-K per tensor, then global top-K
candidates = []
for name, rm in running_max.items():
    k_local = min(TOP_K, rm.numel())
    if k_local == 0:
        continue
    vals, idxs = torch.topk(rm.view(-1), k_local)
    for v, flat_idx in zip(vals, idxs):
        coord = torch.unravel_index(flat_idx, rm.shape)
        candidates.append((v.item(), name, coord))

candidates.sort(key=lambda t: t[0], reverse=True)
topk_entries = candidates[:TOP_K]

# ============================================================
# 5.  Report (with token lookup for embeddings)
# ============================================================
EMB_NAME = "transformer.wte.weight"  # name of the token-embedding parameter

print(f"\nGlobal Top-{TOP_K} most-sensitive scalars "
      f"(scanned {MAX_STEPS}×{BATCH_SIZE} windows):")
for rank, (val, tname, coord) in enumerate(topk_entries, 1):
    coord_str = ", ".join(map(str, coord))
    line = f"  #{rank:2d}: {tname}({coord_str})   |∂L/∂θ|max = {val:.3e}"
    if tname == EMB_NAME:
        token_id = int(coord[0])
        token_str = tok.convert_ids_to_tokens(token_id)
        line += f"   ← token id {token_id!r} = {token_str!r}"
    print(line)


# ---- code cell ----
# ============================================================
# 0) One-off setup (installs)
# ============================================================
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--upgrade",
     "datasets>=2.18.0", "fsspec>=2023.6.0",
     "pandas>=2.0.0", "sacrebleu>=2.4.0",
     "evaluate>=0.4.2", "rouge-score>=0.1.2",
     "bert-score>=0.3.13", "tabulate>=0.9.0"],
    check=True
)

# ============================================================
# 1) Imports & config
# ============================================================
import math, random, os, torch, torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import sacrebleu, evaluate
from tabulate import tabulate

# ---- experiment knobs ----
SEED = 123
random.seed(SEED); torch.manual_seed(SEED)

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID           = "gpt2"
SPLIT              = "train"
SEQ_LEN            = 1024
BATCH_SIZE         = 4
MAX_STEPS          = 4
TOP_K              = 3
V_SELECT           = "all"
N_TRIALS_PER_CLASS = 5
MAX_NEW_TOKENS     = 20
MAX_COL_WIDTH      = 100

TEST_PROMPTS = [
    "The weather today is",
    "The patient should take",
    "The bank transfer amount is",
    "The recommended dose for a child is",
    "The evacuation order status is",
]

# enable metrics
ENABLE_BLEU       = True
ENABLE_METEOR     = True
ENABLE_BERTSCORE  = True
ENABLE_ROUGE      = True

# ============================================================
# 2) Model & tokenizer
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ============================================================
# 3) Data helpers
# ============================================================
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split=SPLIT)

def chunk_generator():
    cache = []
    for doc in wiki:
        cache.extend(tok(doc["text"]).input_ids)
        while len(cache) >= SEQ_LEN+1:
            win, cache = cache[:SEQ_LEN+1], cache[SEQ_LEN+1:]
            yield win[:-1], win[1:]

def get_batch(gen, bs=BATCH_SIZE):
    buf = []
    for x, _ in gen:
        buf.append(x)
        if len(buf) == bs:
            yield torch.tensor(buf, device=DEVICE)
            buf = []

# ============================================================
# 4) Gradient scan → find top-K sensitive coords
# ============================================================
param_dict  = {n: p for n, p in model.named_parameters() if p.requires_grad}
running_max = {n: torch.zeros_like(p, device="cpu") for n, p in param_dict.items()}

for step, inp in enumerate(get_batch(chunk_generator()), 1):
    model.zero_grad(set_to_none=True)
    model(inp, labels=inp).loss.backward()
    for name, p in param_dict.items():
        running_max[name] = torch.maximum(
            running_max[name],
            p.grad.detach().abs().to("cpu")
        )
    if step >= MAX_STEPS:
        break

candidates = []
for name, rm in running_max.items():
    k_local = min(TOP_K, rm.numel())
    if k_local == 0:
        continue
    vals, idxs = torch.topk(rm.view(-1), k_local)
    for v, flat_idx in zip(vals, idxs):
        coord = torch.unravel_index(flat_idx, rm.shape)
        candidates.append((v.item(), name, coord))

candidates.sort(key=lambda t: t[0], reverse=True)
topk_entries = candidates[:TOP_K]
coords_list  = [(n, c) for _, n, c in topk_entries]

print(f"\nGlobal Top-{TOP_K} |∂L/∂θ| scalars:")
for rank, (v, n, c) in enumerate(topk_entries, 1):
    print(f"  #{rank}: {n}{tuple(map(int,c))}   |grad|={v:.3e}")

def normalize_v_select(sel, k):
    if sel == "all": return list(range(1, k+1))
    if isinstance(sel, int): return [sel]
    if isinstance(sel, (list, tuple)): return list(sel)
    raise ValueError
ranks_to_test = normalize_v_select(V_SELECT, TOP_K)
print(f"\nTesting ranks: {ranks_to_test}")

# ============================================================
# 5) Bit-flip helpers
# ============================================================

def flip_bit(val_tensor: torch.Tensor, bit: int):
    """Flip the specified bit of a float32 tensor element-by-element."""
    if val_tensor.dtype != torch.float32:
        raise TypeError("flip_bit expects a float32 tensor")
    if bit < 0 or bit > 31:
        raise ValueError("bit index must be in [0, 31]")
    device = val_tensor.device
    # Work on CPU for bit manipulation then move back to the original device
    np_view = val_tensor.detach().cpu().numpy().copy().view(np.uint32)
    np_view ^= (1 << bit)
    flipped = torch.from_numpy(np_view.view(np.float32)).to(device)
    return flipped.view_as(val_tensor)
BIT_CLASSES = {
    "sign":     [31],
    "exponent": list(range(23,31)),
    "mantissa": list(range(0,23)),
}

# ============================================================
# 6) Load metrics
# ============================================================
meteor = evaluate.load("meteor")    if ENABLE_METEOR    else None
berts  = evaluate.load("bertscore") if ENABLE_BERTSCORE else None
rouge  = evaluate.load("rouge")     if ENABLE_ROUGE    else None

# ============================================================
# 7) Scoring function
# ============================================================
def edit_distance(a: str, b: str):
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[m]

def score_pair(clean: str, corrupt: str):
    scores = {}
    ed = edit_distance(clean, corrupt)
    scores["EditDist"]      = float(ed)
    scores["EditDist_Norm"] = float(ed / max(1, len(clean)))

    if ENABLE_BLEU:
        scores["BLEU"] = sacrebleu.corpus_bleu([corrupt], [[clean]]).score
    if meteor is not None:
        scores["METEOR"] = float(meteor.compute(predictions=[corrupt], references=[clean])["meteor"])
    if berts is not None:
        bs = berts.compute(predictions=[corrupt], references=[clean], lang="en")
        scores["BERTScore_F1"] = float(bs["f1"][0])
    if rouge is not None:
        r = rouge.compute(predictions=[corrupt], references=[clean], use_stemmer=True)
        scores["ROUGE1_F1"] = float(r["rouge1"])
        scores["ROUGE2_F1"] = float(r["rouge2"])
        scores["ROUGEL_F1"] = float(r["rougeL"])

    return scores

# ============================================================
# 8) Generation helpers (no clamping)
# ============================================================
class NanInfDetector(LogitsProcessor):
    def __init__(self, flag_dict=None):
        self.flag_dict = flag_dict
    def __call__(self, input_ids, scores):
        if self.flag_dict is not None and (torch.isnan(scores).any() or torch.isinf(scores).any()):
            self.flag_dict["had_nan"] = True
        return scores

class MaxRepeatGuard(LogitsProcessor):
    def __init__(self, max_consecutive=6):
        self.max_consecutive = max_consecutive
    def __call__(self, input_ids, scores):
        if input_ids.size(0) != 1 or input_ids.size(1) == 0:
            return scores
        seq, last = input_ids[0], input_ids[0,-1].item()
        run = 0
        for t in range(seq.size(0)-1, -1, -1):
            if seq[t].item() == last:
                run += 1
            else:
                break
        if run >= self.max_consecutive:
            scores[:, last] = -1e9
        return scores

@torch.no_grad()
def generate_clean(prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True)

@torch.no_grad()
def generate_corrupt(prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    diag = {"had_nan": False}
    procs = LogitsProcessorList([
        NanInfDetector(flag_dict=diag),
        MaxRepeatGuard(max_consecutive=6),
    ])
    out = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
                         logits_processor=procs, no_repeat_ngram_size=3)
    text = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
    return text, diag["had_nan"]

# ============================================================
# 9) Run trials & collect
# ============================================================
CLEAN_CACHE = {p: generate_clean(p) for p in TEST_PROMPTS}
rows = []

for rank in ranks_to_test:
    tensor_name, coord = coords_list[rank-1]
    W = param_dict[tensor_name]
    orig = W.data[coord].clone()

    for bit_class, pool in BIT_CLASSES.items():
        for trial in range(1, N_TRIALS_PER_CLASS+1):
            bit = random.choice(pool)
            W.data[coord] = flip_bit(orig, bit)
            try:
                for prompt in TEST_PROMPTS:
                    clean_out = CLEAN_CACHE[prompt]
                    corrupt_out, had_nan = generate_corrupt(prompt)
                    scores = score_pair(clean_out, corrupt_out)
                    rows.append({
                        "rank": rank,
                        "tensor": tensor_name,
                        "coord": tuple(map(int, coord)),
                        "bit_class": bit_class,
                        "bit_index": bit,
                        "trial": trial,
                        "prompt": prompt,
                        "clean": clean_out,
                        "corrupt": corrupt_out,
                        "corrupt_logits_had_nan": had_nan,
                        **scores
                    })
            finally:
                W.data[coord] = orig

df = pd.DataFrame(rows)

# ============================================================
# 10) Tabulate & save
# ============================================================
metric_cols = [c for c in [
    "EditDist","EditDist_Norm","BLEU","METEOR",
    "BERTScore_F1","ROUGE1_F1","ROUGE2_F1","ROUGEL_F1"
] if c in df.columns]
base_cols   = ["rank","tensor","coord","bit_class","bit_index","trial","prompt","clean","corrupt","corrupt_logits_had_nan"]
display_cols= base_cols + metric_cols

def _truncate(s, w=MAX_COL_WIDTH):
    return s if not isinstance(s, str) or len(s) <= w else s[:w-1] + "…"

if not df.empty:
    df_disp = df[display_cols].copy()
    for col in ["prompt","clean","corrupt"]:
        df_disp[col] = df_disp[col].map(lambda x: _truncate(x))
    print("\n" + "="*16 + " Per-trial results " + "="*16)
    print(tabulate(df_disp.head(30), headers="keys", tablefmt="grid", showindex=False))

    df.to_csv("bitflip_per_trial.csv", index=False)
    print(f"\nSaved → {os.path.abspath('bitflip_per_trial.csv')}")

    if metric_cols:
        summary = df.groupby(
            ["rank","tensor","coord","bit_class","prompt"], as_index=False
        ).agg({m: ["mean","median","std"] for m in metric_cols})
        if isinstance(summary.columns, pd.MultiIndex):
            summary.columns = ["_".join(filter(None, c)) for c in summary.columns]
        summary["prompt"] = summary["prompt"].map(lambda x: _truncate(x))

        print("\n" + "="*14 + " Aggregated results " + "="*14)
        print(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))

        summary.to_csv("bitflip_aggregated.csv", index=False)
        print(f"\nSaved → {os.path.abspath('bitflip_aggregated.csv')}")
else:
    print("No results — check your settings.")

# ---- code cell ----
import pandas as pd
pd.set_option('display.max_rows', None)    # show all rows
pd.set_option('display.max_columns', None) # show all columns
df = pd.read_csv('/content/bitflip_per_trial.csv')
df

# ---- code cell ----
# ============================================================
# 0) One-off setup (installs)
# ============================================================
%env CUDA_LAUNCH_BLOCKING=1
import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--upgrade",
     "datasets>=2.18.0", "fsspec>=2023.6.0",
     "pandas>=2.0.0", "sacrebleu>=2.4.0",
     "evaluate>=0.4.2", "rouge-score>=0.1.2",
     "bert-score>=0.3.13", "tabulate>=0.9.0"],
    check=True
)

# ============================================================
# 1) Imports & Config
# ============================================================
import os
import math
import random
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import sacrebleu
import evaluate
from tabulate import tabulate

# ---- Experiment knobs ----
SEED               = 123
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID           = "gpt2"
SPLIT              = "train"
SEQ_LEN            = 1024
BATCH_SIZE         = 16
MAX_STEPS          = 1100
TOP_K              = 3
V_SELECT           = "all"
N_TRIALS_PER_CLASS = 5
MAX_NEW_TOKENS     = 100
MAX_COL_WIDTH      = 100

# Decoding strategy: "greedy", "top-k", or "top-p"
DECODING_STRATEGY = "greedy"
TOP_K_SAMPLING    = 10
TOP_P_SAMPLING    = 0.9
TEMPERATURE       = 1.0

TEST_PROMPTS = [
    "The weather today is",
    "The patient should take",
    "The bank transfer amount is",
    "The recommended dose for a child is",
    "The evacuation order status is",
]

# Metric toggles
ENABLE_BLEU       = True
ENABLE_METEOR     = True
ENABLE_BERTSCORE  = True
ENABLE_ROUGE      = True

# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 2) Model & Tokenizer
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ============================================================
# 3) Load & check metrics
# ============================================================
meteor = evaluate.load("meteor")    if ENABLE_METEOR    else None
berts  = evaluate.load("bertscore") if ENABLE_BERTSCORE else None
rouge  = evaluate.load("rouge")     if ENABLE_ROUGE    else None

print(f"METEOR loaded: {meteor}")
print(f"BERTScore loaded: {berts}")
print(f"ROUGE loaded:   {rouge}")

# ============================================================
# 4) Data & gradient scan for top-K sensitive coords
# ============================================================
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split=SPLIT)

def chunk_generator():
    cache = []
    for doc in wiki:
        cache.extend(tok(doc["text"]).input_ids)
        while len(cache) >= SEQ_LEN + 1:
            win, cache = cache[:SEQ_LEN+1], cache[SEQ_LEN+1:]
            yield win[:-1], win[1:]

def get_batch(gen, bs=BATCH_SIZE):
    buf = []
    for x, _ in gen:
        buf.append(x)
        if len(buf) == bs:
            yield torch.tensor(buf, device=DEVICE)
            buf = []

param_dict  = {n: p for n, p in model.named_parameters() if p.requires_grad}
running_max = {n: torch.zeros_like(p, device="cpu") for n, p in param_dict.items()}

for step, inp in enumerate(get_batch(chunk_generator()), 1):
    model.zero_grad(set_to_none=True)
    model(inp, labels=inp).loss.backward()
    for name, p in param_dict.items():
        running_max[name] = torch.maximum(
            running_max[name],
            p.grad.detach().abs().to("cpu")
        )
    if step >= MAX_STEPS:
        break

candidates = []
for name, rm in running_max.items():
    k_local = min(TOP_K, rm.numel())
    if k_local == 0:
        continue
    vals, idxs = torch.topk(rm.view(-1), k_local)
    for v, flat in zip(vals, idxs):
        coord = torch.unravel_index(flat, rm.shape)
        candidates.append((v.item(), name, coord))

candidates.sort(key=lambda t: t[0], reverse=True)
topk_entries = candidates[:TOP_K]
coords_list  = [(name, coord) for _, name, coord in topk_entries]

print(f"\nGlobal Top-{TOP_K} |∂L/∂θ| scalars:")
for rank, (val, name, coord) in enumerate(topk_entries, 1):
    print(f"  #{rank}: {name}{tuple(map(int,coord))}  |grad|={val:.3e}")

def normalize_v_select(sel, k):
    if sel == "all":
        return list(range(1, k+1))
    if isinstance(sel, int):
        return [sel]
    if isinstance(sel, (list, tuple)):
        return list(sel)
    raise ValueError("V_SELECT must be 'all', int, or list[int]'")

ranks_to_test = normalize_v_select(V_SELECT, TOP_K)
print(f"\nTesting ranks: {ranks_to_test}")

# ============================================================
# 5) Bit-flip helpers
# ============================================================

def flip_bit(val_tensor: torch.Tensor, bit: int):
    """Flip the specified bit of a float32 tensor element-by-element."""
    if val_tensor.dtype != torch.float32:
        raise TypeError("flip_bit expects a float32 tensor")
    if bit < 0 or bit > 31:
        raise ValueError("bit index must be in [0, 31]")
    device = val_tensor.device
    # Work on CPU for bit manipulation then move back to the original device
    np_view = val_tensor.detach().cpu().numpy().copy().view(np.uint32)
    np_view ^= (1 << bit)
    flipped = torch.from_numpy(np_view.view(np.float32)).to(device)
    return flipped.view_as(val_tensor)
BIT_CLASSES = {
    "sign":     [31],
    "exponent": list(range(23, 31)),
    "mantissa": list(range(0, 23)),
}

# ============================================================
# 6) Scoring function with try/except
# ============================================================
def edit_distance(a: str, b: str):
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j-1] + 1, prev + cost), dp[j]
    return dp[m]

def score_pair(clean: str, corrupt: str):
    scores = {}
    ed = edit_distance(clean, corrupt)
    scores["EditDist"]      = float(ed)
    scores["EditDist_Norm"] = float(ed / max(1, len(clean)))

    if ENABLE_BLEU:
        try:
            scores["BLEU"] = sacrebleu.corpus_bleu([corrupt], [[clean]]).score
        except Exception as e:
            print("BLEU compute failed:", e)

    if ENABLE_METEOR and meteor is not None:
        try:
            scores["METEOR"] = float(
                meteor.compute(predictions=[corrupt], references=[clean])["meteor"]
            )
        except Exception as e:
            print("METEOR compute failed:", e)

    if ENABLE_BERTSCORE and berts is not None:
        try:
            bs = berts.compute(
                predictions=[corrupt], references=[clean], lang="en"
            )
            scores["BERTScore_F1"] = float(bs["f1"][0])
        except Exception as e:
            print("BERTScore compute failed:", e)

    if ENABLE_ROUGE and rouge is not None:
        try:
            r = rouge.compute(
                predictions=[corrupt], references=[clean], use_stemmer=True
            )
            scores["ROUGE1_F1"] = float(r["rouge1"])
            scores["ROUGE2_F1"] = float(r["rouge2"])
            scores["ROUGEL_F1"] = float(r["rougeL"])
        except Exception as e:
            print("ROUGE compute failed:", e)

    return scores

# ============================================================
# 7) Generation helpers
# ============================================================
class NanInfDetector(LogitsProcessor):
    def __init__(self, flag_dict=None):
        self.flag_dict = flag_dict
    def __call__(self, input_ids, scores):
        if self.flag_dict and (torch.isnan(scores).any() or torch.isinf(scores).any()):
            self.flag_dict["had_nan"] = True
        return scores

class MaxRepeatGuard(LogitsProcessor):
    def __init__(self, max_consecutive=6):
        self.max_consecutive = max_consecutive
    def __call__(self, input_ids, scores):
        if input_ids.size(0) != 1 or input_ids.size(1) == 0:
            return scores
        seq, last = input_ids[0], input_ids[0, -1].item()
        run = 0
        for t in range(seq.size(0)-1, -1, -1):
            if seq[t].item() == last:
                run += 1
            else:
                break
        if run >= self.max_consecutive:
            scores[:, last] = -1e9
        return scores

@torch.no_grad()
def _generate(prompt: str, corrupt: bool = False):
    enc = tok(prompt, return_tensors="pt", return_attention_mask=True)
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    do_sample = DECODING_STRATEGY != "greedy"
    gen_kwargs = {"temperature": TEMPERATURE}
    if DECODING_STRATEGY == "top-k":
        gen_kwargs["top_k"] = TOP_K_SAMPLING
    elif DECODING_STRATEGY == "top-p":
        gen_kwargs["top_p"] = TOP_P_SAMPLING

    if not corrupt:
        out = model.generate(
            input_ids=ids,
            attention_mask=mask,
            do_sample=do_sample,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            **gen_kwargs
        )
        return tok.decode(out[0, ids.size(1):], skip_special_tokens=True), False

    diag = {"had_nan": False}
    procs = LogitsProcessorList([
        NanInfDetector(flag_dict=diag),
        MaxRepeatGuard(max_consecutive=6),
    ])
    out = model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=do_sample,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        logits_processor=procs,
        no_repeat_ngram_size=3,
        **gen_kwargs
    )
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True), diag["had_nan"]

generate_clean   = lambda p: _generate(p, corrupt=False)[0]
generate_corrupt = lambda p: _generate(p, corrupt=True)

# ============================================================
# 8) Run bit-flip trials
# ============================================================
CLEAN_CACHE = {p: generate_clean(p) for p in TEST_PROMPTS}
rows = []
for rank in ranks_to_test:
    name, coord = coords_list[rank-1]
    W, orig = param_dict[name], param_dict[name].data[coord].clone()
    for bit_class, pool in BIT_CLASSES.items():
        for trial in range(1, N_TRIALS_PER_CLASS + 1):
            bit = random.choice(pool)
            W.data[coord] = flip_bit(orig, bit)
            try:
                for prompt in TEST_PROMPTS:
                    clean_out = CLEAN_CACHE[prompt]
                    corrupt_out, had_nan = generate_corrupt(prompt)
                    scores = score_pair(clean_out, corrupt_out)
                    rows.append({
                        "rank": rank,
                        "tensor": name,
                        "coord": tuple(map(int, coord)),
                        "bit_class": bit_class,
                        "bit_index": bit,
                        "trial": trial,
                        "prompt": prompt,
                        "clean": clean_out,
                        "corrupt": corrupt_out,
                        "corrupt_logits_had_nan": had_nan,
                        **scores
                    })
            finally:
                W.data[coord] = orig

df = pd.DataFrame(rows)

# ============================================================
# 9) Debug: show final columns & sample
# ============================================================
print("Final DataFrame columns:", df.columns.tolist())
print(df.head(3))

# ============================================================
# 10) Save CSVs to Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

GDRIVE_DIR = '/content/drive/MyDrive/bitflip_outputs'
os.makedirs(GDRIVE_DIR, exist_ok=True)

# Per-trial CSV
trial_path = os.path.join(GDRIVE_DIR, 'bitflip_per_trial.csv')
df.to_csv(trial_path, index=False)
print(f"Saved per-trial CSV → {trial_path}")

# Aggregated CSV
metric_cols = [c for c in [
    "EditDist","EditDist_Norm","BLEU","METEOR",
    "BERTScore_F1","ROUGE1_F1","ROUGE2_F1","ROUGEL_F1"
] if c in df.columns]
print("Aggregated metrics present:", metric_cols)
if metric_cols:
    summary = df.groupby(
        ["rank","tensor","coord","bit_class","prompt"],
        as_index=False
    ).agg({m: ["mean","median","std"] for m in metric_cols})
    if isinstance(summary.columns, pd.MultiIndex):
        summary.columns = ["_".join(filter(None, c)) for c in summary.columns]
    agg_path = os.path.join(GDRIVE_DIR, 'bitflip_aggregated.csv')
    summary.to_csv(agg_path, index=False)
    print(f"Saved aggregated CSV → {agg_path}")
