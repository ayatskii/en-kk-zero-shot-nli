# Zero-shot cross-lingual transfer EN→KK on NLI (Baurzhanova, Mergenbay, SE-2416)
RMT course project, Assignments 3–4. Release tag for Assignment 4: `assignment-4`.

**Purpose.** Measure how much zero-shot NLI accuracy open LLMs lose when the same XNLI items are
given in Kazakh instead of English (13 LLMs, a supervised NLI baseline and a majority baseline;
1,968 paired items).

## Reproduce the Assignment 4 tables and figures (no GPU, about one minute)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p figures
python analyze.py          # Table 2: accuracy, macro-F1, gap + CI, McNemar -> results/summary_table.csv, analysis.json
python extra_stats.py      # Holm, chance tests, retention, contrasts, GEE   -> results/extra_stats.json
python a4_analysis.py      # CIs, baselines, Cochran's Q, subgroups, prior-corrected rule, error sample, Fig. 1/2/4
python a4_error_figure.py  # error-type counts from results/a4_error_coding.csv, Fig. 3
```
These scripts read the raw prediction files `results/preds_<model>.csv` (one row per model × item ×
language) and regenerate every reported number. Inference is deterministic; to regenerate the
prediction files themselves see sections 1–2 below (GPU, `requirements-inference.txt`).

## Structure
| Path | Content |
|---|---|
| `prepare.py` | builds `data/eval_set.csv` (1,968 items) from Kardeş-NLU `xnli.test.kk.csv` |
| `evaluate.py`, `evaluate_gguf.py`, `evaluate_nli.py`, `run_batch*.sh` | inference (Transformers, llama.cpp, NLI classifiers) |
| `analyze.py`, `extra_stats.py`, `analyze_names.py` | frozen analysis (Assignment 3 plan); short model names |
| `a4_analysis.py`, `a4_error_figure.py` | Assignment 4 additions (post hoc analyses, error analysis) |
| `results/preds_*.csv`, `results/meta_*.json`, `results/logs/` | raw outputs, run metadata, run logs |
| `results/a4_error_sample.csv`, `results/a4_error_coding.csv` | 40 sampled flips and their manual codes |
| `figures/` | all figures; `pilot/` the 300-item pilot, unchanged |

## Data and licences
Code: MIT (see `LICENSE`). Data: Kardeş-NLU Kazakh XNLI test split
(github.com/lksenel/Kardes-NLU, commit 48eea77; Senel et al., 2024), derived from XNLI
(Conneau et al., 2018; CC BY-NC 4.0); used here for non-commercial academic evaluation.
`data/eval_set.csv` SHA-256: `aa161faa41189d227094d25595165c5cf7e59105973f9d221a64eded70a5fb18`.
No personal data, tokens or keys are stored in this repository.

---
# Original pipeline notes (Assignment 3)
Data: Kardeş-NLU data/kazakh/xnli.test.kk.csv (github.com/lksenel/Kardes-NLU, commit 48eea77)
Full eval set: 1,968 parallel items (2,000 minus 32 flagged), EN and KK → 3,936 predictions per model.
The 300-item pilot (scripts, results, figures) is archived unchanged in `pilot/`.

Tested on: Ryzen 5 7640HS, RTX 4060 Laptop 8 GB, 14 GB RAM, Arch Linux, Python 3.14.

## Model batches
| Batch | Script | Backend | Models |
|---|---|---|---|
| 1 | `run_batch1.sh` | transformers on GPU (bf16, or 4-bit NF4 when >8 GB) | Qwen2.5-0.5B/1.5B, SmolLM2-1.7B, Qwen3.5-0.8B/2B/4B; NLI classifiers mDeBERTa-v3-base-xnli, XLM-R-large-xnli |
| 2 | `run_batch2.sh` | in this order: 1. transformers 4-bit, embedding table in RAM · 2. llama.cpp Q4_K_M, MoE experts in RAM · 3. llama.cpp MXFP4 · 4. transformers 4-bit, gated | 1. Qwen3.5-9B, Gemma-4-E4B · 2. Gemma-4-26B-A4B, Qwen3-30B-A3B-2507, Qwen3.5-35B-A3B · 3. gpt-oss-20b (A: answer-only scoring, full set; B: with reasoning, 200 items) · 4. Llama-3.1-8B-Instruct (base), Sherkala-8B, KazLLM-8B (Kazakh-adapted Llama-3.1) |

All LLMs use the same prompt and the same decision rule (first answer token, argmax over the
three labels, deterministic), except gpt-oss run B. The NLI classifiers use their trained
3-way head.

## 1. Setup (once)
```bash
# Python: CUDA build of torch + quantisation
python -m venv .venv && source .venv/bin/activate
# --force-reinstall: a CPU torch of the same version would otherwise count as "already satisfied"
pip install --force-reinstall torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt -r requirements-inference.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True, RTX 4060

# Gated models (end of batch 2) - see "Gated model access" below
hf auth login

# llama.cpp with CUDA (batch 2)
sudo pacman -S --needed cuda base-devel cmake git
source /etc/profile.d/cuda.sh            # nvcc on PATH without re-login
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
# CUDA 13.3 needs GCC 15 as host compiler (system GCC is 16); 89 = RTX 40xx only
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
      -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build ~/llama.cpp/build --target llama-server -j 10
export LLAMA_SERVER=~/llama.cpp/build/bin/llama-server
```

## 2. Run
```bash
python prepare.py                       # -> data/eval_set.csv (already done)

N=10 ./run_batch1.sh                    # optional smoke test: first 10 items per model
rm -f results/preds_* results/meta_*    # ...then delete the smoke-test outputs
CLEAN_CACHE=1 ./run_batch1.sh           # batch 1 (CLEAN_CACHE frees each model's download after its run)
CLEAN_CACHE=1 ./run_batch2.sh           # batch 2 (needs LLAMA_SERVER, see setup)

python analyze.py                       # -> results/summary_table.csv, analysis.json, figures/
python extra_stats.py                   # -> results/extra_stats.json
```
Every batch skips models that already have `results/preds_<name>.csv`, so you can stop
(Ctrl-C) and resume. Logs go to `results/logs/`. One model can be run by hand, e.g.
`python evaluate.py Qwen/Qwen3.5-9B --quant 4bit`.

## Gated model access (last part of batch 2)
While logged in to huggingface.co, open each page, fill in the form at the top and submit:
| Model | Page | Approval |
|---|---|---|
| KazLLM-8B | https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B | automatic, instant (CC-BY-NC) |
| Sherkala-8B | https://huggingface.co/inception42/Llama-3.1-Sherkala-8B-Chat | automatic, instant (CC-BY-NC-SA) |
| Llama-3.1-8B-Instruct | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | manual by Meta: minutes to days; use your real name/affiliation |
Check status at https://huggingface.co/settings/gated-repos. The token used by `hf auth login`
must belong to the same account. A model without access fails with "403 … not in the
authorized list"; the batch logs it and moves on, so rerun `./run_batch2.sh` later
(finished models are skipped, only the missing ones run).

## Notes and caveats
- **Disk:** batch 1 ~25 GB, batch 2 ~150 GB (transformers ~70 GB, GGUF ~70 GB). Use
  `CLEAN_CACHE=1`, which frees each transformers model after its run. For llama.cpp, delete a finished GGUF from `~/.cache/huggingface/hub/` (or
  `~/.cache/llama.cpp/`) if you need space.
- **RAM:** close the browser and other apps for batch 2. Qwen3.5-35B-A3B (22 GB) is larger
  than VRAM + RAM, so parts are read from the SSD and it is the slowest. If llama-server runs
  out of memory, set `LLAMA_ARGS="--n-cpu-moe 40"` (more expert layers on the CPU).
- **Embedding tables in RAM:** 4-bit does not shrink embedding tables, so Qwen3.5-9B (2 GB
  table) and Gemma-4-E4B (5.6 GB per-layer table) keep them in CPU RAM via `--cpu-modules`.
  Predictions are identical to the all-GPU run (checked); Gemma needs ~11 GB free RAM while
  loading.
- **Quantisation is a confound:** batch 1 small models run in bf16, ≥4B in 4-bit NF4, batch 2
  in Q4_K_M, gpt-oss in its native MXFP4. Report the `quant` column of `summary_table.csv`.
- **Thinking models:** Qwen3/3.5 run with `enable_thinking=False`. gpt-oss run A pre-fills
  the harmony final channel (no reasoning). Run B allows low-effort reasoning, uses a
  different decision rule and must be reported separately.
- **gpt-oss template date:** the gpt-oss chat template inserts the current date, so run
  both gpt-oss runs in a single session for exact reproducibility.
- **NLI classifiers** are supervised on NLI (mDeBERTa also on English XNLI data), so they are
  a cross-lingual-transfer reference, not zero-shot.

## Report
`report/build.js` regenerates the report (.docx via docx-js, .pdf via headless Chromium) from
`results/*.json` and `figures/`: `cd report && npm install && node build.js ~/Downloads/Assignment3_Baurzhanova_Mergenbay_v3`
(needs `/usr/bin/chromium`; `render.js` is the shared docx/PDF renderer).
