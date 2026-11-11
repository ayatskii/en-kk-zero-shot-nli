#!/usr/bin/env bash
# Batch 1 - models that fit in the 8 GB GPU (transformers + bitsandbytes) and the
# NLI classifier ("decision") models. Run from the project folder with the venv active.
#   CLEAN_CACHE=1 ./run_batch1.sh   -> delete each model's HF download after it finishes
#   N=20 ./run_batch1.sh            -> smoke test on the first 20 items
# Qwen3.5-9B, Gemma-4-E4B and the gated Llama-3.1 models are in run_batch2.sh.
set -uo pipefail
mkdir -p results/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less VRAM fragmentation
N_ARG=${N:+--n $N}

# "<hf id> <quant> [modules kept in CPU RAM]"  (bf16 if it fits in 8 GB, else 4-bit NF4)
LLMS=(
  "Qwen/Qwen2.5-0.5B-Instruct none"
  "Qwen/Qwen2.5-1.5B-Instruct none"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct none"
  "Qwen/Qwen3.5-0.8B none"
  "Qwen/Qwen3.5-2B none"
  "Qwen/Qwen3.5-4B 4bit"
)
NLI=(
  "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
  "joeddav/xlm-roberta-large-xnli"
)

clean() {  # remove one model from the HF cache (only when CLEAN_CACHE=1)
  [ "${CLEAN_CACHE:-0}" = 1 ] && rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}"
  return 0
}

for entry in "${LLMS[@]}"; do
  read -r model quant cpu <<< "$entry"
  name=${model##*/}
  if [ -z "${N:-}" ] && [ -f "results/preds_$name.csv" ]; then echo "skip $name (done)"; continue; fi
  echo "=== $model ($quant)"
  if python evaluate.py "$model" --quant "$quant" --cpu-modules "${cpu:-}" $N_ARG 2>&1 \
       | tee "results/logs/$name.log"; then
    clean "$model"
  else
    echo "!!! $model failed - see results/logs/$name.log"
  fi
done

for model in "${NLI[@]}"; do
  name=${model##*/}
  if [ -z "${N:-}" ] && [ -f "results/preds_$name.csv" ]; then echo "skip $name (done)"; continue; fi
  echo "=== $model (NLI classifier)"
  python evaluate_nli.py "$model" $N_ARG 2>&1 | tee "results/logs/$name.log" || echo "!!! $model failed"
done
