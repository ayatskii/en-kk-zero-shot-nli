#!/usr/bin/env bash
# Batch 2 - everything after batch 1, in this order:
#   1. Qwen3.5-9B, Gemma-4-E4B          transformers 4-bit, largest embedding table in CPU RAM
#   2. Gemma-4-26B, Qwen3-30B, Qwen3.5-35B   llama.cpp Q4_K_M MoE, experts offloaded to RAM
#   3. gpt-oss-20b                       llama.cpp MXFP4; run A = answer-only scoring (full set),
#                                        run B = reasoning allowed, label parsed (REASON_N items)
#   4. Llama-3.1-8B, Sherkala, KazLLM    transformers 4-bit, gated: need HF access (README)
# Finished models (results/preds_<name>.csv exists) are skipped, so rerun after access is granted.
#   LLAMA_SERVER=~/llama.cpp/build/bin/llama-server ./run_batch2.sh
#   N=20 ...            smoke test on the first 20 items
#   CLEAN_CACHE=1 ...   delete each transformers model's HF download after it finishes
#   LLAMA_ARGS="--n-cpu-moe 40" ...   manual expert placement if llama-server runs out of memory
#   SKIP_REASON=1 ...   skip gpt-oss run B;  EFFORT=medium ...  gpt-oss run B reasoning effort
set -uo pipefail
cd "$(dirname "$0")"
source llama_server_lib.sh
mkdir -p results/logs
N_ARG=${N:+--n $N}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less VRAM fragmentation

done_already() { [ -z "${N:-}" ] && [ -f "results/preds_$1.csv" ]; }

run_hf() {  # $1 = HF id, $2 = modules kept in CPU RAM (optional)
  local model=$1 cpu=${2:-} name=${1##*/}
  if done_already "$name"; then echo "skip $name (done)"; return; fi
  echo "=== $model (4bit)"
  if python evaluate.py "$model" --quant 4bit --cpu-modules "$cpu" $N_ARG 2>&1 \
       | tee "results/logs/$name.log"; then
    [ "${CLEAN_CACHE:-0}" = 1 ] && rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${model//\//--}"
  else
    echo "!!! $model failed - see results/logs/$name.log"
  fi
  return 0
}

# 1. models that needed their embedding table moved to RAM (4-bit does not shrink it)
run_hf Qwen/Qwen3.5-9B model.language_model.embed_tokens                  # 2 GB table
run_hf google/gemma-4-E4B-it model.language_model.embed_tokens_per_layer  # 5.6 GB table

# 2. MoE models - all plain Q4_K_M so the quantisation is comparable
MOE=(
  "bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M gemma-4-26B-A4B-it-Q4_K_M"        # 17.0 GB
  "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M Qwen3-30B-A3B-Instruct-2507-Q4_K_M"  # 18.6 GB
  "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M Qwen3.5-35B-A3B-Q4_K_M"                        # 22.0 GB, tightest
)
for entry in "${MOE[@]}"; do
  read -r hf name <<< "$entry"
  if done_already "$name"; then echo "skip $name (done)"; continue; fi
  echo "=== $hf"
  start_server "$hf" "results/logs/server_$name.log" || continue
  python evaluate_gguf.py --url "$URL" --name "$name" $N_ARG 2>&1 | tee "results/logs/$name.log" \
    || echo "!!! $name failed"
  stop_server
done

# 3. gpt-oss-20b
NAME=gpt-oss-20b-MXFP4
RNAME=$NAME-reason-${EFFORT:-low}
need_a=$(done_already "$NAME" || echo 1)
need_b=$([ "${SKIP_REASON:-0}" != 1 ] && ! done_already "$RNAME" && echo 1)
if [ -z "$need_a$need_b" ]; then
  echo "skip gpt-oss-20b (done)"
elif CTX=${CTX:-8192} start_server "ggml-org/gpt-oss-20b-GGUF:MXFP4" "results/logs/server_$NAME.log"; then
  if [ -n "$need_a" ]; then
    echo "=== gpt-oss run A: score mode (harmony final-channel pre-fill)"
    python evaluate_gguf.py --url "$URL" --name "$NAME" --harmony $N_ARG 2>&1 | tee "results/logs/$NAME.log"
  fi
  if [ -n "$need_b" ]; then
    echo "=== gpt-oss run B: reasoning mode, effort ${EFFORT:-low}, first ${REASON_N:-200} items"
    python evaluate_gguf.py --url "$URL" --name "$NAME" --mode reason --effort "${EFFORT:-low}" \
        --n "${N:-${REASON_N:-200}}" 2>&1 | tee "results/logs/$RNAME.log"
  fi
  stop_server
fi

# 4. gated models (last): base Llama-3.1-8B and its two Kazakh-adapted versions, ~6 GB VRAM
run_hf meta-llama/Llama-3.1-8B-Instruct
run_hf inception42/Llama-3.1-Sherkala-8B-Chat
run_hf issai/LLama-3.1-KazLLM-1.0-8B
