# Helpers shared by run_batch2.sh / run_batch3.sh: start llama-server for one GGUF,
# wait until it is ready, stop it afterwards.
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}   # path to the CUDA build of llama-server
PORT=${PORT:-8080}
URL="http://127.0.0.1:$PORT"

start_server() {  # $1 = HF repo:quant, $2 = log file, rest = extra llama-server args
  local hf=$1 log=$2; shift 2
  # fail fast: llama-server only binds the port after the (long) model download
  if ss -ltn | grep -q ":$PORT\b"; then
    echo "!!! port $PORT is already in use - rerun with PORT=<free port>, e.g. PORT=8091"; return 1
  fi
  # -fit (default on) places as much as fits on the GPU; MoE experts that do not fit
  # stay in CPU RAM. -np 1: one request slot, whole context for it. No vision projector.
  "$LLAMA_SERVER" -hf "$hf" --no-mmproj -c "${CTX:-4096}" -np 1 --port "$PORT" \
      ${LLAMA_ARGS:-} "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  echo "llama-server pid $SERVER_PID, log $log (first run downloads the model)"
  until curl -sf "$URL/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then echo "!!! llama-server exited - see $log"; return 1; fi
    sleep 5
  done
  echo "server ready"
}

stop_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill "$SERVER_PID" 2> /dev/null; wait "$SERVER_PID" 2> /dev/null
  SERVER_PID=
  return 0
}
trap stop_server EXIT
