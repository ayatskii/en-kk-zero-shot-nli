"""Step 2c - Zero-shot NLI evaluation of a GGUF model served by llama.cpp's llama-server
(batch 2: MoE models with experts offloaded to CPU RAM; batch 3: gpt-oss-20b).

Start the server first (run_batch2.sh / run_batch3.sh do this), then:
  python evaluate_gguf.py [--url http://127.0.0.1:8080] [--name NAME] [--mode score|reason]
                          [--harmony] [--effort low|medium|high] [--n N]

--mode score  (default) same decision rule as evaluate.py: one forward pass, predicted
              label = highest next-token log-probability among the first tokens of the
              three labels. Prompt = the model's own chat template (rendered by the server)
              + assistant prefix. Deterministic.
--harmony     gpt-oss uses the "harmony" format, where the assistant normally writes a
              hidden analysis channel first. In score mode we pre-fill the start of the
              final channel ("<|channel|>final<|message|>") so the next token is the answer,
              i.e. reasoning is skipped; this keeps the decision rule identical to the
              other models.
--mode reason the model may reason freely (temperature 0, gpt-oss reasoning effort via
              --effort); the label is parsed from the final answer text. Different
              decision rule and much slower - report separately (use --n for a subset).
"""
import argparse, json, re, time, platform, urllib.request
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--name", default=None, help="output name (default: GGUF file name)")
ap.add_argument("--mode", choices=["score", "reason"], default="score")
ap.add_argument("--harmony", action="store_true", help="gpt-oss: pre-fill the final channel")
ap.add_argument("--effort", choices=["low", "medium", "high"], default="low")
ap.add_argument("--max-tokens", type=int, default=2048, help="reason mode generation cap")
ap.add_argument("--n-probs", type=int, default=200)
ap.add_argument("--n", type=int, default=None)
args = ap.parse_args()
LABELS = ["entailment", "neutral", "contradiction"]
HARMONY_FINAL = "<|channel|>final<|message|>"

SYSTEM = "You are a careful annotator for a natural language inference task."
TEMPLATE = ("Read the premise and the hypothesis.\n"
            "Premise: {p}\nHypothesis: {h}\n\n"
            "Does the premise entail the hypothesis, contradict it, or is it neutral? "
            "Answer with exactly one word: entailment, neutral, or contradiction.")

def call(path, body=None):
    req = urllib.request.Request(args.url + path, method="POST" if body is not None else "GET",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())

def tokenize(text):
    return call("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

props = call("/props")
model_path = props.get("model_path", "unknown")
name = args.name or model_path.split("/")[-1].removesuffix(".gguf")
if args.mode == "reason":
    name += f"-reason-{args.effort}"
# Qwen3.x: skip thinking; gpt-oss: reasoning effort. Templates ignore unknown kwargs.
TPL_KW = {"enable_thinking": False, "reasoning_effort": args.effort}

def messages(p, h):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(p=p, h=h)}]

def build(p, h):
    s = call("/apply-template", {"messages": messages(p, h), "chat_template_kwargs": TPL_KW})["prompt"]
    return s + HARMONY_FINAL if args.harmony else s

# First token id of each label as the assistant would start the answer
label_ids = [tokenize(l)[0] for l in LABELS]
assert len(set(label_ids)) == 3, f"label first-tokens collide: {label_ids}"

def score(p, h):
    # Send token ids, not text: the template already contains BOS where needed,
    # and a token-array prompt is not given a second BOS by the server.
    res = call("/completion", {"prompt": tokenize(build(p, h)), "n_predict": 1, "temperature": 0,
                               "n_probs": args.n_probs, "cache_prompt": True})
    top = res["completion_probabilities"][0]["top_logprobs"]
    lp = {t["id"]: t["logprob"] for t in top}
    floor = min(lp.values()) - 1.0          # label outside the top-n: below everything listed
    missing = [l for l, i in zip(LABELS, label_ids) if i not in lp]
    s = [lp.get(i, floor) for i in label_ids]
    m = max(s); e = [2.718281828459045 ** (x - m) for x in s]
    probs = [x / sum(e) for x in e]
    return LABELS[s.index(m)], probs, {"labels_outside_top_n": ",".join(missing)}

def reason(p, h):
    res = call("/v1/chat/completions", {"messages": messages(p, h), "temperature": 0,
                                        "max_tokens": args.max_tokens,
                                        "chat_template_kwargs": TPL_KW})
    msg = res["choices"][0]["message"]
    text = (msg.get("content") or "").lower()
    found = re.findall(r"entailment|neutral|contradiction", text)
    pred = found[-1] if found else "invalid"   # last mention = final verdict
    probs = [float(pred == l) for l in LABELS]
    return pred, probs, {"completion_tokens": res.get("usage", {}).get("completion_tokens"),
                         "answer_text": text[:200]}

df = pd.read_csv("data/eval_set.csv")
if args.n: df = df.head(args.n)
print("model:", model_path, "| output name:", name, "| mode:", args.mode, flush=True)
print("prompt tail:", repr(build("P", "H")[-160:]), flush=True)
rows = []
t0 = time.time()
for i, r in df.iterrows():
    for lang in ["en", "kk"]:
        p, h = r[f"premise_{lang}"], r[f"hypothesis_{lang}"]
        pred, probs, extra = (score if args.mode == "score" else reason)(p, h)
        n_tok_pair = len(tokenize(p + " " + h))
        n_words = len((p + " " + h).split())
        rows.append({"xnli_id": r.xnli_id, "lang": lang, "gold": r.label, "pred": pred,
                     "correct": pred == r.label, "p_ent": probs[0], "p_neu": probs[1],
                     "p_con": probs[2], "pair_tokens": n_tok_pair, "pair_words": n_words,
                     "fertility": n_tok_pair / n_words, **extra})
    if (i + 1) % 50 == 0:
        print(f"{name}: {i+1}/{len(df)} items, {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
out.to_csv(f"results/preds_{name}.csv", index=False)
meta = {"model": model_path, "backend": "llama.cpp (llama-server)", "items": len(df),
        "seconds": round(time.time() - t0, 1),
        "quantization": (re.findall(r"(?:UD-)?(?:Q\d\w*|MXFP4\w*|BF16|F16)", model_path) or ["see file"])[-1],
        "mode": args.mode, "harmony_final_prefill": args.harmony,
        "decoding": ("constrained first-token argmax over server log-probs (deterministic)"
                     if args.mode == "score" else
                     f"free generation, temperature 0, reasoning effort {args.effort}, label parsed"),
        "chat_template_kwargs": TPL_KW, "label_first_token_ids": dict(zip(LABELS, label_ids)),
        "server_build": props.get("build_info"), "python": platform.python_version()}
if args.mode == "score":
    meta["items_with_label_outside_top_n"] = int((out.labels_outside_top_n != "").sum())
else:
    meta["invalid_answers"] = int((out.pred == "invalid").sum())
json.dump(meta, open(f"results/meta_{name}.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
