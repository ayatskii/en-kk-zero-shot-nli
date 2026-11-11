"""Step 2 - Zero-shot NLI evaluation of one model on parallel EN and KK items.

Usage: python evaluate.py <hf_model_id> [n_items]
Decision rule: constrained label scoring. One forward pass per item; the predicted
label is the one whose first answer token has the highest next-token logit after the
assistant turn starts. No sampling -> fully deterministic.
The instruction is always in English; only premise/hypothesis language changes,
so the language of the items is the only manipulated condition.
"""
import sys, time, json, platform
import pandas as pd, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else None
LABELS = ["entailment", "neutral", "contradiction"]
torch.manual_seed(0)
torch.set_num_threads(2)

SYSTEM = "You are a careful annotator for a natural language inference task."
TEMPLATE = ("Read the premise and the hypothesis.\n"
            "Premise: {p}\nHypothesis: {h}\n\n"
            "Does the premise entail the hypothesis, contradict it, or is it neutral? "
            "Answer with exactly one word: entailment, neutral, or contradiction.")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval()

# First token id of each label as the assistant would start the answer
label_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LABELS]
assert len(set(label_ids)) == 3, f"label first-tokens collide: {label_ids}"

def build(p, h):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(p=p, h=h)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

df = pd.read_csv("data/pilot_sample.csv")
if N: df = df.head(N)
rows = []
t0 = time.time()
for i, r in df.iterrows():
    for lang in ["en", "kk"]:
        p, h = r[f"premise_{lang}"], r[f"hypothesis_{lang}"]
        enc = tok(build(p, h), return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float()
        scores = logits[label_ids]
        probs = torch.softmax(scores, -1).tolist()
        pred = LABELS[int(scores.argmax())]
        n_tok_pair = len(tok.encode(p + " " + h, add_special_tokens=False))
        n_words = len((p + " " + h).split())
        rows.append({"xnli_id": r.xnli_id, "lang": lang, "gold": r.label, "pred": pred,
                     "correct": pred == r.label, "p_ent": probs[0], "p_neu": probs[1],
                     "p_con": probs[2], "pair_tokens": n_tok_pair, "pair_words": n_words,
                     "fertility": n_tok_pair / n_words})
    if (i + 1) % 25 == 0:
        print(f"{MODEL}: {i+1}/{len(df)} items, {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
safe = MODEL.split("/")[-1]
out.to_csv(f"results/preds_{safe}.csv", index=False)
meta = {"model": MODEL, "items": len(df), "seconds": round(time.time() - t0, 1),
        "dtype": "bfloat16", "decoding": "constrained first-token argmax (deterministic)",
        "label_first_token_ids": dict(zip(LABELS, label_ids)),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "cpu_threads": torch.get_num_threads()}
json.dump(meta, open(f"results/meta_{safe}.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
