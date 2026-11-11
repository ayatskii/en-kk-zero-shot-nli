"""Step 2b - NLI classifier ("decision model") evaluation on parallel EN and KK items.

Usage: python evaluate_nli.py <hf_model_id> [--n N] [--batch 32] [--device cuda|cpu]
These are encoder cross-encoders fine-tuned on NLI (MNLI/XNLI); none saw Kazakh NLI
data, so KK results measure cross-lingual transfer of a supervised classifier.
Decision rule: argmax over the model's 3 NLI logits (deterministic). The label order
differs between checkpoints, so it is read from config.id2label.
Output files have the same columns as evaluate.py, so analyze.py treats them alike.
"""
import argparse, time, json, platform
import pandas as pd, torch, transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
MODEL = args.model
LABELS = ["entailment", "neutral", "contradiction"]
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(args.device).eval()
id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
assert sorted(id2label.values()) == sorted(LABELS), f"unexpected labels: {id2label}"
order = [next(k for k, v in id2label.items() if v == l) for l in LABELS]  # logit column per LABELS entry

df = pd.read_csv("data/eval_set.csv")
if args.n: df = df.head(args.n)
rows = []
t0 = time.time()
for lang in ["en", "kk"]:
    P, H = df[f"premise_{lang}"].tolist(), df[f"hypothesis_{lang}"].tolist()
    for s in range(0, len(df), args.batch):
        enc = tok(P[s:s + args.batch], H[s:s + args.batch], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(args.device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits.float()[:, order], -1).cpu()
        for j, pr in enumerate(probs.tolist()):
            r = df.iloc[s + j]
            p, h = P[s + j], H[s + j]
            pred = LABELS[max(range(3), key=pr.__getitem__)]
            n_tok_pair = len(tok.encode(p + " " + h, add_special_tokens=False))
            n_words = len((p + " " + h).split())
            rows.append({"xnli_id": r.xnli_id, "lang": lang, "gold": r.label, "pred": pred,
                         "correct": pred == r.label, "p_ent": pr[0], "p_neu": pr[1],
                         "p_con": pr[2], "pair_tokens": n_tok_pair, "pair_words": n_words,
                         "fertility": n_tok_pair / n_words})
    print(f"{MODEL}: {lang} done, {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
safe = MODEL.split("/")[-1]
out.to_csv(f"results/preds_{safe}.csv", index=False)
meta = {"model": MODEL, "backend": "transformers (NLI classifier)", "items": len(df),
        "seconds": round(time.time() - t0, 1), "dtype": "float32", "quantization": "none (fp32)",
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "decoding": "argmax over NLI classification head (deterministic)",
        "id2label": id2label, "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version()}
json.dump(meta, open(f"results/meta_{safe}.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
