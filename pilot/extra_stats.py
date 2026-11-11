"""Step 4 - Extra pilot statistics used in the report: KK accuracy vs chance (binomial),
Holm-corrected McNemar p-values, above-chance retention, gap difference between Qwen scales,
and example flip items. Writes results/extra_stats.json (numbers only)."""
import json
import numpy as np, pandas as pd
from scipy.stats import binomtest

A = json.load(open("results/analysis.json"))
s = pd.read_csv("data/pilot_sample.csv").set_index("xnli_id")
out = {}
# Holm correction over the one-sided McNemar p-values
names = list(A)
ps = np.array([A[m]["mcnemar_p_one_sided_en_better"] for m in names])
order = np.argsort(ps); k = len(ps); adj = np.empty(k); running = 0
for rank, i in enumerate(order):
    running = max(running, min(1, (k - rank) * ps[i])); adj[i] = running
for m, a in zip(names, adj):
    r = A[m]
    n = 300
    kk_correct = round(r["acc_kk"] * n); en_correct = round(r["acc_en"] * n)
    out[m] = {
        "mcnemar_p_holm_one_sided": float(a),
        "kk_vs_chance_p": float(binomtest(kk_correct, n, 1/3, alternative="greater").pvalue),
        "en_vs_chance_p": float(binomtest(en_correct, n, 1/3, alternative="greater").pvalue),
        "above_chance_retention": (r["acc_kk"] - 1/3) / (r["acc_en"] - 1/3),
    }
# Gap difference Qwen 0.5B vs 1.5B (paired bootstrap over items)
rng = np.random.default_rng(42)
def correct(m):
    d = pd.read_csv(f"results/preds_{m}.csv")
    en = d[d.lang == "en"].set_index("xnli_id").sort_index().correct.values.astype(float)
    kk = d[d.lang == "kk"].set_index("xnli_id").sort_index().correct.values.astype(float)
    return en, kk
e5, k5 = correct("Qwen2.5-0.5B-Instruct"); e15, k15 = correct("Qwen2.5-1.5B-Instruct")
idx = rng.integers(0, 300, (5000, 300))
diff = (e5[idx].mean(1) - k5[idx].mean(1)) - (e15[idx].mean(1) - k15[idx].mean(1))
out["gap_diff_qwen05_minus_qwen15"] = {"point": float((e5.mean() - k5.mean()) - (e15.mean() - k15.mean())),
                                        "CI95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
# Example flips for Qwen2.5-1.5B (EN correct, KK wrong), first 3 by sample order
d = pd.read_csv("results/preds_Qwen2.5-1.5B-Instruct.csv")
en = d[d.lang == "en"].set_index("xnli_id"); kk = d[d.lang == "kk"].set_index("xnli_id")
flip_ids = [i for i in s.index if en.loc[i, "correct"] and not kk.loc[i, "correct"]][:3]
out["examples"] = [{"xnli_id": int(i), "gold": s.loc[i, "label"], "pred_kk": kk.loc[i, "pred"],
                    "premise_en": s.loc[i, "premise_en"], "hypothesis_en": s.loc[i, "hypothesis_en"],
                    "premise_kk": s.loc[i, "premise_kk"], "hypothesis_kk": s.loc[i, "hypothesis_kk"]} for i in flip_ids]
json.dump(out, open("results/extra_stats.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
