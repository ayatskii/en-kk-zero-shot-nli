"""Step 4 - Extra statistics used in the report: KK accuracy vs chance (binomial),
Holm-corrected McNemar p-values, above-chance retention, paired-bootstrap gap
differences for the planned model contrasts (Table 2, Holm-corrected), the O4 item-level
logistic regression, the O5 language x scale model, and example flip items.
Writes results/extra_stats.json (numbers only). Excluded models (analyze.py EXCLUDED) are
left out of every test; contrasts whose prediction files are missing are skipped."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
import statsmodels.api as sm
import statsmodels.formula.api as smf

A = json.load(open("results/analysis.json"))
s = pd.read_csv("data/eval_set.csv").set_index("xnli_id")
out = {}

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); k = len(ps); adj = np.empty(k); running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (k - rank) * ps[i])); adj[i] = running
    return adj

# Holm correction over the one-sided McNemar p-values of the included models
names = [m for m in A if not A[m]["excluded"]]
adj = holm([A[m]["mcnemar_p_one_sided_en_better"] for m in names])
for m, a in zip(names, adj):
    r = A[m]
    n = r["n_items"]
    kk_correct = round(r["acc_kk"] * n); en_correct = round(r["acc_en"] * n)
    out[m] = {
        "mcnemar_p_holm_one_sided": float(a),
        "kk_vs_chance_p": float(binomtest(kk_correct, n, 1/3, alternative="greater").pvalue),
        "en_vs_chance_p": float(binomtest(en_correct, n, 1/3, alternative="greater").pvalue),
        "above_chance_retention": (r["acc_kk"] - 1/3) / (r["acc_en"] - 1/3) if r["acc_en"] > 1/3 else None,
    }

# Gap differences (gap_A - gap_B) with a paired bootstrap over the items both models share
rng = np.random.default_rng(42)
def correct(m):
    d = pd.read_csv(f"results/preds_{m}.csv")
    en = d[d.lang == "en"].set_index("xnli_id").correct.astype(float)
    kk = d[d.lang == "kk"].set_index("xnli_id").correct.astype(float)
    return en, kk
CONTRASTS = [  # (factor, A, B): positive point estimate = A has the larger EN-KK gap
    ("scale", "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
    ("scale", "Qwen3.5-0.8B", "Qwen3.5-2B"),
    ("scale", "Qwen3.5-2B", "Qwen3.5-4B"),
    ("scale", "Qwen3.5-4B", "Qwen3.5-9B"),
    ("scale", "Qwen3.5-0.8B", "Qwen3.5-9B"),
    ("scale", "Qwen3.5-9B", "Qwen3.5-35B-A3B-Q4_K_M"),
    ("family", "Qwen3.5-4B", "gemma-4-E4B-it"),
    ("family", "Qwen3.5-4B", "gpt-oss-20b-MXFP4"),
    ("total vs active", "gemma-4-E4B-it", "gemma-4-26B-A4B-it-Q4_K_M"),
    ("total vs active", "Qwen3.5-4B", "Qwen3.5-35B-A3B-Q4_K_M"),
    ("generation / tokenizer", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", "Qwen3.5-35B-A3B-Q4_K_M"),
    ("generation / tokenizer", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", "Qwen3.5-9B"),
    ("Kazakh adaptation", "Llama-3.1-8B-Instruct", "LLama-3.1-KazLLM-1.0-8B"),
    ("Kazakh adaptation", "Llama-3.1-8B-Instruct", "Llama-3.1-Sherkala-8B-Chat"),
    ("Kazakh adaptation", "Llama-3.1-Sherkala-8B-Chat", "LLama-3.1-KazLLM-1.0-8B"),
    ("multilingual coverage", "SmolLM2-1.7B-Instruct", "Qwen2.5-1.5B-Instruct"),
    ("multilingual coverage", "SmolLM2-1.7B-Instruct", "Qwen3.5-2B"),
    ("prompting vs supervised", "Qwen3.5-9B", "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
]
gd = {}
for factor, a, b in CONTRASTS:
    if not (os.path.exists(f"results/preds_{a}.csv") and os.path.exists(f"results/preds_{b}.csv")):
        continue
    (ea, ka), (eb, kb) = correct(a), correct(b)
    ids = ea.index.intersection(eb.index)
    ea, ka, eb, kb = (x.loc[ids].values for x in (ea, ka, eb, kb))
    idx = rng.integers(0, len(ids), (5000, len(ids)))
    diff = (ea[idx].mean(1) - ka[idx].mean(1)) - (eb[idx].mean(1) - kb[idx].mean(1))
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())     # two-sided bootstrap p
    gd[f"{a} minus {b}"] = {
        "factor": factor, "n_items": len(ids), "point": float((ea.mean() - ka.mean()) - (eb.mean() - kb.mean())),
        "CI95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "p_bootstrap": float(max(p, 1 / 5000))}
for k, a in zip(gd, holm([v["p_bootstrap"] for v in gd.values()])):
    gd[k]["p_holm"] = float(a)
out["gap_differences"] = gd

# Long table of LLM predictions with model descriptors (NLI classifiers and excluded runs left out)
PARAMS = {  # model: (family, total B, active B, Kazakh-adapted); from the model cards
    "SmolLM2-1.7B-Instruct": ("SmolLM2", 1.7, 1.7, 0),
    "Qwen2.5-0.5B-Instruct": ("Qwen", 0.49, 0.49, 0),
    "Qwen2.5-1.5B-Instruct": ("Qwen", 1.54, 1.54, 0),
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": ("Qwen", 30.5, 3.3, 0),
    "Qwen3.5-0.8B": ("Qwen", 0.8, 0.8, 0),
    "Qwen3.5-2B": ("Qwen", 2.0, 2.0, 0),
    "Qwen3.5-4B": ("Qwen", 4.0, 4.0, 0),
    "Qwen3.5-9B": ("Qwen", 9.0, 9.0, 0),
    "Qwen3.5-35B-A3B-Q4_K_M": ("Qwen", 35.0, 3.0, 0),
    "gemma-4-E4B-it": ("Gemma", 8.0, 4.5, 0),
    "gemma-4-26B-A4B-it-Q4_K_M": ("Gemma", 26.0, 3.8, 0),
    "gpt-oss-20b-MXFP4": ("gpt-oss", 21.0, 3.6, 0),
    "Llama-3.1-8B-Instruct": ("Llama", 8.0, 8.0, 0),
    "LLama-3.1-KazLLM-1.0-8B": ("Llama", 8.0, 8.0, 1),
    "Llama-3.1-Sherkala-8B-Chat": ("Llama", 8.0, 8.0, 1),
}
rows = []
for m in names:
    if m not in PARAMS or A[m]["n_items"] != len(s):
        continue
    fam, tot, act, kz = PARAMS[m]
    d = pd.read_csv(f"results/preds_{m}.csv").join(s[["word_len_kk"]], on="xnli_id")
    d = d.assign(model=m, family=fam, log2_total=np.log2(tot), log2_active=np.log2(act), kz_adapted=kz,
                 kk=(d.lang == "kk").astype(int), correct=d.correct.astype(int))
    rows.append(d)
L = pd.concat(rows, ignore_index=True)
out["regression_models"] = sorted(L.model.unique())

def coefs(res):
    ci = res.conf_int()
    return {k: {"b": float(res.params[k]), "OR": float(np.exp(res.params[k])), "se": float(res.bse[k]),
                "CI95_OR": [float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))], "p": float(res.pvalues[k])}
            for k in res.params.index}

# O4: among items a model solves in English, is the Kazakh answer less often right when the
# Kazakh pair is fragmented more (fertility, z-scored within model) or has longer words?
kkL = L[L.lang == "kk"].copy()
en_ok = L[L.lang == "en"].set_index(["model", "xnli_id"]).correct
kkL = kkL[en_ok.loc[list(zip(kkL.model, kkL.xnli_id))].values == 1].copy()
kkL["z_fert"] = kkL.groupby("model").fertility.transform(lambda x: (x - x.mean()) / x.std())
kkL["z_wlen"] = (kkL.word_len_kk - kkL.word_len_kk.mean()) / kkL.word_len_kk.std()
res = smf.glm("correct ~ z_fert + z_wlen + C(gold) + C(model)", kkL, family=sm.families.Binomial()) \
         .fit(cov_type="cluster", cov_kwds={"groups": kkL.xnli_id})
c = coefs(res)
out["O4_logit_kk_correct_given_en_correct"] = {
    "n_rows": int(len(kkL)), "n_items": int(kkL.xnli_id.nunique()), "se": "cluster-robust by item",
    "coefficients": {k: v for k, v in c.items() if not k.startswith("C(model)")}}

# O5: language x scale, logistic GEE with items as clusters (exchangeable working correlation).
# Fitted with total and with active parameters (they differ for MoE models), then with Kazakh adaptation.
gee = {}
for label, f in [("total", "correct ~ kk * log2_total + C(family)"),
                 ("active", "correct ~ kk * log2_active + C(family)"),
                 ("total_plus_adaptation", "correct ~ kk * log2_total + kk * kz_adapted + C(family)")]:
    r = smf.gee(f, "xnli_id", L, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    gee[label] = {"formula": f, "n_rows": int(len(L)), "n_clusters": int(L.xnli_id.nunique()),
                  "coefficients": coefs(r)}
out["O5_gee_language_x_scale"] = gee

# Example flips (EN correct, KK wrong), first 3 by item order, for the best included LLM in Kazakh
best = max((m for m in names if "NLI classifier" not in A[m]["backend"]), key=lambda m: A[m]["acc_kk"])
d = pd.read_csv(f"results/preds_{best}.csv")
en = d[d.lang == "en"].set_index("xnli_id"); kk = d[d.lang == "kk"].set_index("xnli_id")
flip_ids = [i for i in en.index if en.loc[i, "correct"] and not kk.loc[i, "correct"]][:3]
out["examples_model"] = best
out["examples"] = [{"xnli_id": int(i), "gold": s.loc[i, "label"], "pred_kk": kk.loc[i, "pred"],
                    "premise_en": s.loc[i, "premise_en"], "hypothesis_en": s.loc[i, "hypothesis_en"],
                    "premise_kk": s.loc[i, "premise_kk"], "hypothesis_kk": s.loc[i, "hypothesis_kk"]} for i in flip_ids]
json.dump(out, open("results/extra_stats.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))

# Figure 4: above-chance retention per included full-set model (same order as analyze.py figures)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_names import short
ret = [(m, out[m]["above_chance_retention"], out[m]["kk_vs_chance_p"] < .05) for m in names
       if A[m]["n_items"] == len(s) and out[m]["above_chance_retention"] is not None]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (m, r, ok) in enumerate(ret):
    ax.bar(i, r, 0.65, color="#3b6ea5" if ok else "white", edgecolor="#3b6ea5")
    ax.text(i, max(r, 0) + 0.015, f"{r:.2f}", ha="center", fontsize=7)
ax.axhline(0, c="gray", lw=0.8); ax.axhline(1, c="gray", lw=0.8, ls=":")
ax.set_xticks(range(len(ret))); ax.set_xticklabels([short(m) for m, _, _ in ret], fontsize=7, rotation=35, ha="right")
ax.set_ylabel("(acc KK − 1/3) / (acc EN − 1/3)"); ax.set_ylim(-0.15, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Above-chance retention in Kazakh (hollow: KK not above chance)", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig4_retention.png", dpi=200)
