"""Step 3 - Analysis: accuracy/F1 per language, transfer gap with paired bootstrap CI,
exact McNemar test (H1), confusion matrices, tokenizer fertility, and error analysis
of EN-correct -> KK-wrong flips (Objective 4)."""
import glob, json
import numpy as np, pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["entailment", "neutral", "contradiction"]
rng = np.random.default_rng(42)
B = 5000
sample = pd.read_csv("data/pilot_sample.csv")
report, flips_all = {}, []

for f in sorted(glob.glob("results/preds_*.csv")):
    name = f.split("preds_")[1][:-4]
    d = pd.read_csv(f)
    en = d[d.lang == "en"].set_index("xnli_id").sort_index()
    kk = d[d.lang == "kk"].set_index("xnli_id").sort_index()
    assert (en.index == kk.index).all()
    r = {}
    for lang, x in [("en", en), ("kk", kk)]:
        r[f"acc_{lang}"] = accuracy_score(x.gold, x.pred)
        r[f"macroF1_{lang}"] = f1_score(x.gold, x.pred, labels=LABELS, average="macro")
        r[f"perclassF1_{lang}"] = dict(zip(LABELS, f1_score(x.gold, x.pred, labels=LABELS, average=None).round(3).tolist()))
        r[f"pred_dist_{lang}"] = x.pred.value_counts().reindex(LABELS, fill_value=0).to_dict()
        r[f"confusion_{lang}"] = confusion_matrix(x.gold, x.pred, labels=LABELS).tolist()
        r[f"mean_fertility_{lang}"] = float(x.fertility.mean())
        r[f"mean_tokens_{lang}"] = float(x.pair_tokens.mean())
    r["acc_gap"] = r["acc_en"] - r["acc_kk"]
    r["macroF1_gap"] = r["macroF1_en"] - r["macroF1_kk"]
    # paired bootstrap over items
    ce, ck = en.correct.values.astype(float), kk.correct.values.astype(float)
    n = len(ce); idx = rng.integers(0, n, (B, n))
    gaps = ce[idx].mean(1) - ck[idx].mean(1)
    r["acc_gap_CI95"] = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
    # exact McNemar on discordant pairs
    b = int(((ce == 1) & (ck == 0)).sum()); c = int(((ce == 0) & (ck == 1)).sum())
    r["mcnemar_b_en_only"], r["mcnemar_c_kk_only"] = b, c
    r["mcnemar_p_two_sided"] = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    r["mcnemar_p_one_sided_en_better"] = float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c else 1.0
    # agreement between languages
    r["cross_lingual_prediction_agreement"] = float((en.pred.values == kk.pred.values).mean())
    # Error analysis: among EN-correct items, do KK failures have longer KK words / higher fertility?
    m = en.correct.values
    sub = kk[m].join(sample.set_index("xnli_id")[["word_len_kk"]])
    fl, ok = sub[~sub.correct], sub[sub.correct]
    r["flip_rate_given_en_correct"] = float(len(fl) / max(len(sub), 1))
    if len(fl) and len(ok):
        r["flip_vs_kept_mean_word_len_kk"] = [float(fl.word_len_kk.mean()), float(ok.word_len_kk.mean())]
        r["flip_vs_kept_fertility_kk"] = [float(fl.fertility.mean()), float(ok.fertility.mean())]
        r["mannwhitney_p_word_len"] = float(mannwhitneyu(fl.word_len_kk, ok.word_len_kk).pvalue)
        r["mannwhitney_p_fertility"] = float(mannwhitneyu(fl.fertility, ok.fertility).pvalue)
    fl = fl.reset_index().assign(model=name)
    flips_all.append(fl[["model", "xnli_id", "gold", "pred", "fertility", "word_len_kk"]])
    report[name] = r

json.dump(report, open("results/analysis.json", "w"), indent=2)
pd.concat(flips_all).to_csv("results/flips_en_correct_kk_wrong.csv", index=False)

# Summary table
rows = []
for k, r in report.items():
    rows.append([k, r["acc_en"], r["acc_kk"], r["acc_gap"], f"[{r['acc_gap_CI95'][0]:.3f}, {r['acc_gap_CI95'][1]:.3f}]",
                 r["macroF1_en"], r["macroF1_kk"], r["mcnemar_b_en_only"], r["mcnemar_c_kk_only"],
                 r["mcnemar_p_two_sided"], r["mean_fertility_en"], r["mean_fertility_kk"]])
tab = pd.DataFrame(rows, columns=["model", "acc_en", "acc_kk", "gap", "gap_CI95", "F1_en", "F1_kk",
                                  "b_EN_only", "c_KK_only", "McNemar_p", "fert_en", "fert_kk"])
tab.to_csv("results/summary_table.csv", index=False)
print(tab.round(3).to_string(index=False))

# Figure 1: accuracy by language
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(len(tab)); w = 0.36
ax.bar(x - w/2, tab.acc_en, w, label="English", color="#3b6ea5")
ax.bar(x + w/2, tab.acc_kk, w, label="Kazakh", color="#d08c3a")
ax.axhline(1/3, ls="--", c="gray", lw=1); ax.text(-0.62, 1/3+0.012, "chance", ha="left", fontsize=7, color="gray")
for i, r in tab.iterrows():
    ax.text(i - w/2, r.acc_en + 0.01, f"{r.acc_en:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, r.acc_kk + 0.01, f"{r.acc_kk:.2f}", ha="center", fontsize=8)
ax.set_xlim(-0.65, len(tab)-0.35); ax.set_xticks(x); ax.set_xticklabels(tab.model, fontsize=8); ax.set_ylabel("Accuracy (n = 300 per language)")
ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Zero-shot NLI accuracy: English vs Kazakh (pilot)", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig1_accuracy.png", dpi=200)

# Figure 2: Kazakh confusion matrices
fig, axes = plt.subplots(1, len(report), figsize=(3.3 * len(report), 3.1))
for ax, (k, r) in zip(np.atleast_1d(axes), report.items()):
    cm = np.array(r["confusion_kk"]); ax.imshow(cm, cmap="Oranges")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["ent", "neu", "con"]); ax.set_yticklabels(["ent", "neu", "con"])
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(k + " (KK)", fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig2_confusion_kk.png", dpi=200)
print(json.dumps(report, indent=1))
