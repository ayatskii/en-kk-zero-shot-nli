"""Step 3 - Analysis: accuracy/F1 per language, transfer gap with paired bootstrap CI,
exact McNemar test (H1), confusion matrices, tokenizer fertility, and error analysis
of EN-correct -> KK-wrong flips (Objective 4). Reads every results/preds_*.csv
(batch 1 transformers, NLI classifiers, batch 2/3 llama.cpp) and the matching meta file."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_names import short

LABELS = ["entailment", "neutral", "contradiction"]
rng = np.random.default_rng(42)
B = 5000
sample = pd.read_csv("data/eval_set.csv")
report, flips_all = {}, []
# kept in the tables but left out of the figures and comparisons
EXCLUDED = {"xlm-roberta-large-xnli": "fine-tuned on the XNLI test set (model card)"}
# figure order: family, then size; runs not listed go last
ORDER = ["SmolLM2-1.7B-Instruct", "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct",
         "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B",
         "Qwen3.5-9B", "Qwen3.5-35B-A3B-Q4_K_M", "gemma-4-E4B-it", "gemma-4-26B-A4B-it-Q4_K_M",
         "gpt-oss-20b-MXFP4", "Llama-3.1-8B-Instruct", "LLama-3.1-KazLLM-1.0-8B",
         "Llama-3.1-Sherkala-8B-Chat", "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"]
files = sorted(glob.glob("results/preds_*.csv"),
               key=lambda f: (ORDER.index(n) if (n := f.split("preds_")[1][:-4]) in ORDER else len(ORDER), n))
for f in files:
    name = f.split("preds_")[1][:-4]
    d = pd.read_csv(f)
    en = d[d.lang == "en"].set_index("xnli_id").sort_index()
    kk = d[d.lang == "kk"].set_index("xnli_id").sort_index()
    assert (en.index == kk.index).all()
    mf = f"results/meta_{name}.json"
    meta = json.load(open(mf)) if os.path.exists(mf) else {}
    r = {"n_items": len(en), "excluded": EXCLUDED.get(name, ""), "backend": meta.get("backend", "?"), "quantization": meta.get("quantization", "?")}
    for lang, x in [("en", en), ("kk", kk)]:
        r[f"acc_{lang}"] = accuracy_score(x.gold, x.pred)
        r[f"macroF1_{lang}"] = f1_score(x.gold, x.pred, labels=LABELS, average="macro")
        r[f"perclassF1_{lang}"] = dict(zip(LABELS, f1_score(x.gold, x.pred, labels=LABELS, average=None).round(3).tolist()))
        r[f"pred_dist_{lang}"] = x.pred.value_counts().reindex(LABELS, fill_value=0).to_dict()
        r[f"confusion_{lang}"] = confusion_matrix(x.gold, x.pred, labels=LABELS).tolist()
        r[f"mean_fertility_{lang}"] = float(x.fertility.mean())
        r[f"mean_tokens_{lang}"] = float(x.pair_tokens.mean())
        r[f"invalid_{lang}"] = int((x.pred == "invalid").sum())   # only reason-mode runs
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
    rows.append([k, r["excluded"], r["backend"], r["quantization"], r["n_items"], r["acc_en"], r["acc_kk"], r["acc_gap"], f"[{r['acc_gap_CI95'][0]:.3f}, {r['acc_gap_CI95'][1]:.3f}]",
                 r["macroF1_en"], r["macroF1_kk"], r["mcnemar_b_en_only"], r["mcnemar_c_kk_only"],
                 r["mcnemar_p_two_sided"], r["mean_fertility_en"], r["mean_fertility_kk"]])
tab = pd.DataFrame(rows, columns=["model", "excluded", "backend", "quant", "n", "acc_en", "acc_kk", "gap", "gap_CI95", "F1_en", "F1_kk",
                                  "b_EN_only", "c_KK_only", "McNemar_p", "fert_en", "fert_kk"])
tab.to_csv("results/summary_table.csv", index=False)
print(tab.round(3).to_string(index=False))

# Figures: full-set runs only (reason-mode subsets and excluded models are left out)
fig_tab = tab[(tab.excluded == "") & (tab.n == tab.n.max())].reset_index(drop=True)
fig_rep = {k: report[k] for k in fig_tab.model}
labels = [short(m) for m in fig_tab.model]

# Figure 1: accuracy by language
fig, ax = plt.subplots(figsize=(max(6.4, 0.6 * len(fig_tab) + 1.5), 4.2))
x = np.arange(len(fig_tab)); w = 0.38
ax.bar(x - w/2, fig_tab.acc_en, w, label="English", color="#3b6ea5")
ax.bar(x + w/2, fig_tab.acc_kk, w, label="Kazakh", color="#d08c3a")
ax.axhline(1/3, ls="--", c="gray", lw=1); ax.text(-0.62, 1/3+0.012, "chance", ha="left", fontsize=7, color="gray")
for i, r in fig_tab.iterrows():
    ax.text(i - w/2, r.acc_en + 0.01, f"{r.acc_en:.2f}", ha="center", fontsize=6)
    ax.text(i + w/2, r.acc_kk + 0.01, f"{r.acc_kk:.2f}", ha="center", fontsize=6)
ax.set_xlim(-0.65, len(fig_tab)-0.35); ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
ax.set_ylabel(f"Accuracy (n = {fig_tab.n.max()} per language)")
ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("NLI accuracy: English vs Kazakh", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig1_accuracy.png", dpi=200)

# Figure 2: Kazakh confusion matrices
ncol = min(4, len(fig_rep)); nrow = -(-len(fig_rep) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow), squeeze=False)
for ax in axes.flat[len(fig_rep):]:
    ax.axis("off")
for ax, (k, r) in zip(axes.flat, fig_rep.items()):
    cm = np.array(r["confusion_kk"]); ax.imshow(cm, cmap="Oranges")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.6 * cm.max() else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["ent", "neu", "con"]); ax.set_yticklabels(["ent", "neu", "con"])
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(short(k) + " (KK)", fontsize=9)
fig.tight_layout(); fig.savefig("figures/fig2_confusion_kk.png", dpi=200)

# Figure 3: gap vs Kazakh token premium (tokens per KK pair / tokens per EN pair), LLMs only
llm = fig_tab[~fig_tab.backend.str.contains("NLI classifier")]
prem = [report[m]["mean_tokens_kk"] / report[m]["mean_tokens_en"] for m in llm.model]
# hollow marker = Kazakh accuracy not above chance (binomial p >= .05): the gap there is a floor effect
above = [binomtest(round(a * n), n, 1/3, alternative="greater").pvalue < .05 for a, n in zip(llm.acc_kk, llm.n)]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for px, g, ok in zip(prem, llm.gap, above):
    ax.scatter(px, g, s=36, zorder=3, color="#3b6ea5" if ok else "white", edgecolors="#3b6ea5")
placed = []
for px, g, m in sorted(zip(prem, llm.gap, llm.model), key=lambda t: (t[0], -t[1])):
    clash = any(abs(px - qx) < 0.15 and abs(g - qg) < 0.012 for qx, qg in placed)
    ax.annotate(short(m), (px, g), textcoords="offset points", xytext=(5, -9 if clash else 3), fontsize=7)
    placed.append((px, g))
ax.scatter([], [], s=36, color="white", edgecolors="#3b6ea5", label="Kazakh accuracy not above chance")
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.axhline(0, c="gray", lw=0.8)
ax.set_xlabel("Kazakh token premium (tokens per KK pair / per EN pair)")
ax.set_ylabel("Accuracy gap EN − KK"); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Transfer gap vs tokenizer cost of Kazakh", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig3_gap_vs_token_premium.png", dpi=200)
print(json.dumps(report, indent=1))
