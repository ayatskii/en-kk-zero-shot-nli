"""Assignment 4 - additional statistics computed from the frozen prediction files
(results/preds_<model>.csv). No model is run here; inference is deterministic and the
prediction files are the ones produced by run_batch1.sh / run_batch2.sh.

Writes results/a4_stats.json, results/a4_main_table.csv,
       results/a4_error_sample.csv (40 random flips of the best zero-shot LLM, to be coded by hand)
       figures/a4_fig1..fig3 (fig4 is drawn by a4_error_figure.py after the manual coding).
Everything marked "post hoc" in the report comes from this file.
"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_names import short

SEED, B = 42, 5000
LABELS = ["entailment", "neutral", "contradiction"]
A = json.load(open("results/analysis.json"))
X = json.load(open("results/extra_stats.json"))
items = pd.read_csv("data/eval_set.csv").set_index("xnli_id")
MODELS = [m for m in A if not A[m]["excluded"]]                      # 14 analysed models
NLI = "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"                 # strong baseline
LLMS = [m for m in MODELS if m != NLI]                               # 13 zero-shot LLMs
ABOVE = [m for m in LLMS if X[m]["kk_vs_chance_p"] < .05]            # 11 LLMs above chance in KK
BEST = max(LLMS, key=lambda m: A[m]["acc_kk"])                       # best zero-shot LLM in Kazakh
ids = items.index.values
N = len(ids)
out = {"n_items": int(N), "models": MODELS, "above_chance_llms": ABOVE, "best_llm_kk": BEST}

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); k = len(ps); adj = np.empty(k); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (k - rank) * ps[i])); adj[i] = run
    return adj

P = {}
for m in MODELS:
    d = pd.read_csv(f"results/preds_{m}.csv")
    P[m] = {l: d[d.lang == l].set_index("xnli_id").loc[ids] for l in ("en", "kk")}
gold = P[MODELS[0]]["en"].gold.values
C = {m: {l: (P[m][l].pred.values == gold) for l in ("en", "kk")} for m in MODELS}

# ---- 1. baselines -----------------------------------------------------------------
maj_label = pd.Series(gold).value_counts().idxmax()
maj_acc = float((gold == maj_label).mean())
out["baselines"] = {"chance": 1 / 3, "majority_label": maj_label, "majority_accuracy": maj_acc}

# ---- 2. main table: accuracy with bootstrap CI, effect sizes ------------------------
rng = np.random.default_rng(SEED)
idx = rng.integers(0, N, size=(B, N))
rows = []
for m in MODELS:
    en, kk = C[m]["en"].astype(float), C[m]["kk"].astype(float)
    be, bk = en[idx].mean(1), kk[idx].mean(1)
    b = int(((en == 1) & (kk == 0)).sum()); c = int(((en == 0) & (kk == 1)).sum())
    ci = binomtest(b, b + c).proportion_ci(.95, method="exact")       # share of discordant pairs lost in KK
    h = 2 * np.arcsin(np.sqrt(en.mean())) - 2 * np.arcsin(np.sqrt(kk.mean()))
    rows.append({
        "model": m, "short": short(m), "quant": A[m]["quantization"],
        "acc_en": en.mean(), "acc_en_lo": np.percentile(be, 2.5), "acc_en_hi": np.percentile(be, 97.5),
        "acc_kk": kk.mean(), "acc_kk_lo": np.percentile(bk, 2.5), "acc_kk_hi": np.percentile(bk, 97.5),
        "gap": en.mean() - kk.mean(),
        "gap_lo": np.percentile(be - bk, 2.5), "gap_hi": np.percentile(be - bk, 97.5),
        "f1_en": A[m]["macroF1_en"], "f1_kk": A[m]["macroF1_kk"],
        "b_en_only": b, "c_kk_only": c, "discordant_odds_ratio": b / c,
        "discordant_share_en_only": b / (b + c), "disc_share_lo": ci.low, "disc_share_hi": ci.high,
        "cohens_h": h,
        "mcnemar_p_holm": X[m]["mcnemar_p_holm_one_sided"],
        "kk_vs_chance_p": X[m]["kk_vs_chance_p"],
        "kk_vs_majority_p": binomtest(int(kk.sum()), N, maj_acc, alternative="greater").pvalue,
        "retention": X[m]["above_chance_retention"],
        "seconds": json.load(open(f"results/meta_{m}.json"))["seconds"],
    })
T = pd.DataFrame(rows)
T["sec_per_1000_pred"] = T.seconds / (2 * N) * 1000
T.to_csv("results/a4_main_table.csv", index=False)
out["main_table"] = T.set_index("model").to_dict("index")

# ---- 3. omnibus test across models (Cochran's Q = Friedman test for binary outcomes) ----
def cochran_q(M):                       # M: items x k matrix of 0/1
    M = np.asarray(M, float); k = M.shape[1]
    col, row = M.sum(0), M.sum(1); Ntot = M.sum()
    q = (k - 1) * (k * (col ** 2).sum() - Ntot ** 2) / (k * Ntot - (row ** 2).sum())
    return float(q), k - 1, float(chi2.sf(q, k - 1))
out["cochran_q"] = {}
for lang in ("en", "kk"):
    q, df, p = cochran_q(np.column_stack([C[m][lang] for m in MODELS]))
    out["cochran_q"][lang] = {"Q": q, "df": df, "p": p, "k_models": len(MODELS)}

# ---- 4. every LLM vs the strong baseline on the Kazakh items (exact McNemar, Holm) ----
vs = {}
for m in LLMS:
    a, bb = C[m]["kk"], C[NLI]["kk"]
    llm_only = int((a & ~bb).sum()); nli_only = int((~a & bb).sum())
    vs[m] = {"acc_kk_llm": float(a.mean()), "acc_kk_baseline": float(bb.mean()),
             "diff": float(a.mean() - bb.mean()), "llm_only": llm_only, "baseline_only": nli_only,
             "p_two_sided": float(binomtest(llm_only, llm_only + nli_only, .5).pvalue)}
for m, a in zip(vs, holm([vs[m]["p_two_sided"] for m in vs])):
    vs[m]["p_holm"] = float(a)
out["llm_vs_strong_baseline_kk"] = vs

# ---- 5. subgroup: accuracy by gold label ------------------------------------------------
sub = {}
for m in MODELS:
    sub[m] = {}
    for g in LABELS:
        k = gold == g
        sub[m][g] = {"n": int(k.sum()), "acc_en": float(C[m]["en"][k].mean()), "acc_kk": float(C[m]["kk"][k].mean()),
                     "gap": float(C[m]["en"][k].mean() - C[m]["kk"][k].mean())}
pooled = {}
for g in LABELS:
    k = gold == g
    en = np.mean([C[m]["en"][k].mean() for m in ABOVE]); kk = np.mean([C[m]["kk"][k].mean() for m in ABOVE])
    pooled[g] = {"acc_en": float(en), "acc_kk": float(kk), "gap": float(en - kk)}
out["by_gold_label"] = {"per_model": sub, "mean_over_above_chance_llms": pooled}

# predicted-label shares (label bias) per model and language
out["pred_share"] = {m: {l: {g: float((P[m][l].pred.values == g).mean()) for g in LABELS} for l in ("en", "kk")}
                     for m in MODELS}

# ---- 6. subgroup: item length (tertiles of English words per pair) ----------------------
w = P[BEST]["en"].pair_words.values
t1, t2 = np.quantile(w, [1 / 3, 2 / 3])
bins = {"short": w <= t1, "medium": (w > t1) & (w <= t2), "long": w > t2}
out["by_length"] = {"tertile_cuts_words": [float(t1), float(t2)], "groups": {}}
for name, k in bins.items():
    en = np.mean([C[m]["en"][k].mean() for m in ABOVE]); kk = np.mean([C[m]["kk"][k].mean() for m in ABOVE])
    out["by_length"]["groups"][name] = {"n": int(k.sum()), "acc_en": float(en), "acc_kk": float(kk), "gap": float(en - kk),
                                        "best_llm_gap": float(C[BEST]["en"][k].mean() - C[BEST]["kk"][k].mean())}

# ---- 7. sensitivity: decision rule with the label prior removed (post hoc) ---------------
# For each model and language, subtract the mean log-probability of each label over the
# 1,968 unlabelled inputs before the argmax (batch calibration). No gold label is used.
cal = {}
for m in LLMS:
    r = {}
    for l in ("en", "kk"):
        lp = np.log(np.clip(P[m][l][["p_ent", "p_neu", "p_con"]].values, 1e-12, 1))
        pred = np.array(LABELS)[(lp - lp.mean(0)).argmax(1)]
        r[l] = pred == gold
        r["share_" + l] = {g: float((pred == g).mean()) for g in LABELS}
    en, kk = r["en"].astype(float), r["kk"].astype(float)
    d = en[idx].mean(1) - kk[idx].mean(1)
    b = int((r["en"] & ~r["kk"]).sum()); c = int((~r["en"] & r["kk"]).sum())
    cal[m] = {"acc_en": float(en.mean()), "acc_kk": float(kk.mean()), "gap": float(en.mean() - kk.mean()),
              "gap_lo": float(np.percentile(d, 2.5)), "gap_hi": float(np.percentile(d, 97.5)),
              "b_en_only": b, "c_kk_only": c,
              "mcnemar_p_one_sided": float(binomtest(b, b + c, .5, alternative="greater").pvalue),
              "retention": float((kk.mean() - 1 / 3) / (en.mean() - 1 / 3)),
              "raw_gap": float(C[m]["en"].mean() - C[m]["kk"].mean()),
              "raw_acc_en": float(C[m]["en"].mean()), "raw_acc_kk": float(C[m]["kk"].mean()),
              "pred_share_kk": r["share_kk"], "pred_share_en": r["share_en"]}
for m, a in zip(cal, holm([cal[m]["mcnemar_p_one_sided"] for m in cal])):
    cal[m]["mcnemar_p_holm"] = float(a)
out["prior_corrected_rule"] = cal

# ---- 8. do the same items fail in every model? ------------------------------------------
flip = np.column_stack([C[m]["en"] & ~C[m]["kk"] for m in ABOVE])
nflip = flip.sum(1)
out["flip_overlap"] = {"n_models": len(ABOVE),
                       "items_by_number_of_models_flipping": {int(k): int((nflip == k).sum()) for k in range(len(ABOVE) + 1)},
                       "items_flipped_by_none": int((nflip == 0).sum()),
                       "items_flipped_by_at_least_half": int((nflip >= 6).sum()),
                       "items_flipped_by_8_or_more": int((nflip >= 8).sum()),
                       "total_flips": int(flip.sum())}
# direction of the flips of the 13 LLMs and of the best LLM
def directions(ms):
    cnt = {}
    for m in ms:
        k = C[m]["en"] & ~C[m]["kk"]
        for g, p in zip(gold[k], P[m]["kk"].pred.values[k]):
            cnt[f"{g}->{p}"] = cnt.get(f"{g}->{p}", 0) + 1
    tot = sum(cnt.values())
    return {"total": tot, "counts": cnt, "share": {k: v / tot for k, v in cnt.items()}}
out["flip_directions"] = {"all_llms": directions(LLMS), "best_llm": directions([BEST])}

# ---- 9. error-analysis sample: 40 random flips of the best zero-shot LLM ----------------
k = C[BEST]["en"] & ~C[BEST]["kk"]
flip_ids = ids[k]
samp = np.sort(np.random.default_rng(SEED).choice(flip_ids, size=40, replace=False))
E = items.loc[samp, ["label", "premise_en", "hypothesis_en", "premise_kk", "hypothesis_kk"]].copy()
E.insert(1, "pred_kk", P[BEST]["kk"].set_index(P[BEST]["kk"].index).loc[samp, "pred"].values)
E["p_gold_kk"] = [P[BEST]["kk"].loc[i, {"entailment": "p_ent", "neutral": "p_neu", "contradiction": "p_con"}[items.loc[i, "label"]]] for i in samp]
E["n_other_llms_flipping"] = [int(nflip[list(ids).index(i)]) - 1 for i in samp]
E.to_csv("results/a4_error_sample.csv")
out["error_sample"] = {"model": BEST, "n_flips_total": int(k.sum()), "n_sampled": 40, "seed": SEED,
                       "item_ids": [int(i) for i in samp]}

json.dump(out, open("results/a4_stats.json", "w"), indent=1, default=float)

# ---- figures ------------------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
EN_C, KK_C, INK, MUTED = "#2f6fb5", "#d9730d", "#1f2328", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#c9ced6", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})

# Fig 1: accuracy by language with bootstrap CIs
S = T.sort_values("acc_kk").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
for i, r in S.iterrows():
    ax.plot([r.acc_kk, r.acc_en], [i, i], color="#c9ced6", lw=2, zorder=1)
    ax.errorbar(r.acc_en, i, xerr=[[r.acc_en - r.acc_en_lo], [r.acc_en_hi - r.acc_en]], fmt="o", color=EN_C, ms=6, capsize=2.5, lw=1.2, zorder=3)
    ax.errorbar(r.acc_kk, i, xerr=[[r.acc_kk - r.acc_kk_lo], [r.acc_kk_hi - r.acc_kk]], fmt="s", color=KK_C, ms=6, capsize=2.5, lw=1.2, zorder=3)
    ax.text(1.005, i, f"{r.gap * 100:.1f}", va="center", ha="left", fontsize=8, color=INK, transform=ax.get_yaxis_transform())
ax.text(1.005, len(S) - .2, "gap, pp", va="bottom", ha="left", fontsize=8, color=MUTED, transform=ax.get_yaxis_transform())
ax.axvline(1 / 3, color=MUTED, ls="--", lw=1); ax.text(1 / 3 + .005, -0.85, "chance / majority baseline", fontsize=7.5, color=MUTED, va="center")
ax.set_yticks(range(len(S))); ax.set_yticklabels([("* " if m == NLI else "") + s for m, s in zip(S.model, S.short)])
ax.set_xlim(.25, .95); ax.set_ylim(-1.3, len(S) - .4); ax.set_xlabel("Accuracy on 1,968 paired items (95% bootstrap CI)")
ax.grid(axis="x", color="#eceff3", lw=.8); ax.set_axisbelow(True)
ax.plot([], [], "o", color=EN_C, label="English"); ax.plot([], [], "s", color=KK_C, label="Kazakh")
ax.legend(loc="lower right", frameon=False)
fig.text(.01, .01, "* supervised cross-lingual NLI classifier (strong baseline); all other models are zero-shot LLMs", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, .03, .97, 1)); fig.savefig("figures/a4_fig1_accuracy_ci.png"); plt.close(fig)

# Fig 2: confusion matrices EN vs KK, best LLM and strong baseline
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("b", ["#f3f7fc", "#2f6fb5", "#143a66"])
fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.0), dpi=200)
for r, m in enumerate([BEST, NLI]):
    for c, l in enumerate(("en", "kk")):
        ax = axs[r, c]; M = np.array(A[m][f"confusion_{l}"]); R = M / M.sum(1, keepdims=True)
        ax.imshow(R, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]}\n{R[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if R[i, j] > .5 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["ent.", "neu.", "con."]); ax.set_yticklabels(["ent.", "neu.", "con."])
        ax.set_title(f"{short(m)} - {'English' if l == 'en' else 'Kazakh'} (acc. {A[m]['acc_' + l]:.3f})", fontsize=9)
        ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
        if c == 0: ax.set_ylabel("gold label")
        if r == 1: ax.set_xlabel("predicted label")
fig.tight_layout(); fig.savefig("figures/a4_fig2_confusion_en_kk.png"); plt.close(fig)

# Fig 3: raw vs prior-corrected gap for the 13 LLMs
order = sorted(LLMS, key=lambda m: cal[m]["raw_gap"])
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
for i, m in enumerate(order):
    r = cal[m]; hollow = m not in ABOVE
    ax.plot([r["raw_gap"], r["gap"]], [i, i], color="#c9ced6", lw=2, zorder=1)
    ax.plot(r["raw_gap"], i, "o", ms=7, color=MUTED, mfc="white" if hollow else MUTED, zorder=3)
    ax.errorbar(r["gap"], i, xerr=[[r["gap"] - r["gap_lo"]], [r["gap_hi"] - r["gap"]]], fmt="D", ms=6, color=EN_C, capsize=2.5, lw=1.2, zorder=4)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(order))); ax.set_yticklabels([short(m) for m in order])
ax.set_xlabel("English - Kazakh accuracy gap"); ax.grid(axis="x", color="#eceff3", lw=.8); ax.set_axisbelow(True)
ax.plot([], [], "o", color=MUTED, label="frozen rule (first-token argmax)")
ax.plot([], [], "D", color=EN_C, label="label prior removed (post hoc), 95% CI")
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.text(.01, .01, "Hollow circle: Kazakh accuracy not above chance under the frozen rule.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig("figures/a4_fig3_prior_corrected_gap.png"); plt.close(fig)
print("done", BEST, len(ABOVE))
