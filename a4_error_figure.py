"""Assignment 4 - summary of the manual error coding (results/a4_error_coding.csv, codes A-E
assigned by hand to the 40 sampled flips in results/a4_error_sample.csv).
Writes results/a4_error_summary.json and figures/a4_fig4_error_types.png."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"A": "A. Translation changes the meaning", "B": "B. Literal / inconsistent wording\n(idiom, term, name)",
         "C": "C. Faithful, needs inference:\nmodel retreats to neutral", "D": "D. Faithful, negation /\ncounterfactual",
         "E": "E. Faithful, other wrong label"}
c = pd.read_csv("results/a4_error_coding.csv"); s = pd.read_csv("results/a4_error_sample.csv")
m = s.merge(c, on="xnli_id"); assert len(m) == 40 and set(m.code) <= set(NAMES)
n = m.code.value_counts().reindex(list(NAMES)).fillna(0).astype(int)
out = {"n": int(len(m)), "counts": n.to_dict(), "share": (n / len(m)).round(4).to_dict(),
       "translation_related_A_B": int(n["A"] + n["B"]), "faithful_C_D_E": int(n["C"] + n["D"] + n["E"]),
       "kk_prediction_by_code": pd.crosstab(m.code, m.pred_kk).to_dict("index"),
       "gold_by_code": pd.crosstab(m.code, m.label).to_dict("index"),
       "gold_to_pred": pd.crosstab(m.label, m.pred_kk).to_dict("index"),
       "mean_other_llms_flipping_by_code": m.groupby("code").n_other_llms_flipping.mean().round(2).to_dict()}
json.dump(out, open("results/a4_error_summary.json", "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=200)
cols = ["#d9730d", "#d9730d", "#2f6fb5", "#2f6fb5", "#2f6fb5"]
y = range(len(n))[::-1]
ax.barh(list(y), n.values, color=cols, height=.62)
for yi, v in zip(y, n.values):
    ax.text(v + .25, yi, f"{v}  ({v / len(m) * 100:.1f}%)", va="center", fontsize=9, color="#1f2328")
ax.set_yticks(list(y)); ax.set_yticklabels([NAMES[k] for k in n.index]); ax.set_xlim(0, 16)
ax.set_xlabel("Number of flips (n = 40 sampled from 331)"); ax.spines["left"].set_color("#c9ced6"); ax.spines["bottom"].set_color("#c9ced6")
ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#d9730d", label=f"translation-related ({n['A'] + n['B']})"),
                   Patch(color="#2f6fb5", label=f"translation faithful ({n['C'] + n['D'] + n['E']})")],
          frameon=False, loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig("figures/a4_fig4_error_types.png"); plt.close(fig)
print(out["counts"])
