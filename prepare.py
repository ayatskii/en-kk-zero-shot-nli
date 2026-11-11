"""Step 1 - Data preparation for the EN->KK zero-shot NLI study.

Input : Kardes-NLU Kazakh XNLI test file (parallel English + Kazakh columns)
Output: data/eval_set.csv  (all items that pass the quality flags, cleaned)
        results/data_summary.json
The full set is used (no sampling); the source labels are already balanced.
Set PER_CLASS to an int to draw a stratified subsample instead (pilot used 100).
"""
import json, re, unicodedata
import pandas as pd

SEED = 42
PER_CLASS = None         # None = full eval set; e.g. 100 -> 300-item stratified pilot
SRC = "data/xnli.test.kk.csv"

def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def cyr_ratio(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    return sum("Ѐ" <= c <= "ӿ" for c in letters) / len(letters)

df = pd.read_csv(SRC, encoding="utf-8-sig").rename(columns={
    "Unnamed: 0": "xnli_id", "sentence1": "premise_en", "sentence2": "hypothesis_en",
    "s1_translation": "premise_kk", "s2_translation": "hypothesis_kk", "gold_label": "label"})
for c in ["premise_en", "hypothesis_en", "premise_kk", "hypothesis_kk"]:
    df[c] = df[c].map(clean)

# Quality flags on the Kazakh side (translation checks)
df["kk_cyr_ratio"] = (df.premise_kk + " " + df.hypothesis_kk).map(cyr_ratio)
df["len_ratio"] = (df.premise_kk.str.len() + df.hypothesis_kk.str.len()) / \
                  (df.premise_en.str.len() + df.hypothesis_en.str.len())
df["flag_low_cyrillic"] = df.kk_cyr_ratio < 0.80
df["flag_len_outlier"] = (df.len_ratio < 0.5) | (df.len_ratio > 2.0)
df["flag_untranslated"] = (df.premise_kk == df.premise_en) | (df.hypothesis_kk == df.hypothesis_en)
df["flag_any"] = df.flag_low_cyrillic | df.flag_len_outlier | df.flag_untranslated

full = len(df)
clean_df = df[~df.flag_any]
if PER_CLASS is None:
    sample = clean_df.reset_index(drop=True).copy()
else:
    sample = (pd.concat([clean_df[clean_df.label == l].sample(PER_CLASS, random_state=SEED)
                         for l in ["entailment", "neutral", "contradiction"]])
              .sample(frac=1, random_state=SEED).reset_index(drop=True))
sample["word_len_kk"] = (sample.premise_kk + " " + sample.hypothesis_kk).str.split().map(
    lambda w: sum(map(len, w)) / len(w))
sample.to_csv("data/eval_set.csv", index=False)

summary = {
    "source_file": SRC, "items_in_source": full,
    "label_distribution_source": df.label.value_counts().to_dict(),
    "flagged_low_cyrillic": int(df.flag_low_cyrillic.sum()),
    "flagged_length_outlier": int(df.flag_len_outlier.sum()),
    "flagged_untranslated": int(df.flag_untranslated.sum()),
    "excluded_total": int(df.flag_any.sum()),
    "eligible_after_exclusion": len(clean_df),
    "eval_set_size": len(sample),
    "eval_label_distribution": sample.label.value_counts().to_dict(),
    "mean_len_ratio_kk_over_en_chars": round(float(df.len_ratio.mean()), 3),
    "mean_words_en": round(float((df.premise_en + " " + df.hypothesis_en).str.split().map(len).mean()), 2),
    "mean_words_kk": round(float((df.premise_kk + " " + df.hypothesis_kk).str.split().map(len).mean()), 2),
    "seed": SEED,
}
json.dump(summary, open("results/data_summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(df[df.flag_any][["xnli_id", "premise_kk", "hypothesis_kk", "kk_cyr_ratio", "len_ratio"]].head(8).to_string())
