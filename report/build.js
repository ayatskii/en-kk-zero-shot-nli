// Assignment 3 report (final version, full study). Numbers are read from the pipeline's result files.
const fs = require("fs");
const { writeDocx, writePdf } = require("./render");

const ROOT = "/home/ayatskii/Downloads/a3_pipeline_code_and_results/a3_pipeline";
const OUT = process.argv[2] || "/home/ayatskii/Downloads/Assignment3_Baurzhanova_Mergenbay_v4";
const A = JSON.parse(fs.readFileSync(`${ROOT}/results/analysis.json`));
const E = JSON.parse(fs.readFileSync(`${ROOT}/results/extra_stats.json`));
const DS = JSON.parse(fs.readFileSync(`${ROOT}/results/data_summary.json`));
const PA = JSON.parse(fs.readFileSync(`${ROOT}/pilot/results/analysis.json`));
const src = (f) => fs.readFileSync(`${ROOT}/${f}`, "utf8").trimEnd();

const f3 = (x) => x.toFixed(3);
const f2 = (x) => x.toFixed(2);
const pv = (p) => (p < 0.001 ? "< .001" : "= " + p.toFixed(3).replace(/^0/, ""));
const pv2 = (p) => (p < 0.001 ? "< .001" : p.toFixed(3).replace(/^0/, ""));
const ci = (c) => `[${f3(c[0])}, ${f3(c[1])}]`;
const SHORT = {
  "SmolLM2-1.7B-Instruct": "SmolLM2-1.7B", "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
  "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": "Qwen3-30B-A3B", "Qwen3.5-0.8B": "Qwen3.5-0.8B", "Qwen3.5-2B": "Qwen3.5-2B",
  "Qwen3.5-4B": "Qwen3.5-4B", "Qwen3.5-9B": "Qwen3.5-9B", "Qwen3.5-35B-A3B-Q4_K_M": "Qwen3.5-35B-A3B",
  "gemma-4-E4B-it": "Gemma-4-E4B", "gemma-4-26B-A4B-it-Q4_K_M": "Gemma-4-26B-A4B", "gpt-oss-20b-MXFP4": "gpt-oss-20b",
  "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "LLama-3.1-KazLLM-1.0-8B": "KazLLM-8B", "Llama-3.1-Sherkala-8B-Chat": "Sherkala-8B",
  "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": "mDeBERTa (NLI clf.)", "xlm-roberta-large-xnli": "XLM-R-large (excluded)",
};
const QUANT = { "none (bf16)": "bf16", "4bit": "NF4", "Q4_K_M": "Q4_K_M", "none (fp32)": "fp32" };
const M = Object.keys(A);
const INC = M.filter((m) => !A[m].excluded);
const LLM = INC.filter((m) => !A[m].backend.includes("NLI"));
const a = (m) => A[m], e = (m) => E[m];
const pct = (x) => (100 * x).toFixed(1);
const gd = (k) => E.gap_differences[k];

// ---------- result tables ----------
const mainRows = M.map((m) => [
  SHORT[m], QUANT[a(m).quantization] || a(m).quantization, f3(a(m).acc_en), f3(a(m).acc_kk),
  `${f3(a(m).acc_gap)}\n${ci(a(m).acc_gap_CI95)}`, f3(a(m).macroF1_en), f3(a(m).macroF1_kk),
  `${a(m).mcnemar_b_en_only} / ${a(m).mcnemar_c_kk_only}`,
  a(m).excluded ? "–" : pv2(e(m).mcnemar_p_holm_one_sided),
  a(m).excluded ? "–" : pv2(e(m).kk_vs_chance_p),
  a(m).excluded ? "–" : (e(m).above_chance_retention == null ? "–" : f2(e(m).above_chance_retention)),
]);
const C = [
  ["Qwen2.5-0.5B-Instruct minus Qwen2.5-1.5B-Instruct", "Scale (Qwen2.5)", "0.5B − 1.5B"],
  ["Qwen3.5-0.8B minus Qwen3.5-2B", "Scale (Qwen3.5)", "0.8B − 2B"],
  ["Qwen3.5-2B minus Qwen3.5-4B", "Scale (Qwen3.5)", "2B − 4B"],
  ["Qwen3.5-4B minus Qwen3.5-9B", "Scale (Qwen3.5)", "4B − 9B"],
  ["Qwen3.5-0.8B minus Qwen3.5-9B", "Scale (Qwen3.5)", "0.8B − 9B"],
  ["Qwen3.5-4B minus gemma-4-E4B-it", "Family at fixed scale", "Qwen3.5-4B − Gemma-4-E4B"],
  ["gemma-4-E4B-it minus gemma-4-26B-A4B-it-Q4_K_M", "Total vs active parameters", "Gemma-4-E4B − Gemma-4-26B-A4B"],
  ["Qwen3-30B-A3B-Instruct-2507-Q4_K_M minus Qwen3.5-9B", "Generation / tokenizer", "Qwen3-30B-A3B − Qwen3.5-9B"],
  ["Llama-3.1-8B-Instruct minus LLama-3.1-KazLLM-1.0-8B", "Kazakh adaptation", "Llama-3.1-8B − KazLLM"],
  ["Llama-3.1-8B-Instruct minus Llama-3.1-Sherkala-8B-Chat", "Kazakh adaptation", "Llama-3.1-8B − Sherkala"],
  ["Llama-3.1-Sherkala-8B-Chat minus LLama-3.1-KazLLM-1.0-8B", "Kazakh adaptation", "Sherkala − KazLLM"],
  ["SmolLM2-1.7B-Instruct minus Qwen2.5-1.5B-Instruct", "Multilingual coverage", "SmolLM2 − Qwen2.5-1.5B"],
  ["SmolLM2-1.7B-Instruct minus Qwen3.5-2B", "Multilingual coverage", "SmolLM2 − Qwen3.5-2B"],
  ["Qwen3.5-9B minus mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "Prompting vs supervised", "Qwen3.5-9B − mDeBERTa"],
];
const contrastRows = C.map(([k, f, lab]) => [f, lab, f3(gd(k).point), ci(gd(k).CI95), pv2(gd(k).p_holm)]);
const O4 = E.O4_logit_kk_correct_given_en_correct;
const G = E.O5_gee_language_x_scale;
const orRow = (lab, c) => [lab, f2(c.OR), `[${f2(c.CI95_OR[0])}, ${f2(c.CI95_OR[1])}]`, pv2(c.p)];

const runtime = (s) => (s > 3600 ? (s / 3600).toFixed(1) + " h" : (s / 60).toFixed(1) + " min");
const META = {};
for (const m of M) META[m] = JSON.parse(fs.readFileSync(`${ROOT}/results/meta_${m}.json`));

// ---------- content ----------
const B = [];
const T = (t, text) => B.push({ t, text });
const P = (text) => B.push({ t: "p", text });

T("title", "Assignment #3. Methodology and Core Practical Implementation");
T("center", "Course: Research Methods and Tools (RMT)");
T("center", "Students: Ayauzhan Baurzhanova, Aiat Mergenbay");
T("center", "Group: SE-2416");
T("center", "*Final version: full study, 14 models analysed (October 2026)*");

T("h1", "1. Final / Revised Research Title");
P("**Evaluating Zero-Shot Cross-Lingual Transfer from English to Kazakh in Large Language Models on Natural Language Inference Tasks**");
P("The title is unchanged from Assignment 2. It fixes the transfer setting (zero-shot), the language pair (English to Kazakh), the object of study (large language models) and the measurement task (natural language inference, NLI).");

T("h1", "2. Primary Research Question, Aim and Objectives");
P("**Primary research question (RQ):** To what extent does zero-shot cross-lingual transfer from English to Kazakh affect the accuracy of large language models on natural language inference tasks?");
P("**SQ1:** How large is the accuracy gap between parallel English and Kazakh NLI items under identical zero-shot prompting?");
P("**SQ2:** Does the size of this gap vary across model families and parameter scales?");
P("**Aim:** To investigate the effect of zero-shot cross-lingual transfer from English on the accuracy of open-source large language models in Kazakh natural language inference, by quantifying the performance gap between the two languages on a parallel benchmark.");
P("**Objectives** (unchanged from Assignment 2):");
B.push({ t: "list", ordered: true, ref: "obj", items: [
  "O1 – Identify and prepare a parallel English–Kazakh NLI dataset based on an established benchmark.",
  "O2 – Evaluate selected open-source LLMs on the English and Kazakh versions of the dataset under identical zero-shot prompting conditions.",
  "O3 – Compare accuracy and F1-scores between English and Kazakh to quantify the cross-lingual transfer gap (SQ1).",
  "O4 – Analyse error patterns to determine whether transfer failures are associated with the morphological complexity of Kazakh items.",
  "O5 – Assess whether the size of the transfer gap varies with model family and parameter scale (SQ2).",
] });
P("**H1:** Zero-shot accuracy on Kazakh NLI items will be significantly lower than on the parallel English items. **H2:** The English–Kazakh gap will be smaller for larger models with broader multilingual coverage than for smaller models.");
P("**Refinement since Assignment 2.** The dataset for O1 is now concrete: instead of translating XNLI ourselves, we use the existing Kardeş-NLU Kazakh XNLI test set (Senel et al., 2024), which already pairs each English XNLI item with a post-edited Kazakh translation. This keeps O1 within scope and removes a large translation workload. The model set was also revised and widened (Section 4.1): the Assignment 2 plan named only small dense models, whereas the final set adds newer model generations, mixture-of-experts models, two Kazakh-adapted models and a supervised NLI classifier, chosen so that SQ2 and O4 can be answered with controlled comparisons. The research question, aim, objectives and hypotheses are otherwise unchanged.");

T("h1", "3. Research Approach and Design");
P("The study follows a **quantitative, technical/computational approach**. The specific design is a **comparative zero-shot benchmark experiment with a paired (within-item) structure**: every NLI item is given to every model twice, once in English and once in its Kazakh translation, while the prompt, label words, decoding rule, numerical precision and hardware are held constant within each model. Item language is the only manipulated condition; model family, scale, architecture (dense vs mixture-of-experts) and Kazakh adaptation are between-model factors (SQ2).");
P("The **unit of analysis** is one model prediction on one item in one language (model × item × language). For hypothesis testing, predictions are paired by item, so each comparison is the same model answering the same item in two languages.");
P("The design produces objective numerical evidence: item-level correctness aggregated into accuracy, macro-F1 and an English–Kazakh gap with a confidence interval and significance test. It fits the RQ because \"to what extent\" asks for the size of an effect, and pairing removes item difficulty as an explanation: if a model solves an item in English but not in Kazakh, the difference comes from the change of language (including its translation).");
P("**Rejected alternative.** A fine-tuning experiment (training models on Kazakh NLI data and comparing them with zero-shot models) was rejected. It measures the benefit of adaptation rather than zero-shot transfer, which is what the RQ asks about; Kazakh NLI training data barely exists; and fine-tuning several models needs GPU resources beyond the project. Models that were already adapted to Kazakh by their developers (KazLLM, Sherkala) are evaluated zero-shot like every other model; we do no training ourselves.");

T("h1", "4. Sample / Dataset / Cases / Research Context");
P("**Items.** The research material is the Kazakh XNLI test split of Kardeş-NLU (Senel et al., 2024): 2,000 English XNLI test items (Conneau et al., 2018), each paired with a Kazakh translation and a gold label. The Kazakh side was produced with Google Translate and post-edited by one native speaker. Labels are balanced (670 / 671 / 659).");
P("**Inclusion:** a valid gold label, non-empty text in both languages, and Kazakh text at least 80% Cyrillic. **Exclusion:** items failing the script check, untranslated pairs (Kazakh identical to English), and Kazakh/English length ratios below 0.5 or above 2.0 (likely misaligned translations).");
P(`**Size and selection.** The full study evaluates all ${DS.eval_set_size.toLocaleString("en")} eligible items (a census, so the researchers do not choose items; labels ${DS.eval_label_distribution.entailment} / ${DS.eval_label_distribution.neutral} / ${DS.eval_label_distribution.contradiction}), i.e. 3,936 predictions per model. With about 2,000 paired items, an exact McNemar test detects a gap of roughly three percentage points at α = .05. The pilot (Sections 9.1–9.5) used a stratified random sample of 300 items (100 per label, seed 42) so it could run on a CPU.`);
P("**Models** are chosen by criterion (purposive) sampling; the criteria, the final set and the reason for each model are given in Section 4.1.");
P("**Access.** The dataset is public on GitHub (commit 48eea77) and the models on Hugging Face. Three models are gated: Llama-3.1-8B-Instruct (Llama 3.1 Community Licence, approved by Meta), KazLLM (CC BY-NC 4.0) and Sherkala (CC BY-NC-SA 4.0); access was requested with the authors' real names for non-commercial academic use and was granted.");
P("**Bias.** (1) The authors do not say how their 2,000-item subset was chosen, so genres may be unevenly represented; we report distributions and apply no further filtering. (2) The script check over-excludes correct translations containing Latin-script names (32 items); these items were not evaluated, so the effect of names is a limitation rather than a tested factor. (3) Conclusions apply only to open models that run on a consumer laptop (dense models up to 9B parameters, mixture-of-experts models up to 30B total and about 4B active parameters).");

T("h2", "4.1 Model selection and justification");
P("**Inclusion criteria.** A model is included if it (1) has open weights on Hugging Face; (2) has an instruction- or chat-tuned version, which zero-shot prompting needs; (3) runs on the study laptop (RTX 4060 Laptop GPU with 8 GB VRAM, 14 GB RAM), either fully on the GPU or with 4-bit weights and parts kept in CPU RAM; and (4) has a model card that documents its training languages or its Kazakh adaptation. On top of these criteria, the set as a whole must change one factor at a time, so that each comparison in SQ2 and O4 has a matching pair of models (Table 2). A model is only worth its runtime if it adds such a contrast.");
B.push({ t: "table", cols: [2.1, 1.3, 1.4, 1.5, 4.2], size: 16,
  head: ["Model (Hugging Face id)", "Family / developer", "Parameters", "Run as", "Role in the design and justification"],
  rows: [
    ["Qwen2.5-0.5B-Instruct", "Qwen (Alibaba)", "0.5B dense", "transformers, bf16", "Pilot model, kept so the pilot and the full study connect. Lowest scale point of the older Qwen generation; card lists 29+ languages, Kazakh not among the named ones."],
    ["Qwen2.5-1.5B-Instruct", "Qwen (Alibaba)", "1.5B dense", "transformers, bf16", "Second scale point of Qwen2.5 and the best pilot model; with 0.5B gives a within-family scale step on the old tokenizer."],
    ["SmolLM2-1.7B-Instruct", "Hugging Face", "1.7B dense", "transformers, bf16", "English-centric control: its card states it \"primarily understands and generates content in English\". Shows what the gap looks like without multilingual training; it also fragments Kazakh most (6.92 tokens per word)."],
    ["Qwen3.5-0.8B / 2B / 4B / 9B", "Qwen (Alibaba)", "0.8B, 2B, 4B, 9B dense", "bf16 (0.8B, 2B), 4-bit NF4 (4B, 9B)", "Main scale ladder for H2: one generation, one tokenizer and one training recipe, so differences between the four come from scale. Newest Qwen generation; card claims 201 languages and dialects, and its new tokenizer splits Kazakh into 3.45 tokens per word instead of 4.69 (Table 3)."],
    ["gemma-4-E4B-it", "Gemma (Google)", "4.5B effective (8B with per-layer embeddings)", "4-bit NF4, embedding table in CPU RAM", "Second family at the same scale as Qwen3.5-4B, so family can be compared at fixed size. Card: pre-trained on 140+ languages."],
    ["gemma-4-26B-A4B-it", "Gemma (Google)", "≈26B total / 3.8B active (MoE)", "llama.cpp, Q4_K_M, experts in CPU RAM", "Same family and similar active compute per token as E4B, but about six times the stored parameters. Tests whether total capacity (room for rare-language knowledge), rather than compute per token, narrows the gap."],
    ["Qwen3-30B-A3B-Instruct-2507", "Qwen (Alibaba)", "30.5B total / 3.3B active (MoE)", "llama.cpp, Q4_K_M", "Previous Qwen generation; uses the same tokenizer as Qwen2.5 (identical token counts, Table 3). Against Qwen3.5 it separates generation and tokenizer from size."],
    ["Llama-3.1-8B-Instruct", "Llama (Meta)", "8B dense", "4-bit NF4", "Planned in Assignment 2. Officially supports 8 languages, not Kazakh. Serves as the shared base of the two Kazakh models below."],
    ["LLama-3.1-KazLLM-1.0-8B", "ISSAI (Kazakhstan)", "8B dense", "4-bit NF4", "Llama-3.1-8B further trained by ISSAI to improve Kazakh, keeping the original tokenizer (same 4.71 tokens per word). Against the base it isolates the effect of Kazakh training data."],
    ["Llama-3.1-Sherkala-8B-Chat", "Inception / MBZUAI", "8B dense", "4-bit NF4", "Llama-3.1-8B adapted to Kazakh with an extended tokenizer (159,766 vs 128,256 tokens; Kazakh drops from 4.71 to 1.87 tokens per word). With the two models above it forms a three-way contrast: no adaptation, data only, data + tokenizer."],
    ["mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "DeBERTa-v3 (Microsoft), fine-tuned by Laurer", "0.28B encoder", "transformers, fp32", "Supervised NLI reference (\"decision model\"): an encoder trained on NLI in 27 languages, not Kazakh, whose base model was pre-trained on CC100 (which includes Kazakh). Compares task-specific cross-lingual transfer with prompting."],
    ["xlm-roberta-large-xnli", "XLM-R (Meta), fine-tuned by J. Davison", "0.56B encoder", "transformers, fp32", "Intended second NLI reference. **Excluded from all comparisons**: its card states it was trained on the XNLI validation and test sets, which contain the English items of Kardeş-NLU; it scores 0.999 in English. Kept only as a documented contamination case (Section 8)."],
  ],
  caption: "Table 1. Final model set (15 models). 14 models are analysed; XLM-R is excluded (Section 8). Parameters are taken from the model cards; tokens per word are measured on the 1,968 evaluation items (Section 5.1)." });
B.push({ t: "table", cols: [2, 3, 3],
  head: ["Contrast (factor varied)", "Models compared", "Question answered"],
  rows: [
    ["Scale within one family", "Qwen3.5 0.8B → 2B → 4B → 9B; Qwen2.5 0.5B → 1.5B", "H2: does the gap shrink as models grow?"],
    ["Family at fixed scale", "Qwen3.5-4B vs Gemma-4-E4B", "SQ2: is the gap a property of the training recipe rather than size?"],
    ["Total vs active parameters", "Gemma-4-E4B vs Gemma-4-26B-A4B", "Does stored capacity help Kazakh when compute per token is similar?"],
    ["Generation and tokenizer", "Qwen3-30B-A3B (old tokenizer) vs Qwen3.5-9B (new tokenizer)", "O4: does a tokenizer that fragments Kazakh less go with a smaller gap?"],
    ["Kazakh adaptation", "Llama-3.1-8B vs KazLLM (data) vs Sherkala (data + tokenizer)", "Does targeted Kazakh training close the gap, and does the tokenizer add to it?"],
    ["Multilingual coverage", "SmolLM2-1.7B (English-centric) vs Qwen2.5/Qwen3.5 of similar size", "Lower bound: transfer without multilingual pre-training."],
    ["Prompting vs supervised", "Best LLM vs mDeBERTa NLI classifier", "How far is zero-shot prompting from a supervised cross-lingual classifier?"],
  ],
  caption: "Table 2. Planned contrasts. Each factor is varied while the others are held as constant as the available models allow." });
P("**Models planned in Assignment 2 but dropped.** Qwen2.5-3B/7B were replaced by the Qwen3.5 ladder, which fills the same role (scale within one family) with a newer generation, a much broader language list and a tokenizer that fits Kazakh better. Llama-3.2-1B/3B do not list Kazakh among their supported languages, and the Llama family is now represented by Llama-3.1-8B together with its two Kazakh-adapted versions, which answer a sharper question than two more scale points. Mistral-7B was dropped to keep the total runtime manageable on one laptop (each MoE run alone takes 5–7 hours); its role, a second family at 7–9B, is covered by Gemma 4 and Llama-3.1. Two further models were planned for the full study but not run within the project's time: Qwen3.5-35B-A3B, the top of the Qwen3.5 ladder (a 22 GB file, larger than the laptop's VRAM and RAM combined, so parts would have had to be read from the SSD), and gpt-oss-20b (OpenAI, 2025), which would have added a third family and an exploratory run with reasoning. The contrasts they were meant to answer are listed as open questions in Section 9.8.");

T("h1", "5. Variables / Constructs / Measures / Instruments / Metrics");
P("**Independent variable:** item language (English vs Kazakh), manipulated within items. **Between-model factors (SQ2):** model family, parameter count (log scale; total and active parameters are both recorded for MoE models), architecture (dense vs MoE) and Kazakh adaptation.");
P("**Dependent variables:** correctness of each prediction (binary), aggregated into accuracy, macro-F1 and per-class F1. The **transfer gap** is Δ = accuracy(EN) − accuracy(KK), reported together with the macro-F1 gap and the **above-chance retention** R = (acc_KK − 1/3) / (acc_EN − 1/3), the share of a model's English above-chance performance that survives in Kazakh. Accuracy is appropriate because labels are balanced; macro-F1 is added because a model can reach chance accuracy by predicting one label, which F1 exposes; R was added after the pilot showed that a weak English score makes the raw gap look small (Section 9.4).");
P("**Explanatory measures for O4:** (a) **tokenizer fertility**, the number of subword tokens per whitespace word (Rust et al., 2021; Petrov et al., 2023), showing how heavily a model fragments agglutinative Kazakh words, and the number of tokens per premise–hypothesis pair (both computed as described in Section 5.1); (b) mean Kazakh word length in characters, a simple proxy for suffix stacking; (c) the direction of each error (which wrong label replaces the gold one); (d) the confusion matrix, which shows whether failures concentrate in one class.");
P("**Controlled variables:** the same English instruction and label words for both languages, the same items, deterministic decoding, and the same hardware for every model; within each model, the same precision for both languages. **Possible confounders:** translation quality (machine translation plus one post-editor), the model's prior preference for particular label words, possible exposure to the public English XNLI items during pre-training, which would inflate English scores, and differences in weight precision between models (Section 6, Step 5).");
P("**Instrument.** Our evaluation harness has three evaluators that write the same output columns: `evaluate.py` (Hugging Face Transformers; Wolf et al., 2020) for models that fit the GPU, `evaluate_gguf.py` for GGUF models served locally by llama.cpp (Gerganov et al., 2023–2026), and `evaluate_nli.py` for NLI classifiers. For every prediction they log the gold and predicted label, label probabilities and token counts. **Baselines:** chance accuracy (33.3%) and majority-class accuracy (33.6% on the full set).");

T("h2", "5.1 How tokens are computed and why");
P("The pipeline computes tokens for two different purposes: to **score the three labels** (the decision rule) and to **measure how much a tokenizer fragments each language** (the O4 measures). Both are computed with each model's own tokenizer, deterministically, and for every prediction.");
P("**(a) Label scoring: which token decides the answer.**");
B.push({ t: "steps", items: [
  ["Step 1.", "**Label tokens.** Each label word (entailment, neutral, contradiction) is encoded with the model's tokenizer without special tokens, and only its first token id is kept. The script asserts that the three first tokens are different; for Qwen2.5 they are ids 306 / 59568 / 8222, for Qwen3.5 305 / 57555 / 7965, and for the Llama-3.1 tokenizer (also used by KazLLM and Sherkala) 306 / 60668 / 8386 (all ids are stored in each run's metadata file)."],
  ["Step 2.", "**Prompt tokens.** The system and user messages are rendered with the model's own chat template, ending with the opening of the assistant turn. Qwen3/Qwen3.5 are rendered with thinking disabled, so the assistant turn starts directly with the answer. Sherkala's repository ships no chat template, so the template its model card sets in code is used. The rendered text is tokenized without adding special tokens, because the template already contains the beginning-of-sequence token where the model expects it (adding it again would give two). In llama.cpp the prompt is sent as a list of token ids for the same reason."],
  ["Step 3.", "**Scoring.** One forward pass gives the scores (logits) of every vocabulary token at the next position. The three label first-token scores are taken, the prediction is the label with the highest score, and a softmax over these three scores gives p_ent, p_neu and p_con, which are used only for analysis."],
] });
P("**Formulas.** For model *m*, the label first tokens, the label scores, the prediction and the reported probabilities are:");
B.push({ t: "eq", n: 1, text: "t~c~ = tok~m~(c)[1],   c ∈ {entailment, neutral, contradiction}" });
B.push({ t: "eq", n: 2, text: "s~c~ = z~n~[t~c~]" });
B.push({ t: "eq", n: 3, text: "ŷ = argmax~c~ s~c~" });
B.push({ t: "eq", n: 4, text: "p~c~ = exp(s~c~) / Σ~c′~ exp(s~c′~)" });
P("where tok~m~ is the model's tokenizer applied without special tokens, [1] takes the first token, z~n~ is the vector of next-token scores (logits) at the last prompt position *n*, and the sum in (4) runs over the three labels only. In llama.cpp, z~n~ is replaced by the log-probabilities the server returns for its 200 most likely tokens (top-200):");
B.push({ t: "eq", n: 5, text: "s~c~ = log P(t~c~) if t~c~ ∈ top-200;  otherwise s~c~ = min~t ∈ top-200~ log P(t) − 1" });
P("Because the softmax in (4) is monotonic, the label with the highest probability is always the predicted label ŷ from (3).");
P("**Justification.** (1) It is **deterministic**: there is no sampling, so a rerun gives identical predictions. (2) There is **no free text to parse**, so no answer can be invalid or mis-parsed, and output formatting cannot differ between English and Kazakh items; such rank classification is the standard way to run multiple-choice tasks zero-shot (Brown et al., 2020; Sanh et al., 2022). (3) **One forward pass per item** is the cheapest possible evaluation, which is what makes 3,936 predictions × 15 models feasible on a laptop. (4) The **first token is enough** to tell the labels apart, and it avoids a length bias: summing the scores of all tokens of a label would penalise labels that the tokenizer splits into more pieces (for Qwen, \"neutral\" is one token while \"entailment\" and \"contradiction\" are several). The cost is that the scores reflect the model's prior preference for each label word; this is monitored through prediction distributions and macro-F1 (Section 8), and it turned out to matter for several models (Section 9.7).");
P(`**llama.cpp detail.** The server returns the log-probabilities of the 200 most likely next tokens. If a label's first token is not among them, it is given a value just below the lowest listed one and the item is flagged. This happened for ${META["gemma-4-26B-A4B-it-Q4_K_M"].items_with_label_outside_top_n} of 3,936 predictions of Gemma-4-26B-A4B and ${META["Qwen3-30B-A3B-Instruct-2507-Q4_K_M"].items_with_label_outside_top_n} of Qwen3-30B-A3B, always for a label that was not predicted and never for all three labels, so no prediction was affected; only the reported probability of that unlikely label is approximate. NLI classifiers do not use label tokens: their three-way output layer is read directly, and the label order is taken from each model's configuration because it differs between checkpoints.`);
P("**(b) Token counts and fertility: how much each language is fragmented.** For item *i* in language ℓ ∈ {en, kk}, with premise p~i,ℓ~ and hypothesis h~i,ℓ~, the pipeline computes:");
B.push({ t: "eq", n: 6, text: "T~i,ℓ~ = | tok~m~(p~i,ℓ~ ⊕ \" \" ⊕ h~i,ℓ~) |" });
B.push({ t: "eq", n: 7, text: "W~i,ℓ~ = | split(p~i,ℓ~ ⊕ \" \" ⊕ h~i,ℓ~) |" });
B.push({ t: "eq", n: 8, text: "F~i,ℓ~ = T~i,ℓ~ / W~i,ℓ~" });
P("where ⊕ joins two strings, |·| counts tokens or words, and split cuts the text at whitespace. T is the number of tokens of the pair (`pair_tokens`), W the number of words (`pair_words`) and F the fertility. Over the N = 1,968 items, the per-language means and the Kazakh token premium of model *m* are:");
B.push({ t: "eq", n: 9, text: "F̄~ℓ~ = (1/N) Σ~i~ F~i,ℓ~,    T̄~ℓ~ = (1/N) Σ~i~ T~i,ℓ~" });
B.push({ t: "eq", n: 10, text: "premium~m~ = T̄~kk~ / T̄~en~" });
P("Fertility (9) is the mean of the per-item ratios, so every item counts equally regardless of its length; the premium (10) is a ratio of mean token counts, i.e. how many times longer the same content becomes in Kazakh. Table 3 reports (9) and (10) for every tokenizer. In code, (6)–(8) are:");
B.push({ t: "code", text: `pair_tokens = len(tokenizer.encode(premise + " " + hypothesis, add_special_tokens=False))   # T
pair_words  = len((premise + " " + hypothesis).split())                                        # W
fertility   = pair_tokens / pair_words                                                         # F` });
P("**Worked example** (item 2788, Qwen2.5 tokenizer; the item is shown in Appendix D). English: T = 25 tokens, W = 24 words, F = 25 / 24 = 1.04. Kazakh: T = 85, W = 18, F = 85 / 18 = 4.72. The first Kazakh word, *Жұмыс* (\"work\"), is split into four pieces (Ж | ұ | мы | с) and *беруші* (\"giver\") into three (бер | уш | і), while most English words are a single token. Scoring the same item with Qwen2.5-1.5B by (2)–(4) gives p = (0.001, 0.003, 0.996) for (entailment, neutral, contradiction) in English, correctly contradiction, and (0.893, 0.024, 0.083) in Kazakh, i.e. the wrong label, entailment.");
P("**Justification.** (1) Only the premise and hypothesis are counted. The instruction and chat template are English and identical in both conditions, so including them would add the same constant to both languages and hide the difference that matters. Special tokens are excluded for the same reason, and because they differ between tokenizers. (2) Each model's own tokenizer is used, because fertility should describe what that model actually receives. Models that share a tokenizer get identical values, which doubles as a check of the procedure: Qwen2.5 and Qwen3-30B-A3B, the four Qwen3.5 models, the two Gemma-4 models, and Llama-3.1 and KazLLM all match exactly (Table 3). (3) Whitespace words are the denominator because the rule is the same in both languages and needs no Kazakh morphological analyser; this follows the definition of fertility in Rust et al. (2021). (4) Tokens per pair are reported as well, because Kazakh words are longer (20.9 words per pair vs 27.4 in English for the same content), so even an ideal tokenizer would have a higher per-word fertility in Kazakh. Tokens per pair compare the same content in both languages directly, which is the \"token premium\" of Petrov et al. (2023); for example, the Qwen2.5 tokenizer needs 2.85 times as many tokens for a Kazakh pair as for its English original. (5) The same procedure is used for every backend: the Transformers tokenizer, llama-server's `/tokenize` endpoint with special tokens switched off (the GGUF file carries the same vocabulary; Gemma-4 and Qwen3 values match their Transformers counterparts exactly), and the encoders' SentencePiece tokenizers. (6) The values are stored per prediction row, so they can be joined to correctness for the item-level tests in Section 7.");
B.push({ t: "table", cols: [3, 1.2, 1.1, 1.1, 1, 1.1, 1.1], align: ["l", "r", "r", "r", "r", "r", "r"],
  head: ["Tokenizer (models using it)", "Vocabulary", "EN tokens / pair", "KK tokens / pair", "KK / EN", "Fertility EN", "Fertility KK"],
  rows: [
    ["Qwen2.5 (Qwen2.5-0.5B/1.5B, Qwen3-30B-A3B)", "151,665", "34.4", "98.0", "2.85", "1.264", "4.687"],
    ["Qwen3.5 (0.8B, 2B, 4B, 9B)", "248,077", "34.3", "71.9", "2.10", "1.260", "3.449"],
    ["Gemma 4 (E4B, 26B-A4B)", "262,144", "34.2", "65.9", "1.93", "1.258", "3.163"],
    ["SmolLM2", "49,152", "34.8", "144.9", "4.16", "1.279", "6.917"],
    ["Llama-3.1 (Llama-3.1-8B, KazLLM)", "128,256", "34.0", "98.6", "2.90", "1.248", "4.714"],
    ["Sherkala (extended Llama-3.1)", "159,766", "34.0", "38.6", "1.14", "1.248", "1.865"],
    ["mDeBERTa-v3 (SentencePiece)", "250,102", "41.9", "48.9", "1.17", "1.539", "2.354"],
    ["XLM-R (SentencePiece)", "250,002", "38.0", "42.3", "1.11", "1.398", "2.050"],
  ],
  caption: "Table 3. Tokenizer statistics on all 1,968 items (means over items). English is tokenized almost identically by every LLM tokenizer, while Kazakh costs from 1.1 to 4.2 times as many tokens; this spread is what O4 relates to the accuracy gap." });
P("Table 3 already shows why the model set includes the tokenizer contrasts in Table 2: the English cost is nearly constant (33–35 tokens per pair for every LLM), while the Kazakh cost ranges from 38.6 (Sherkala) to 144.9 tokens (SmolLM2) for the same sentences. Llama-3.1-8B, KazLLM and Sherkala are the cleanest test, because they share an architecture and a starting point and differ only in Kazakh training data and the tokenizer.");

T("h1", "6. Data Collection / Data Preparation / Experimental Procedure");
B.push({ t: "steps", items: [
  ["Step 1.", "**Preparation.** Clone Kardeş-NLU at a fixed commit and record Python, PyTorch, Transformers and llama.cpp versions together with the exact model identifiers (Hugging Face ids, GGUF file names); every run writes these to its metadata file. The 1,000-item dev split is reserved for prompt piloting; the test split is never used to choose prompts or settings, which prevents test-set leakage."],
  ["Step 2.", "**Preprocessing.** Apply Unicode NFC normalisation and collapse whitespace; compute the Cyrillic ratio, length ratio and untranslated-text flags; exclude flagged items; draw the stratified pilot sample (full study: all eligible items)."],
  ["Step 3.", "**Prompt.** One fixed template asks for a one-word answer (entailment, neutral or contradiction), wrapped in the model's chat template. The instruction is always English and only the premise and hypothesis change language, so prompt wording cannot cause the gap."],
  ["Step 4.", "**Inference (zero-shot).** No examples and no fine-tuning. One forward pass per item; the prediction is the label whose first token has the highest score at the first answer position (constrained \"rank classification\", Section 5.1). There is no sampling, so repeated runs give identical outputs. Models that fit the GPU run in Transformers; mixture-of-experts models run in llama.cpp, which keeps the experts that do not fit in VRAM in CPU RAM; NLI classifiers are read from their classification head."],
  ["Step 5.", "**Fairness between conditions.** Same items, order and settings for every model; no language-specific tuning. Pilot: CPU, bfloat16. Full study: one RTX 4060 Laptop GPU (8 GB). Identical precision for all models is impossible in 8 GB, so one rule is applied: bfloat16 when the model fits (up to 2B), otherwise 4-bit NF4 (Dettmers et al., 2023) for dense models, Q4_K_M for the two MoE GGUF models. Embedding tables that 4-bit quantisation does not shrink (Qwen3.5-9B, Gemma-4-E4B) are kept in CPU RAM; this changes where they are computed, not the predictions (checked against an all-GPU run). Precision is always the same for the English and Kazakh items of a model, so the within-model gap is unaffected; for between-model comparisons it is a recorded confounder."],
  ["Step 6.", "**Recording and storage.** Each prediction is written as one CSV row (model, item ID, language, gold, prediction, label probabilities, token count, fertility). Raw outputs are never edited; code and outputs are kept in the team's Git repository."],
  ["Step 7.", "**Manual error audit.** A Kazakh-speaking team member codes a random sample of \"flips\" (correct in English, wrong in Kazakh) as translation error, morphology/negation, named entity or loanword, label ambiguity, or other, and rates the adequacy of 50 randomly chosen Kazakh pairs. This step is not yet complete (Section 9.8)."],
] });

T("h1", "7. Data Analysis Plan");
P("**O1 (dataset).** Descriptive data-quality statistics: exclusions per rule, label distribution, length ratios, and the share of audited Kazakh pairs rated adequate, minor or major error.");
P("**O2–O3 (SQ1, H1).** Per model: accuracy and macro-F1 per language and the gap Δ. A paired bootstrap (5,000 resamples of items, seed 42; Efron & Tibshirani, 1993) gives a 95% confidence interval for Δ. H1 is tested with an exact McNemar test on items answered correctly in only one language (McNemar, 1947), one-sided in the direction of H1, with Holm correction across models (Holm, 1979). Each model's Kazakh accuracy is also tested against chance (exact binomial test) and summarised as above-chance retention R. Paired tests follow recommendations for comparing NLP systems on a shared test set (Dror et al., 2018). H1 is supported for a model if the corrected p < .05 and the confidence interval excludes zero.");
P("**O4 (error patterns).** Among items a model solves in English, Kazakh failures are compared with Kazakh successes on tokenizer fertility and mean word length (Mann–Whitney U test), followed by a pooled logistic regression: KK correct ∼ fertility + word length + gold label + model, with standard errors clustered by item because each item appears once per model. Fertility is standardised within each model, so its coefficient compares items within the same tokenizer. Across models, the gap is related to each tokenizer's Kazakh token premium (Table 3), with the Llama-3.1 / KazLLM / Sherkala and Qwen3 / Qwen3.5 pairs as the controlled cases. Confusion matrices and the direction of flips show which classes dominate.");
P("**O5 (SQ2, H2).** The gap is compared across models against log parameter count and family. The main test is a logistic model of correctness, correct ∼ language × log~2~(parameters) + family, fitted on all LLM predictions; it is fitted once with total and once with active parameters, because the two differ for MoE models, and once more with a language × Kazakh-adaptation term. H2 is supported if the language × scale interaction is significant and points towards a smaller Kazakh penalty for larger models. The planned contrasts in Table 2 are tested with a paired bootstrap CI of the gap difference between the two models (two-sided bootstrap p), with Holm correction across contrasts.");
P("**Change from the earlier plan.** The planned mixed-effects logistic regression with a random intercept per item was replaced by a **generalised estimating equation (GEE)** logistic model with items as clusters and an exchangeable working correlation (Liang & Zeger, 1986), fitted in statsmodels (Seabold & Perktold, 2010). Both account for the fact that every item is answered 26 times (13 LLMs × 2 languages); the GEE gives population-averaged effects with robust standard errors and fits reliably on 51,168 rows, whereas the only binomial mixed model available in our Python stack is a variational-Bayes approximation. The question the model answers, whether the language effect changes with scale, is the same.");
P("**Interpretation rule.** Kazakh accuracy is always read against chance, and prediction distributions are reported so label bias is not mistaken for transfer. The NLI classifier uses a different decision rule and is reported beside, not inside, the LLM comparisons.");

T("h1", "8. Research Quality, Ethics, Privacy and Reproducibility");
P(`**Internal validity.** The paired design, constant instruction and deterministic decoding isolate the language effect. Threats: (a) translation artefacts: the Kazakh side is post-edited machine translation, so part of a gap may reflect translation; addressed by the adequacy audit and a gap on adequate-only items. (b) Prompt sensitivity: only one template was used; template and label-order variants on the dev split are still to be run. (c) Label bias, monitored through prediction distributions and macro-F1; several models show strong label preferences (Section 9.7). (d) Contamination: English XNLI (public since 2018) may be in pre-training data, inflating English accuracy. One case is documented: xlm-roberta-large-xnli was fine-tuned on the XNLI test set according to its model card and reaches ${f3(A["xlm-roberta-large-xnli"].acc_en)} accuracy in English and ${f3(A["xlm-roberta-large-xnli"].acc_kk)} in Kazakh, so it is excluded from all comparisons; for the other models contamination is reported as a limitation. (e) Precision differs between models (bf16, NF4, Q4_K_M); it is constant within a model, recorded for every run, and reported next to every between-model comparison.`);
P("**Reliability and reproducibility.** Model identifiers and library versions are recorded; seed 42 is used for sampling and bootstrap; inference is deterministic; all scripts, raw predictions and logs are archived. The pilot reruns with four commands (`pilot/README.md`); the full study reruns with `run_batch1.sh`, `run_batch2.sh`, `analyze.py` and `extra_stats.py` (Appendix B). External validity is limited to open models that run on a consumer laptop and to the NLI task.");
P("**Ethics and privacy.** There are no human participants and no personal data, so consent is not required; the optional bonus survey was not attempted. The datasets are public research benchmarks and are used only for non-commercial academic evaluation, with full citation. All inference runs locally on open models (Transformers and llama.cpp on the authors' laptop); no data is sent to commercial AI or cloud APIs, so no third party processes the material. Model licences are respected: Apache-2.0 (Qwen, Gemma 4, SmolLM2), MIT (mDeBERTa, XLM-R), the Llama 3.1 Community Licence, and the non-commercial CC BY-NC 4.0 (KazLLM) and CC BY-NC-SA 4.0 (Sherkala) licences, which permit academic evaluation. AI assistance was used for code drafting and language checking; every methodological decision and every reported number was checked by the authors against the generated outputs.");

T("h1", "9. Core Practical Component / Implementation and Results");
P("**Track:** Computer Vision / NLP / Algorithmic Project (an evaluation pipeline for pre-trained models; the ML track's \"training\" step does not apply because the RQ is about zero-shot transfer). The artifact is a working three-stage pipeline, **prepare → evaluate → analyse**, first run on a pilot sample of the Kazakh XNLI benchmark with three open-source LLMs (Sections 9.1–9.5) and then on the full set with the model set of Section 4.1 (Sections 9.6–9.8). It supports O1 (dataset preparation), O2 (identical zero-shot evaluation), O3 (gap quantification and H1 test), O4 (tokenisation and error patterns) and O5 (scale, family and adaptation contrasts).");

T("h2", "9.1 Dataset source and preprocessing output");
B.push({ t: "table", cols: [1.4, 3.6],
  head: ["Step", "Output (from results/data_summary.json)"],
  rows: [
    ["Source", "Kardeş-NLU, data/kazakh/xnli.test.kk.csv (commit 48eea77): 2,000 parallel items"],
    ["Label distribution (source)", "entailment 670, neutral 671, contradiction 659"],
    ["Cleaning", "Unicode NFC normalisation, whitespace collapse; no missing values or duplicate pairs found"],
    ["Quality flags", "low Cyrillic ratio: 32; length outlier: 0; untranslated: 0"],
    ["Eligible items (full study)", `${DS.eligible_after_exclusion}; labels entailment ${DS.eval_label_distribution.entailment}, neutral ${DS.eval_label_distribution.neutral}, contradiction ${DS.eval_label_distribution.contradiction}`],
    ["Pilot sample", "300 items, stratified (100 per label), seed 42; each item evaluated in EN and KK"],
    ["Text statistics (full set)", `mean words per pair EN ${DS.mean_words_en} vs KK ${DS.mean_words_kk}; KK/EN character ratio ${DS.mean_len_ratio_kk_over_en_chars}`],
  ],
  caption: "Table 4. Dataset and preprocessing summary. Kazakh pairs contain fewer but longer words than English, as expected for an agglutinative language." });

T("h2", "9.2 Split strategy and model configuration (pilot)");
P("No model is trained, so there is no train/validation split. The dev split (1,000 items) is reserved for prompt design and the test split is used only for evaluation, which is the zero-shot equivalent of train/validation/test separation and prevents tuning on test data. The pilot used one fixed prompt written before any test results were seen.");
B.push({ t: "table", cols: [1.4, 3.6],
  head: ["Setting", "Value"],
  rows: [
    ["Models (pinned HF commits)", "Qwen/Qwen2.5-0.5B-Instruct (7ae5576), Qwen/Qwen2.5-1.5B-Instruct (989aa79), HuggingFaceTB/SmolLM2-1.7B-Instruct (31b70e2)"],
    ["Prompting", "zero-shot, English instruction + model chat template; identical for EN and KK"],
    ["Decision rule", "argmax over first-token logits of the three label words (constrained, deterministic)"],
    ["Precision / hardware", "bfloat16, CPU, 2 threads, 7 GB RAM"],
    ["Software", "Python 3.13.16, PyTorch 2.14.1+cpu, Transformers 5.20.2, scikit-learn, SciPy"],
    ["Runtime (600 predictions per model)", "Qwen2.5-0.5B-Instruct: 177 s; Qwen2.5-1.5B-Instruct: 391 s; SmolLM2-1.7B-Instruct: 467 s"],
  ],
  caption: "Table 5. Pilot model and run configuration." });

T("h2", "9.3 Pilot evaluation results");
const pRow = (m, lab) => { const r = PA[m]; return [lab, f3(r.acc_en), f3(r.acc_kk), `${f3(r.acc_gap)} ${ci(r.acc_gap_CI95)}`, f3(r.macroF1_en), f3(r.macroF1_kk), `${r.mcnemar_b_en_only} / ${r.mcnemar_c_kk_only}, ${pv2(r.mcnemar_p_two_sided)}`, `${f3(r.mean_fertility_en)} / ${f3(r.mean_fertility_kk)}`]; };
B.push({ t: "table", cols: [1.5, 0.8, 0.8, 1.9, 0.9, 0.9, 1.5, 1.3],
  head: ["Model", "Acc EN", "Acc KK", "Gap Δ [95% CI]", "Macro-F1 EN", "Macro-F1 KK", "McNemar b / c, p", "Fertility EN / KK"],
  rows: [pRow("Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B"), pRow("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B"), pRow("SmolLM2-1.7B-Instruct", "SmolLM2-1.7B")],
  caption: "Table 6. Pilot results (n = 300 paired items per model). b = items correct only in English; c = correct only in Kazakh; two-sided exact McNemar p. Fertility = subword tokens per word." });
B.push({ t: "img", src: `${ROOT}/pilot/figures/fig1_accuracy.png`, w: 5.6, caption: "Figure 1. Pilot: accuracy by language for each model; the dashed line is chance (0.333)." });
P("**Interpretation (SQ1, H1).** All three models are significantly less accurate on the Kazakh version of the same items. Qwen2.5-1.5B falls from 74.7% to 42.7% (Δ = 0.320, 95% CI [0.260, 0.380]) and Qwen2.5-0.5B from 64.3% to 30.0%. Discordant pairs are heavily one-sided (for Qwen2.5-1.5B, 108 items are correct only in English and 12 only in Kazakh), and the one-sided McNemar test stays significant for every model after Holm correction (all adjusted p < .001). H1 was therefore supported in the pilot.");
P("**Comparison with chance.** Only Qwen2.5-1.5B is above chance in Kazakh (binomial test vs 1/3, p < .001); Qwen2.5-0.5B (p = .902) and SmolLM2 (p = .522) are not. Expressed as above-chance retention, Qwen2.5-1.5B keeps about 23% of its English above-chance performance in Kazakh, while the other two keep none.");
P("**First evidence for SQ2.** Within Qwen2.5, the gap is 0.343 at 0.5B and 0.320 at 1.5B; the difference (0.023, 95% CI [−0.067, 0.113]) is not significant, so two small models cannot test H2. SmolLM2 shows the smallest raw gap (0.103), but only because its English accuracy is already low (43.7%); this is a floor effect, not better transfer (see 9.4).");

T("h2", "9.4 Pilot error analysis and the changes it caused");
B.push({ t: "img", src: `${ROOT}/pilot/figures/fig2_confusion_kk.png`, w: 6.0, caption: "Figure 2. Pilot: confusion matrices on the Kazakh items (rows = gold, columns = predicted)." });
P("**Failure case 1 – label collapse.** In Kazakh, SmolLM2 predicts \"contradiction\" for 279 of 300 items and never predicts \"entailment\", and Qwen2.5-0.5B predicts \"entailment\" only 12 times. When a model does not understand the input, it falls back to a default label, so the raw accuracy gap underestimates the transfer failure for models that are already weak in English. *Change:* the full study reports above-chance retention and macro-F1 next to Δ, and treats the gap as interpretable only when Kazakh accuracy is clearly above chance.");
P("**Failure case 2 – tokenisation cost.** Kazakh text is split into 4.68 tokens per word by the Qwen tokenizer versus 1.25 for English, and 6.89 versus 1.27 for SmolLM2. With three models this was an observation, not a test. *Change:* the full model set includes tokenizers with very different Kazakh costs, including a pair of models that differ only in their tokenizer (Section 4.1, Table 3).");
P("**Failure case 3 – translation shifts.** Some pilot flips come from the translation rather than the model. In item 3380, the drone name \"Predator\" is rendered as «жыртқыш» (a predatory animal); in item 504, the English question \"has it been raining\" becomes a statement that the rain is good (Appendix D). *Change:* the adequacy audit and a gap on adequate-only items were added to the plan.");
P("**Failure case 4 – script filter.** The Cyrillic-ratio rule excluded 32 items, mostly correct translations that contain long English proper names (e.g. \"Canal Motorboats BV\"). The rule was kept and is reported as a limitation (Section 4).");

T("h2", "9.5 Reproducibility");
P("All steps are scripted: `python prepare.py`, `./run_batch1.sh`, `./run_batch2.sh` (or `python evaluate.py <model_id>` for one model), `python analyze.py` and `python extra_stats.py` regenerate every table and figure in this report from the public dataset (code in Appendix B). Sampling and bootstrap use seed 42 and inference is deterministic, so a rerun gives identical predictions on the same software versions. Raw prediction files (one row per model × item × language) are kept as submitted evidence. Each batch skips models whose prediction file already exists, so further models can be added later without rerunning the others.");

T("h2", "9.6 Full study: implementation and runs");
P("The pipeline was extended in four ways for the full study. (1) `prepare.py` writes all 1,968 eligible items instead of the pilot sample. (2) `evaluate.py` runs on the GPU, loads multimodal checkpoints (Qwen3.5, Gemma 4) through their native class with text-only input, supports 4-bit loading, can keep named embedding tables in CPU RAM, and computes only the last position's scores (the only one the decision rule needs), which saves memory. (3) `evaluate_gguf.py` applies the same prompt, label tokens and token counts to models served by llama.cpp. (4) `evaluate_nli.py` evaluates NLI classifiers with the same output columns, so `analyze.py` treats every model alike. Two run scripts execute the models in batches, skip models whose results already exist and keep a log per model. `extra_stats.py` adds the corrected tests, contrasts and regressions.");
P("**Hardware:** AMD Ryzen 5 7640HS, NVIDIA RTX 4060 Laptop GPU (8 GB), 14 GB RAM, Arch Linux. **Software:** Python 3.14.7, PyTorch 2.14.1 (CUDA 13.0), Transformers 5.20.2, bitsandbytes, llama.cpp built with CUDA, pandas, SciPy, scikit-learn, statsmodels 0.15.");
const place = {
  "Qwen3.5-9B": `GPU (${META["Qwen3.5-9B"].peak_vram_gb} GB peak); 2 GB embedding table in RAM`,
  "gemma-4-E4B-it": `GPU (${META["gemma-4-E4B-it"].peak_vram_gb} GB peak); 5.6 GB per-layer table in RAM`,
  "gemma-4-26B-A4B-it-Q4_K_M": "GPU + experts in RAM", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": "GPU + experts in RAM",
  "Llama-3.1-8B-Instruct": `GPU (${META["Llama-3.1-8B-Instruct"].peak_vram_gb} GB peak)`,
  "LLama-3.1-KazLLM-1.0-8B": `GPU (${META["LLama-3.1-KazLLM-1.0-8B"].peak_vram_gb} GB peak)`,
  "Llama-3.1-Sherkala-8B-Chat": `GPU (${META["Llama-3.1-Sherkala-8B-Chat"].peak_vram_gb} GB peak)`,
};
const backend = (m) => (META[m].backend.startsWith("llama.cpp") ? "llama.cpp" : "Transformers") + ", " + (QUANT[META[m].quantization] || META[m].quantization || "fp32");
const RUN_ORDER = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct", "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "gemma-4-E4B-it", "gemma-4-26B-A4B-it-Q4_K_M", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", "Llama-3.1-8B-Instruct", "LLama-3.1-KazLLM-1.0-8B", "Llama-3.1-Sherkala-8B-Chat", "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "xlm-roberta-large-xnli"];
B.push({ t: "table", cols: [2.4, 1.6, 2.4, 0.9, 1.2],
  head: ["Model", "Backend / precision", "Memory placement", "Runtime", "Status"],
  rows: [
    ...RUN_ORDER.map((m) => [SHORT[m].replace(" (excluded)", ""), backend(m), place[m] || "GPU", runtime(META[m].seconds), m === "xlm-roberta-large-xnli" ? "done (excluded, Section 8)" : "done"]),
  ],
  caption: "Table 7. Full-study runs (1,968 items × 2 languages = 3,936 predictions per run). The two llama.cpp MoE runs took 12.5 of the 13.6 hours of compute." });

T("h2", "9.7 Full-study results");
const llmAbove = LLM.filter((m) => e(m).kk_vs_chance_p < 0.05);
const gaps = INC.map((m) => a(m).acc_gap);
P(`**Summary.** Every one of the 14 analysed models is less accurate on the Kazakh version of the same items. The gap ranges from ${f3(Math.min(...gaps))} (KazLLM) to ${f3(Math.max(...gaps))} (Qwen2.5-0.5B), every bootstrap interval excludes zero, and every Holm-corrected McNemar test is significant (p < .001), so **H1 is supported for all models**. ${llmAbove.length} of the 13 LLMs are above chance in Kazakh; SmolLM2 and Qwen2.5-0.5B are not, so their gaps are floor effects. The best Kazakh accuracy among LLMs is Qwen3.5-9B (${f3(a("Qwen3.5-9B").acc_kk)}), followed by Qwen3-30B-A3B (${f3(a("Qwen3-30B-A3B-Instruct-2507-Q4_K_M").acc_kk)}) and KazLLM (${f3(a("LLama-3.1-KazLLM-1.0-8B").acc_kk)}); the supervised mDeBERTa classifier is higher still (${f3(a("mDeBERTa-v3-base-xnli-multilingual-nli-2mil7").acc_kk)}). Table 8 gives the full results; Figures 3–6 show them.`);
B.push({ t: "table", cols: [1.75, 0.75, 0.7, 0.7, 1.35, 0.7, 0.7, 0.85, 0.8, 0.8, 0.7], size: 15, align: ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
  shade: (r) => r[0].includes("excluded") || r[0].startsWith("mDeBERTa"),
  head: ["Model", "Prec.", "Acc EN", "Acc KK", "Gap Δ\n[95% CI]", "F1 EN", "F1 KK", "b / c", "McNemar p (Holm)", "KK vs chance p", "R"],
  rows: mainRows,
  caption: "Table 8. Full-study results (n = 1,968 paired items per model). F1 = macro-F1; b / c = items correct only in English / only in Kazakh; McNemar p is one-sided and Holm-corrected over the 14 included models; KK vs chance is an exact one-sided binomial test against 1/3; R = above-chance retention (acc_KK − 1/3) / (acc_EN − 1/3). Shaded: NLI classifiers (different decision rule; XLM-R excluded, Section 8)." });
B.push({ t: "img", src: `${ROOT}/figures/fig1_accuracy.png`, w: 6.4, caption: "Figure 3. Full study: accuracy by language on all 1,968 items for the 14 analysed models; the dashed line is chance (0.333)." });
B.push({ t: "img", src: `${ROOT}/figures/fig4_retention.png`, w: 6.0, caption: "Figure 4. Above-chance retention R. A value of 1 would mean that the model's whole English advantage over chance survives in Kazakh; hollow bars are models whose Kazakh accuracy is not above chance." });

T("h3", "SQ1 / H1 – how large is the gap?");
P(`For the LLMs that are above chance in Kazakh, the gap lies between ${f3(a("LLama-3.1-KazLLM-1.0-8B").acc_gap)} and ${f3(a("Qwen2.5-1.5B-Instruct").acc_gap)}, i.e. between 4.5 and 28 percentage points, and the macro-F1 gap is similar (${f3(a("LLama-3.1-KazLLM-1.0-8B").macroF1_gap)} to ${f3(a("Qwen3.5-4B").macroF1_gap)}). Discordant pairs are one-sided for every model: for Qwen3.5-9B, ${a("Qwen3.5-9B").mcnemar_b_en_only} items are solved only in English and ${a("Qwen3.5-9B").mcnemar_c_kk_only} only in Kazakh. Above-chance retention (Figure 4) gives a scale-free answer to \"to what extent\": the strongest LLMs keep 75–89% of their English above-chance performance in Kazakh (KazLLM ${f2(e("LLama-3.1-KazLLM-1.0-8B").above_chance_retention)}, Gemma-4-E4B ${f2(e("gemma-4-E4B-it").above_chance_retention)}, Qwen3.5-9B ${f2(e("Qwen3.5-9B").above_chance_retention)}, Qwen3-30B-A3B ${f2(e("Qwen3-30B-A3B-Instruct-2507-Q4_K_M").above_chance_retention)}), the other above-chance models keep 33–73%, and the two smallest models keep almost nothing. Compared with the pilot, where the best model kept 23%, the full model set shows that zero-shot transfer to Kazakh is substantial for recent multilingual models but never complete.`);

T("h3", "SQ2 / H2 – does the gap depend on scale and family?");
B.push({ t: "table", cols: [1.6, 2.2, 0.8, 1.3, 0.8], align: ["l", "l", "r", "r", "r"],
  head: ["Factor", "Contrast (gap A − gap B)", "Δ gap", "95% CI", "p (Holm)"],
  rows: contrastRows,
  caption: "Table 9. Planned contrasts (Table 2): difference between the two models' EN–KK accuracy gaps, paired bootstrap over items (5,000 resamples), two-sided p with Holm correction over the 14 contrasts. A negative value means model A has the smaller gap." });
P(`**Scale within one family.** The Qwen3.5 ladder does not show a steadily shrinking gap: ${f3(a("Qwen3.5-0.8B").acc_gap)} (0.8B), ${f3(a("Qwen3.5-2B").acc_gap)} (2B), ${f3(a("Qwen3.5-4B").acc_gap)} (4B), ${f3(a("Qwen3.5-9B").acc_gap)} (9B), and the 0.8B and 9B gaps are practically identical (difference ${f3(gd("Qwen3.5-0.8B minus Qwen3.5-9B").point)}, ${ci(gd("Qwen3.5-0.8B minus Qwen3.5-9B").CI95)}). The raw gap is misleading here: Qwen3.5-0.8B almost never predicts \"neutral\" in either language (${a("Qwen3.5-0.8B").pred_dist_kk.neutral} of 1,968 in Kazakh), so its English accuracy is low (${f3(a("Qwen3.5-0.8B").acc_en)}) and little can be lost, while Qwen3.5-4B collapses towards \"neutral\" in Kazakh (${a("Qwen3.5-4B").pred_dist_kk.neutral} of 1,968 predictions, up from ${a("Qwen3.5-4B").pred_dist_en.neutral} in English; Figure 5). Retention, which accounts for the English level, rises with scale except at 4B (${f2(e("Qwen3.5-0.8B").above_chance_retention)} → ${f2(e("Qwen3.5-2B").above_chance_retention)} → ${f2(e("Qwen3.5-4B").above_chance_retention)} → ${f2(e("Qwen3.5-9B").above_chance_retention)}). Qwen3.5-4B is also the first ladder model run in 4-bit, so its dip may partly be a quantisation effect; this cannot be separated with the present runs. In Qwen2.5 the 0.5B and 1.5B gaps do not differ (${f3(gd("Qwen2.5-0.5B-Instruct minus Qwen2.5-1.5B-Instruct").point)}, Holm p = ${pv2(gd("Qwen2.5-0.5B-Instruct minus Qwen2.5-1.5B-Instruct").p_holm)}), but 0.5B is at chance in Kazakh.`);
const gt = G.total.coefficients["kk:log2_total"], ga = G.active.coefficients["kk:log2_active"], gk = G.total.coefficients.kk;
const gad = G.total_plus_adaptation.coefficients;
P(`**Language × scale model.** Across all 13 LLMs, the GEE logistic model (Table 10) finds a large Kazakh penalty (odds ratio ${f2(gk.OR)} at the reference size) and a significant positive language × log~2~(total parameters) interaction: each doubling of the parameter count multiplies the Kazakh-versus-English odds ratio by ${f2(gt.OR)} (95% CI ${f2(gt.CI95_OR[0])}–${f2(gt.CI95_OR[1])}, p ${pv(gt.p)}). With active parameters the interaction is stronger (${f2(ga.OR)} per doubling, p ${pv(ga.p)}). On the log-odds scale, therefore, larger models lose less in Kazakh, which supports H2. The accuracy gaps in Table 8 do not show this as clearly because the same log-odds penalty produces a smaller accuracy difference near chance than in the middle of the range, and because label collapse distorts single models. **H2 is therefore partly supported:** larger and more multilingual models retain more of their English ability, but scale alone does not guarantee a smaller gap in accuracy points.`);
B.push({ t: "table", cols: [3, 1, 1.4, 1], align: ["l", "r", "r", "r"],
  head: ["Term", "Odds ratio", "95% CI", "p"],
  rows: [
    ["*Model 1: total parameters*", "", "", ""],
    orRow("Kazakh (vs English), at 1B parameters", G.total.coefficients.kk),
    orRow("log~2~ total parameters (per doubling)", G.total.coefficients.log2_total),
    orRow("Kazakh × log~2~ total parameters", gt),
    ["*Model 2: active parameters*", "", "", ""],
    orRow("Kazakh (vs English), at 1B parameters", G.active.coefficients.kk),
    orRow("log~2~ active parameters (per doubling)", G.active.coefficients.log2_active),
    orRow("Kazakh × log~2~ active parameters", ga),
    ["*Model 3: total parameters + Kazakh adaptation*", "", "", ""],
    orRow("Kazakh × log~2~ total parameters", gad["kk:log2_total"]),
    orRow("Kazakh-adapted model (KazLLM, Sherkala)", gad.kz_adapted),
    orRow("Kazakh × Kazakh-adapted", gad["kk:kz_adapted"]),
  ],
  caption: `Table 10. GEE logistic models of correctness on all 13 LLMs (${G.total.n_rows.toLocaleString("en")} predictions, ${G.total.n_clusters.toLocaleString("en")} item clusters, exchangeable correlation); family terms (Gemma = reference, Llama, Qwen, SmolLM2) are included but not shown. An interaction odds ratio above 1 means the Kazakh penalty becomes smaller.` });
P(`**Family at fixed scale and total vs active parameters.** At roughly 4B, Gemma-4-E4B has a much smaller gap than Qwen3.5-4B (difference ${f3(gd("Qwen3.5-4B minus gemma-4-E4B-it").point)}, Holm p ${pv(gd("Qwen3.5-4B minus gemma-4-E4B-it").p_holm)}), so the training recipe matters at least as much as size; part of this difference is Qwen3.5-4B's neutral collapse. Within Gemma, the 26B-A4B mixture-of-experts model is more accurate in English (${f3(a("gemma-4-26B-A4B-it-Q4_K_M").acc_en)} vs ${f3(a("gemma-4-E4B-it").acc_en)}) but not in Kazakh (${f3(a("gemma-4-26B-A4B-it-Q4_K_M").acc_kk)} vs ${f3(a("gemma-4-E4B-it").acc_kk)}), so its gap is larger (difference ${f3(gd("gemma-4-E4B-it minus gemma-4-26B-A4B-it-Q4_K_M").point)}, p ${pv(gd("gemma-4-E4B-it minus gemma-4-26B-A4B-it-Q4_K_M").p_holm)}). Six times more stored parameters at similar active compute did not help Kazakh in this family. The two models also differ in backend and quantisation (NF4 vs Q4_K_M), which is a recorded confounder.`);
P(`**Kazakh adaptation.** This is the clearest result of the study. Further training Llama-3.1-8B on Kazakh (KazLLM) raises Kazakh accuracy from ${f3(a("Llama-3.1-8B-Instruct").acc_kk)} to ${f3(a("LLama-3.1-KazLLM-1.0-8B").acc_kk)} and also English accuracy from ${f3(a("Llama-3.1-8B-Instruct").acc_en)} to ${f3(a("LLama-3.1-KazLLM-1.0-8B").acc_en)}, and shrinks the gap from ${f3(a("Llama-3.1-8B-Instruct").acc_gap)} to ${f3(a("LLama-3.1-KazLLM-1.0-8B").acc_gap)} (difference ${f3(gd("Llama-3.1-8B-Instruct minus LLama-3.1-KazLLM-1.0-8B").point)}, ${ci(gd("Llama-3.1-8B-Instruct minus LLama-3.1-KazLLM-1.0-8B").CI95)}, Holm p ${pv(gd("Llama-3.1-8B-Instruct minus LLama-3.1-KazLLM-1.0-8B").p_holm)}); its retention (${f2(e("LLama-3.1-KazLLM-1.0-8B").above_chance_retention)}) is the highest of all LLMs. Sherkala, which adds an extended tokenizer, also has a smaller gap than the base (${f3(a("Llama-3.1-Sherkala-8B-Chat").acc_gap)}, Holm p = ${pv2(gd("Llama-3.1-8B-Instruct minus Llama-3.1-Sherkala-8B-Chat").p_holm)}), but its accuracy is lower than KazLLM's in both languages because it strongly prefers the label \"contradiction\" (${a("Llama-3.1-Sherkala-8B-Chat").pred_dist_kk.contradiction} of 1,968 Kazakh and ${a("Llama-3.1-Sherkala-8B-Chat").pred_dist_en.contradiction} English predictions; Kazakh entailment F1 ${f3(a("Llama-3.1-Sherkala-8B-Chat").perclassF1_kk.entailment)}). The Sherkala − KazLLM gap difference (${f3(gd("Llama-3.1-Sherkala-8B-Chat minus LLama-3.1-KazLLM-1.0-8B").point)}) is not significant after correction (p = ${pv2(gd("Llama-3.1-Sherkala-8B-Chat minus LLama-3.1-KazLLM-1.0-8B").p_holm)}). In the GEE (Table 10, Model 3), Kazakh adaptation multiplies the Kazakh-versus-English odds ratio by ${f2(gad["kk:kz_adapted"].OR)} (p ${pv(gad["kk:kz_adapted"].p)}), and the scale interaction becomes weaker (${f2(gad["kk:log2_total"].OR)}), i.e. part of what looked like a scale effect is targeted Kazakh training. Under our first-token decision rule, a Kazakh-friendly tokenizer did not add anything beyond the Kazakh training data.`);
P(`**Multilingual coverage and supervised reference.** SmolLM2 has a smaller raw gap than Qwen2.5-1.5B and Qwen3.5-2B (Table 9), but it is at chance in Kazakh and predicts \"contradiction\" for ${a("SmolLM2-1.7B-Instruct").pred_dist_kk.contradiction} of 1,968 Kazakh items; its small gap is a floor effect, not better transfer. The supervised mDeBERTa classifier, which has never seen Kazakh NLI data, has a smaller gap than the best LLM (${f3(a("mDeBERTa-v3-base-xnli-multilingual-nli-2mil7").acc_gap)} vs ${f3(a("Qwen3.5-9B").acc_gap)}; difference ${f3(gd("Qwen3.5-9B minus mDeBERTa-v3-base-xnli-multilingual-nli-2mil7").point)}, p ${pv(gd("Qwen3.5-9B minus mDeBERTa-v3-base-xnli-multilingual-nli-2mil7").p_holm)}) and the highest Kazakh accuracy. A 0.28B task-specific encoder still transfers better than zero-shot prompting of a 9B LLM.`);

T("h3", "O4 – error patterns and morphology");
B.push({ t: "img", src: `${ROOT}/figures/fig2_confusion_kk.png`, w: 6.2, caption: "Figure 5. Full study: confusion matrices on the Kazakh items (rows = gold, columns = predicted)." });
B.push({ t: "img", src: `${ROOT}/figures/fig3_gap_vs_token_premium.png`, w: 5.6, caption: "Figure 6. Accuracy gap against the Kazakh token premium of each LLM's tokenizer (Table 3). Hollow markers: Kazakh accuracy not above chance (binomial p ≥ .05), so the gap there is a floor effect." });
const zf = O4.coefficients.z_fert, zw = O4.coefficients.z_wlen, ge = O4.coefficients["C(gold)[T.entailment]"];
P(`**Tokenizer cost across models.** Figure 6 shows no simple relation between how much a tokenizer fragments Kazakh and the gap. Qwen3-30B-A3B uses the old Qwen tokenizer (premium 2.85) yet has the same gap as Qwen3.5-9B (premium 2.10; difference ${f3(gd("Qwen3-30B-A3B-Instruct-2507-Q4_K_M minus Qwen3.5-9B").point)}, p = ${pv2(gd("Qwen3-30B-A3B-Instruct-2507-Q4_K_M minus Qwen3.5-9B").p_holm)}), and in the controlled Llama-3.1 triple the model with the cheapest Kazakh tokenizer (Sherkala, 1.14) does not beat KazLLM (2.90), which shares the base model's tokenizer. Token premium alone does not explain the gap.`);
P(`**Item-level morphology.** Among the ${O4.n_rows.toLocaleString("en")} Kazakh predictions on items the same model solved in English, the pooled logistic regression finds no effect of fertility (odds ratio ${f2(zf.OR)} per standard deviation, p = ${pv2(zf.p)}) or mean word length (${f2(zw.OR)}, p = ${pv2(zw.p)}). The per-model Mann–Whitney tests agree: no model shows a fertility difference between flips and kept items, and the two nominal word-length differences (Qwen3-30B-A3B p = .012, Sherkala p = .020) do not survive correction for 14 tests. As in the pilot, item-to-item variation in these simple proxies is small next to the language-wide difference, so they cannot explain *which* items fail.`);
P(`**Direction of errors.** The gold label matters much more: an English-solved entailment item is far more likely to fail in Kazakh than a contradiction item (odds ratio of staying correct ${f2(ge.OR)}, p ${pv(ge.p)}). Across the 13 LLMs, 54% of all flips move to \"neutral\" (entailment → neutral 29%, contradiction → neutral 25%), and the share of \"neutral\" predictions rises in Kazakh for Qwen3.5-4B (54% → 83%), Gemma-4-26B-A4B (50% → 64%), Llama-3.1-8B (54% → 67%) and Gemma-4-E4B (45% → 56%). When the Kazakh input is understood less well, these models stop committing to a relation and answer \"neutral\"; models without a neutral habit fall back to their own default label instead (SmolLM2, Qwen3.5-0.8B and Sherkala to \"contradiction\").`);
P(`**Examples.** Appendix D lists the first three flips of the best LLM in Kazakh (${SHORT[E.examples_model]}). In item 19 the Kazakh premise says Fannie Flono *wants to* talk about her childhood (*сөйлескісі келеді*) where the English says she *is going to*, a translation shift that makes \"neutral\" defensible. In item 74 the translation appears faithful and the model fails on the concessive *-са да* (\"even though\") construction; such cases are what the planned manual coding (Step 7) will count.`);

T("h2", "9.8 Limitations and remaining work");
B.push({ t: "list", items: [
  "**Model coverage.** The two planned models that were not run (Qwen3.5-35B-A3B and gpt-oss-20b, Section 4.1) leave four questions open: whether the Qwen3.5 gap keeps changing beyond 9B, whether Qwen3 and Qwen3.5 differ at the same mixture-of-experts size, how a third family behaves at about 4B active parameters, and whether letting a model reason recovers Kazakh accuracy. The largest model analysed has 30B total and 3.3B active parameters.",
  "**Manual audit.** The 50-pair adequacy rating and the coding of flips by a Kazakh-speaking team member are not yet done, so the share of the gap caused by translation is unknown; the examples above show that some of it is.",
  "**Prompt sensitivity.** Only one prompt template and one label order were used. Label preference effects (Sherkala, SmolLM2, Qwen3.5-0.8B/4B) suggest that some models would rank differently under paraphrased prompts; the dev-split check remains to be run.",
  "**Precision confound.** Precision differs between models (bf16, NF4, Q4_K_M) and is confounded with size in the Qwen3.5 ladder (4-bit from 4B up) and with backend in the Gemma pair.",
  "**Contamination.** English XNLI items are public; one model was excluded for training on them, and English scores of the others may also be inflated.",
] });

T("h1", "10. Methodology Alignment Table");
B.push({ t: "table", cols: [1.2, 1.3, 1.3, 1.6, 1.6, 1.6], size: 15,
  head: ["Research objective", "Data / evidence needed", "Source / sample / dataset", "Collection / instrument", "Analysis method", "Evidence for RQ (result)"],
  rows: [
    ["O1 Prepare parallel EN–KK NLI dataset", "Clean, paired EN/KK items with gold labels; quality statistics", "Kardeş-NLU Kazakh XNLI test (2,000 items; XNLI origin)", "prepare.py: NFC, script/length/untranslated checks, exclusion, stratified pilot; manual adequacy audit (50 pairs, pending)", "Descriptive statistics: exclusions per rule, label balance, length ratios, adequacy ratings", "1,968 balanced items; 32 excluded by the script rule. Shows the KK condition is a valid parallel of EN"],
    ["O2 Evaluate LLMs under identical zero-shot prompts", "Item-level predictions per model × language", "All 1,968 eligible items; 15 purposively selected open models (14 analysed)", "evaluate.py / evaluate_gguf.py / evaluate_nli.py: fixed English prompt, chat template, constrained deterministic decoding; CSV logs", "Per-language accuracy, macro-F1, per-class F1, prediction distribution", "55,104 predictions; Table 8, Figure 3"],
    ["O3 Quantify EN–KK gap (SQ1, H1)", "Paired correctness for each item in both languages", "Same items and models as O2", "Prediction logs from O2", "Δ accuracy and Δ F1; paired bootstrap 95% CI; exact one-sided McNemar test with Holm correction; retention R", "Gap 0.045–0.291, all p < .001: H1 supported; retention up to 0.89"],
    ["O4 Error patterns and morphology", "Fertility and tokens per pair, Kazakh word length, flip items, error direction", "Items correct in EN but wrong in KK, plus EN-and-KK-correct controls; tokenizer contrasts (Table 2)", "Token counts logged per prediction (Section 5.1); manual coding of flips (pending)", "Mann–Whitney U, logistic regression with item-clustered SEs, gap vs token premium, confusion matrices", "No item-level fertility or word-length effect; token premium does not explain the gap; errors drift to \"neutral\""],
    ["O5 Gap across family and scale (SQ2, H2)", "Gap per model with parameter count (total and active), family, adaptation", "Qwen3.5 0.8B–9B; Qwen2.5; Qwen3-30B-A3B; Gemma 4; Llama-3.1-8B, KazLLM, Sherkala; SmolLM2", "Same harness, same items and settings for all models", "GEE logistic model (language × log-params + family, item clusters); bootstrap CIs for planned contrasts with Holm correction", "Kazakh penalty shrinks with scale on the log-odds scale (H2 partly supported); Kazakh adaptation is the strongest factor"],
  ],
  caption: "Table 11. Each objective linked to its evidence, source, instrument, analysis and result." });

T("h1", "11. References");
[
  "Allal, L. B., Lozhkov, A., Bakouch, E., Blázquez, G. M., Penedo, G., Tunstall, L., Marafioti, A., Kydlíček, H., Lajarín, A. P., Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon, M., … Wolf, T. (2025). SmolLM2: When smol goes big – Data-centric training of a small language model. *arXiv*. https://arxiv.org/abs/2502.02737",
  "Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., et al. (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. https://arxiv.org/abs/2005.14165",
  "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440–8451). https://doi.org/10.18653/v1/2020.acl-main.747",
  "Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S., Schwenk, H., & Stoyanov, V. (2018). XNLI: Evaluating cross-lingual sentence representations. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 2475–2485). Association for Computational Linguistics. https://doi.org/10.18653/v1/D18-1269",
  "Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://arxiv.org/abs/2305.14314",
  "Ding, Y., Han, X., Yang, J., Wang, T., Bi, Z., Song, X., Hao, J., Song, J., Ge, E., Peng, B., Liu, Z., Liang, C. X., Zhang, Y., Liu, M., Xu, J., Huang, B., Mo, Y., Yu, Z., Qiao, J., … Ma, Y. (2026). Cross-lingual transfer learning in large language models: Multilingual representations and low-resource adaptation. *TechRxiv*. https://doi.org/10.36227/techrxiv.176799777.79169627/v1",
  "Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1383–1392). Association for Computational Linguistics. https://doi.org/10.18653/v1/P18-1128",
  "Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.",
  "Gerganov, G., et al. (2023–2026). *llama.cpp: LLM inference in C/C++* [Computer software]. GitHub. https://github.com/ggml-org/llama.cpp",
  "Google DeepMind. (2026). *Gemma 4 model card*. Hugging Face. https://huggingface.co/google/gemma-4-E4B-it",
  "Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., et al. (2024). The Llama 3 herd of models. *arXiv*. https://arxiv.org/abs/2407.21783",
  "He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. In *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2111.09543",
  "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.",
  "ISSAI. (2024). *LLama-3.1-KazLLM-1.0-8B* [Model card]. Institute of Smart Systems and Artificial Intelligence, Nazarbayev University. Hugging Face. https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B",
  "Koto, F., Joshi, R., Mukhituly, N., Wang, Y., Xie, Z., Pal, R., et al. (2025). Sherkala-Chat: Building a state-of-the-art LLM for Kazakh in a moderately resourced setting. *arXiv*. https://arxiv.org/abs/2503.01493",
  "Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. *Political Analysis, 32*(1), 84–100. https://doi.org/10.1017/pan.2023.20",
  "Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika, 73*(1), 13–22. https://doi.org/10.1093/biomet/73.1.13",
  "Maxutov, A., Myrzakhmet, A., & Braslavski, P. (2024). Do LLMs speak Kazakh? A pilot evaluation of seven models. In *Proceedings of the First Workshop on Natural Language Processing for Turkic Languages (SIGTURK 2024)* (pp. 81–91). Association for Computational Linguistics. https://aclanthology.org/2024.sigturk-1.8/",
  "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996",
  "OpenAI. (2025). gpt-oss-120b & gpt-oss-20b model card. *arXiv*. https://arxiv.org/abs/2508.10925",
  "Petrov, A., La Malfa, E., Torr, P., & Bibi, A. (2023). Language model tokenizers introduce unfairness between languages. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://arxiv.org/abs/2305.15425",
  "Qwen Team. (2025). Qwen3 technical report. *arXiv*. https://arxiv.org/abs/2505.09388",
  "Qwen Team. (2026). *Qwen3.5 model cards*. Hugging Face. https://huggingface.co/Qwen/Qwen3.5-9B",
  "Qwen Team: Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., et al. (2024). Qwen2.5 technical report. *arXiv*. https://arxiv.org/abs/2412.15115",
  "Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How good is your tokenizer? On the monolingual performance of multilingual language models. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*. Association for Computational Linguistics. https://aclanthology.org/2021.acl-long.243",
  "Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., et al. (2022). Multitask prompted training enables zero-shot task generalization. In *International Conference on Learning Representations (ICLR 2022)*. https://arxiv.org/abs/2110.08207",
  "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In *Proceedings of the 9th Python in Science Conference* (pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011",
  "Senel, L. K., Ebing, B., Baghirova, K., Schuetze, H., & Glavaš, G. (2024). Kardeş-NLU: Transfer to low-resource languages with big brother's help – A benchmark and evaluation for Turkic languages. In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1672–1688). Association for Computational Linguistics. https://aclanthology.org/2024.eacl-long.100",
  "Togmanov, M., Mukhituly, N., Turmakhan, D., Mansurov, J., Goloburda, M., Sakip, A., Xie, Z., Wang, Y., Syzdykov, B., Laiyk, N., Aji, A. F., Kochmar, E., Nakov, P., & Koto, F. (2025). KazMMLU: Evaluating language models on Kazakh, Russian, and regional knowledge of Kazakhstan. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 14403–14416). Association for Computational Linguistics. https://aclanthology.org/2025.acl-long.701/",
  "Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 38–45). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-demos.6",
].forEach((r) => B.push({ t: "ref", text: r }));

B.push({ t: "pagebreak" });
T("h1", "12. Appendix");
T("h2", "Appendix A. Pipeline architecture");
B.push({ t: "code", size: 15, text: `Kardeş-NLU xnli.test.kk.csv (EN + KK columns, gold label)
      │
      ▼ prepare.py    NFC + whitespace │ script/length/untranslated flags │ exclusion
data/eval_set.csv (1,968 items; pilot: pilot/data/pilot_sample.csv, 300) + results/data_summary.json
      │
      ▼ for each item × {EN, KK}: fixed English prompt → model's chat template → 1 forward pass
      │    evaluate.py      Transformers (bf16 / 4-bit NF4, optional embedding tables in RAM)
      │    evaluate_gguf.py llama.cpp server (Q4_K_M MoE), token ids via /tokenize
      │    evaluate_nli.py  NLI classifiers (3-way head, label order from config)
      │ → scores of the first tokens of 'entailment' / 'neutral' / 'contradiction' → argmax
      │ → token counts of premise + hypothesis with the model's own tokenizer (Section 5.1)
      │    (run_batch1.sh, run_batch2.sh: run every model, skip finished ones, log each run)
results/preds_<model>.csv (item, lang, gold, pred, probs, tokens, fertility) + results/meta_<model>.json
      │
      ▼ analyze.py    accuracy, macro-F1, confusion │ paired bootstrap CI │ exact McNemar │ flips
results/summary_table.csv, analysis.json, flips_en_correct_kk_wrong.csv, figures/fig1-3
      │
      ▼ extra_stats.py  KK vs chance │ Holm-corrected McNemar │ retention │ planned contrasts
      │                 O4 logistic regression │ O5 GEE language × scale │ figures/fig4
results/extra_stats.json` });

T("h2", "Appendix B. Source code (full-study version)");
P("The listings below are the scripts that produced every number and figure in Sections 9.6–9.8. The pilot versions that produced Sections 9.1–9.4 differ only in the input file, CPU execution and the three pilot models, and are archived unchanged in `pilot/`. `run_batch2.sh` still contains the entries for Qwen3.5-35B-A3B and gpt-oss-20b, and `evaluate_gguf.py` the code path for gpt-oss; these were prepared but not run (Section 4.1).");
[
  ["B.1 prepare.py", "prepare.py"], ["B.2 evaluate.py", "evaluate.py"], ["B.3 evaluate_gguf.py", "evaluate_gguf.py"],
  ["B.4 evaluate_nli.py", "evaluate_nli.py"], ["B.5 analyze.py and analyze_names.py", "analyze.py", "analyze_names.py"],
  ["B.6 extra_stats.py", "extra_stats.py"], ["B.7 run_batch1.sh", "run_batch1.sh"], ["B.8 run_batch2.sh and llama_server_lib.sh", "run_batch2.sh", "llama_server_lib.sh"],
].forEach(([h, ...files]) => {
  T("h3", h);
  files.forEach((f) => B.push({ t: "code", size: 14, text: (files.length > 1 ? `# ===== ${f} =====\n` : "") + src(f) }));
});

T("h2", "Appendix C. Run outputs");
T("h3", "C.1 Data preparation output (results/data_summary.json)");
B.push({ t: "code", text: JSON.stringify(DS, null, 2) });
T("h3", "C.2 Run metadata examples (results/meta_<model>.json)");
const metaShow = (m) => { const x = { ...META[m] }; if (x.model.includes("/snapshots/")) x.model = ".../" + x.model.split("/").pop(); return JSON.stringify(x, null, 2); };
B.push({ t: "code", size: 14, text: metaShow("Qwen3.5-9B") });
B.push({ t: "code", size: 14, text: metaShow("gemma-4-26B-A4B-it-Q4_K_M") });
B.push({ t: "code", size: 14, text: metaShow("Llama-3.1-Sherkala-8B-Chat") });

T("h2", "Appendix D. Example items");
T("h3", "D.1 Pilot (Qwen2.5-1.5B correct in English, wrong in Kazakh)");
const ex = (id, gold, model, pred, pe, he, pk, hk) => {
  P(`**Item ${id}** — gold: ${gold}; ${model} correct in English, predicted ${pred} in Kazakh.`);
  P(`EN premise: ${pe}`); P(`EN hypothesis: ${he}`); P(`KK premise: ${pk}`); P(`KK hypothesis: ${hk}`);
};
ex(3380, "contradiction", "Qwen2.5-1.5B", "entailment",
  "I do not believe the possible recon value outweighs the risk of possible program termination when the stakes are raised by the Taliban parading a charred Predator in front of CNN, he wrote.",
  "The Taliban was shown fleeing from the Predator on CNN.",
  "Талибандар CNN алдында күйіп кеткен жыртқышты шеруге көтерген кезде, ықтимал барлау құны бағдарламаны тоқтату қаупінен асып түсетініне сенбеймін, деп жазды ол.",
  "CNN арнасында Талибанның Жыртқыштан қашып бара жатқаны көрсетілді.");
ex(2788, "contradiction", "Qwen2.5-1.5B", "entailment",
  "The employer and employee contributions are generally calculated in the same way",
  "There are two different methods of calculating the employer and employee contributions.",
  "Жұмыс беруші мен қызметкердің үлестері әдетте бірдей есептеледі",
  "Жұмыс беруші мен қызметкердің үлестерін есептеудің екі түрлі әдісі бар.");
ex(504, "neutral", "Qwen2.5-1.5B", "entailment", "yeah it's real nice has it been raining", "I do not mind the rain.",
  "иә, жаңбыр жауғаны өте жақсы", "Жаңбырға қарсы емеспін.");
T("h3", `D.2 Full study (${SHORT[E.examples_model]}, first three flips by item order)`);
E.examples.forEach((x) => ex(x.xnli_id, x.gold, SHORT[E.examples_model], x.pred_kk, x.premise_en, x.hypothesis_en, x.premise_kk, x.hypothesis_kk));

(async () => {
  await writeDocx(B, OUT + ".docx");
  await writePdf(B, OUT + ".pdf");
  console.log("wrote", OUT + ".docx", OUT + ".pdf");
})();
