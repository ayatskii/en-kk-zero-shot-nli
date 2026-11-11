# Assignment 3 pilot – EN→KK zero-shot NLI (Baurzhanova, Mergenbay, SE-2416)
Data: Kardeş-NLU data/kazakh/xnli.test.kk.csv (github.com/lksenel/Kardes-NLU, commit 48eea77)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas scipy scikit-learn matplotlib
python prepare.py
python evaluate.py Qwen/Qwen2.5-0.5B-Instruct
python evaluate.py Qwen/Qwen2.5-1.5B-Instruct
python evaluate.py HuggingFaceTB/SmolLM2-1.7B-Instruct
python analyze.py
python extra_stats.py

Read the premise and the hypothesis.
Premise: Жұмыс беруші мен қызметкердің үлестері әдетте бірдей есептеледі
Hypothesis: Жұмыс беруші мен қызметкердің үлестерін есептеудің екі түрлі әдісі бар.
Does the premise entail the hypothesis, contradict it, or is it neutral?
Answer with exactly one word: entailment, neutral, or contradiction.