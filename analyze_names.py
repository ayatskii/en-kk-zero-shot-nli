"""Short model names for figures (shared by analyze.py and extra_stats.py)."""


def short(name):
    for a, b in [("-Instruct-2507", ""), ("-Instruct", ""), ("-it", ""), ("-Q4_K_M", ""), ("-MXFP4", ""),
                 ("-Chat", ""), ("LLama-3.1-KazLLM-1.0-8B", "KazLLM-8B"), ("Llama-3.1-Sherkala-8B", "Sherkala-8B"),
                 ("mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "mDeBERTa (NLI clf.)"), ("gemma", "Gemma")]:
        name = name.replace(a, b)
    return name
