"""Step 2a - Zero-shot NLI evaluation of one Hugging Face model on parallel EN and KK items
(batch 1: models that fit in GPU memory, optionally 4-/8-bit quantised with bitsandbytes).

Usage: python evaluate.py <hf_model_id> [--quant none|4bit|8bit] [--n N] [--device cuda|cpu]
Decision rule: constrained label scoring. One forward pass per item; the predicted
label is the one whose first answer token has the highest next-token logit after the
assistant turn starts. No sampling -> fully deterministic.
The instruction is always in English; only premise/hypothesis language changes,
so the language of the items is the only manipulated condition.
Hybrid "thinking" models (Qwen3/Qwen3.5) are run with enable_thinking=False, so the
assistant turn starts directly with the answer.
"""
import argparse, time, json, platform
import pandas as pd, torch, transformers
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForImageTextToText, BitsAndBytesConfig)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
ap.add_argument("--n", type=int, default=None, help="only the first N items (smoke test)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--cpu-modules", default="",
                help="comma-separated module names kept in CPU RAM (large embedding tables)")
args = ap.parse_args()
CPU_MODULES = [m for m in args.cpu_modules.split(",") if m]
MODEL = args.model
LABELS = ["entailment", "neutral", "contradiction"]
torch.manual_seed(0)
if args.device == "cpu":
    torch.set_num_threads(2)

SYSTEM = "You are a careful annotator for a natural language inference task."
TEMPLATE = ("Read the premise and the hypothesis.\n"
            "Premise: {p}\nHypothesis: {h}\n\n"
            "Does the premise entail the hypothesis, contradict it, or is it neutral? "
            "Answer with exactly one word: entailment, neutral, or contradiction.")

tok = AutoTokenizer.from_pretrained(MODEL)
# Sherkala's repo ships no chat template; this is the one its model card sets in code,
# minus a stray space that would put a token before BOS
SHERKALA_TEMPLATE = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role']+'<|end_header_id|>\n\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}")
if not tok.chat_template and "Sherkala" in MODEL:
    tok.chat_template = SHERKALA_TEMPLATE
assert tok.chat_template, f"{MODEL} has no chat template - use an instruct/chat checkpoint"

# Multimodal checkpoints (Qwen3.5, Gemma 4) are loaded with their native class so the
# weight names match; text-only input still returns next-token logits.
cfg = AutoConfig.from_pretrained(MODEL)
multimodal = any("ConditionalGeneration" in a for a in (cfg.architectures or []))
Auto = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
quant = None
# bitsandbytes refuses CPU-placed modules unless this flag is set; they stay unquantised
offload = {"llm_int8_enable_fp32_cpu_offload": True} if CPU_MODULES else {}
if args.quant == "4bit":
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                               **offload)
elif args.quant == "8bit":
    quant = BitsAndBytesConfig(load_in_8bit=True, **offload)

def split_device_map(names):
    """Put the named modules on the CPU and everything else on the GPU. 4-bit only shrinks
    the transformer layers; embedding tables stay bf16 (Qwen3.5-9B input table 2 GB,
    Gemma-4-E4B per-layer table 5.6 GB) and are only looked up, so CPU RAM costs ~nothing."""
    from accelerate import init_empty_weights
    with init_empty_weights():
        skeleton = Auto.from_config(cfg)
    dm = {}
    def walk(mod, prefix):
        for n, _ in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
            dm[prefix + n] = 0
        for n, child in mod.named_children():
            full = prefix + n
            if full in names:
                dm[full] = "cpu"
            elif any(x.startswith(full + ".") for x in names):
                walk(child, full + ".")
            else:
                dm[full] = 0
    walk(skeleton, "")
    missing = [x for x in names if x not in dm]
    assert not missing, f"no such modules: {missing}"
    return dm

device_map = split_device_map(CPU_MODULES) if CPU_MODULES else args.device
model = Auto.from_pretrained(MODEL, dtype=torch.bfloat16, quantization_config=quant,
                             device_map=device_map)
model.eval()

def run_on_cpu(mod):
    """accelerate treats CPU-mapped modules as storage and copies their weights to the GPU
    on every call (a 5 GB spike for Gemma). Instead: put the stored weights on the module,
    drop accelerate's hooks, run the module on the CPU and hand its output to the GPU."""
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import set_module_tensor_to_device
    for sub in mod.modules():
        wm = getattr(getattr(sub, "_hf_hook", None), "weights_map", None)
        if wm is None:
            continue
        for n, t in list(sub.named_parameters(recurse=False)) + list(sub.named_buffers(recurse=False)):
            # non-persistent buffers (e.g. embed_scale) are not in the map but already real
            value = wm[n] if f"{wm.prefix}{n}" in wm.dataset.all_keys else t
            set_module_tensor_to_device(sub, n, "cpu", value=value)
    remove_hook_from_module(mod, recurse=True)
    to = lambda x, d: x.to(d) if torch.is_tensor(x) else x
    mod.register_forward_pre_hook(lambda m, a, kw: (tuple(to(x, "cpu") for x in a),
                                                    {k: to(v, "cpu") for k, v in kw.items()}),
                                  with_kwargs=True)
    mod.register_forward_hook(lambda m, i, o: to(o, args.device))

for name in CPU_MODULES:
    run_on_cpu(model.get_submodule(name))

# First token id of each label as the assistant would start the answer
label_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LABELS]
assert len(set(label_ids)) == 3, f"label first-tokens collide: {label_ids}"

def build(p, h):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(p=p, h=h)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)

df = pd.read_csv("data/eval_set.csv")
if args.n: df = df.head(args.n)
print("prompt tail:", repr(build("P", "H")[-120:]), flush=True)
rows = []
t0 = time.time()
for i, r in df.iterrows():
    for lang in ["en", "kk"]:
        p, h = r[f"premise_{lang}"], r[f"hypothesis_{lang}"]
        # the chat template already contains BOS where the model needs one
        enc = tok(build(p, h), return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():  # logits_to_keep=1: only the last position (saves VRAM)
            logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
        scores = logits[label_ids]
        probs = torch.softmax(scores, -1).tolist()
        pred = LABELS[int(scores.argmax())]
        n_tok_pair = len(tok.encode(p + " " + h, add_special_tokens=False))
        n_words = len((p + " " + h).split())
        rows.append({"xnli_id": r.xnli_id, "lang": lang, "gold": r.label, "pred": pred,
                     "correct": pred == r.label, "p_ent": probs[0], "p_neu": probs[1],
                     "p_con": probs[2], "pair_tokens": n_tok_pair, "pair_words": n_words,
                     "fertility": n_tok_pair / n_words})
    if (i + 1) % 100 == 0:
        print(f"{MODEL}: {i+1}/{len(df)} items, {time.time()-t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
safe = MODEL.split("/")[-1]
out.to_csv(f"results/preds_{safe}.csv", index=False)
meta = {"model": MODEL, "backend": "transformers", "items": len(df),
        "seconds": round(time.time() - t0, 1), "dtype": "bfloat16",
        "quantization": args.quant if args.quant != "none" else "none (bf16)",
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "cpu_offloaded_modules": CPU_MODULES,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if args.device == "cuda" else None,
        "decoding": "constrained first-token argmax (deterministic)",
        "label_first_token_ids": dict(zip(LABELS, label_ids)),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "cpu_threads": torch.get_num_threads()}
json.dump(meta, open(f"results/meta_{safe}.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
