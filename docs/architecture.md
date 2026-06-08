# Completus: a highly efficient LoRA Code Assistant

---

## Phases

- Phase 1: `dataset`
- Phase 2: `data cleaning + EDA`
- Phase 3&4: `baseline, LoRA, GGUF, eval`

## File Structure

```
completus/
├── data/
│   ├── processed/
│   │   ├── python_train.jsonl
│   │   └── go_train.jsonl
│   └── eval/
├── src/
│   ├── prepare_data.py       # Phase 1 - dataset
│   ├── eda.py                # Phase 2 - cleaning + plots
│   ├── evaluate.py           # Phase 3+4 - baseline + final eval
│   ├── train.py              # Phase 4 - LoRA training
│   └── merge_convert.py      # Phase 4 - merge + GGUF
├── adapters/
│   ├── python_lora/
│   └── go_lora/
├── merged/
│   ├── python_merged/
│   └── go_merged/
├── gguf/
│   ├── python_coder.gguf
│   └── go_coder.gguf
├── ollama/
│   ├── Modelfile_python
│   └── Modelfile_go
├── results/
│   ├── baseline.jsonl
│   ├── python_lora.jsonl
│   ├── go_lora.jsonl
│   └── eda_*.png
├── llama.cpp/                # clone separately: github.com/ggerganov/llama.cpp
├── requirements.txt
└── README.md
```

---

## requirements.txt

```
unsloth
trl
transformers
datasets
torch
matplotlib
human-eval
```

---

## Full Pipeline (run top to bottom)

```bash
# Phase 1
python src/prepare_data.py

# Phase 2
python src/eda.py

# Phase 3 - baseline eval
python src/evaluate.py Qwen/Qwen2.5-Coder-1.5B results/baseline.jsonl

# Phase 4 - train
python src/train.py python
python src/train.py go

# Phase 4 - merge + convert to GGUF
python src/merge_convert.py python
python src/merge_convert.py go

cd llama.cpp
python convert_hf_to_gguf.py ../merged/python_merged --outtype q4_k_m --outfile ../gguf/python_coder.gguf
python convert_hf_to_gguf.py ../merged/go_merged   --outtype q4_k_m --outfile ../gguf/go_coder.gguf
cd ..

# Phase 4 - register in Ollama
ollama create python-coder -f ollama/Modelfile_python
ollama create go-coder     -f ollama/Modelfile_go

# Phase 4 - eval fine-tuned models
python src/evaluate.py ./merged/python_merged results/python_lora.jsonl
python src/evaluate.py ./merged/go_merged     results/go_lora.jsonl

# Score everything
evaluate_functional_correctness results/baseline.jsonl
evaluate_functional_correctness results/python_lora.jsonl
```

---

## src/prepare_data.py (Phase 1)

```python
from datasets import load_dataset

def make_chatml(ex, lang):
    return {"text": (
        f"<|im_start|>user\nWrite a {lang} function: {ex['func_documentation_string']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{ex['func_code_string']}\n<|im_end|>"
    )}

def prepare(lang, n=5000):
    ds = load_dataset("code_search_net", lang, split="train")
    ds = ds.filter(lambda x:
        len(x["func_documentation_string"]) > 20 and
        len(x["func_code_string"]) > 50 and
        len(x["func_code_string"]) // 4 <= 2048 and
        "TODO" not in x["func_code_string"] and
        x["func_code_string"].count("\n") > 2
    )
    ds = ds.select(range(min(n, len(ds)))).map(lambda x: make_chatml(x, lang))
    ds.to_json(f"data/processed/{lang}_train.jsonl")
    print(f"{lang}: {len(ds)} samples saved")

prepare("python", n=5000)
prepare("go",     n=3000)
```

> Pitfalls to mention in paper: (1) data leakage - HumanEval problems exist in The Stack, filter any file
> containing benchmark signatures before training. (2) Class imbalance - Python has ~10x more samples
> than Go. Both are subsampled to fixed sizes (5000 / 3000), note this as a limitation.

---

## src/eda.py (Phase 2)

```python
import matplotlib.pyplot as plt, collections, re
from datasets import load_dataset

def valid(x):
    return (len(x["func_documentation_string"]) > 20 and
            len(x["func_code_string"]) > 50 and
            len(x["func_code_string"]) // 4 <= 2048 and
            "TODO" not in x["func_code_string"] and
            x["func_code_string"].count("\n") > 2)

for lang in ["python", "go"]:
    raw   = load_dataset("code_search_net", lang, split="train")
    clean = raw.filter(valid)
    print(f"{lang}: {len(raw)} raw -> {len(clean)} clean ({100*len(clean)/len(raw):.1f}% kept)")

    lengths_raw   = [len(x["func_code_string"]) // 4 for x in raw]
    lengths_clean = [len(x["func_code_string"]) // 4 for x in clean]

    counter = collections.Counter()
    for x in clean:
        counter.update(re.findall(r'^(?:import|from)\s+(\S+)', x["func_code_string"], re.MULTILINE))
    labels, counts = zip(*counter.most_common(15))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(lengths_raw,   bins=50, color="#B4B2A9", label="raw")
    axes[0].hist(lengths_clean, bins=50, color="#5DCAA5", label="clean", alpha=0.7)
    axes[0].axvline(2048, color="red", linestyle="--", label="2048 token cutoff")
    axes[0].set_title(f"{lang} - token lengths"); axes[0].legend()

    axes[1].bar(labels, counts, color="#5DCAA5")
    axes[1].set_title(f"{lang} - top imports"); axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(["raw", "clean"], [len(raw), len(clean)], color=["#B4B2A9", "#5DCAA5"])
    axes[2].set_title(f"{lang} - sample count before/after")

    plt.tight_layout()
    plt.savefig(f"results/eda_{lang}.png", dpi=150)
    print(f"Saved results/eda_{lang}.png")
```

> Deliverable: run this, save the 2 PNG files, embed them in your Jupyter notebook /
> preprocessing section. Include the printed before/after table in the paper.

---

## src/evaluate.py (Phase 3 + Phase 4)

```python
import json, torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def eval_model(model_path, out_file, n=30):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto")
    mdl.eval()
    problems = load_dataset("openai_humaneval", split="test").select(range(n))
    results  = []
    for p in problems:
        inp = tok(p["prompt"], return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=256, temperature=0.1,
                               do_sample=True, pad_token_id=tok.eos_token_id)
        results.append({"task_id": p["task_id"],
                         "completion": tok.decode(out[0][inp["input_ids"].shape[1]:],
                                                  skip_special_tokens=True)})
    with open(out_file, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    print(f"Done: {len(results)} completions -> {out_file}")
    print(f"Score: evaluate_functional_correctness {out_file}")

# python src/evaluate.py <model_path> <output_file>
eval_model(sys.argv[1], sys.argv[2])
```

> Run three times: once with base model (Phase 3), once each with merged LoRA models (Phase 4).
> The `evaluate_functional_correctness` CLI from the `human-eval` package prints pass@1 directly.

---

## src/train.py (Phase 4)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import sys

LANG = sys.argv[1]  # "python" or "go"
SEED = 42

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B", max_seq_length=2048, load_in_4bit=True)

model = FastLanguageModel.get_peft_model(
    model, r=8,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0.05, bias="none",
    use_gradient_checkpointing="unsloth", random_state=SEED)

SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=load_dataset("json",
                               data_files=f"data/processed/{LANG}_train.jsonl",
                               split="train"),
    dataset_text_field="text", max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine",
        fp16=True, output_dir=f"adapters/{LANG}_lora",
        logging_steps=25, seed=SEED, data_seed=SEED)
).train()

model.save_pretrained(f"adapters/{LANG}_lora")
tokenizer.save_pretrained(f"adapters/{LANG}_lora")
print(f"Adapter saved -> adapters/{LANG}_lora")
```

> Local LSP note: r=8 (not 16) keeps the merged model smaller and faster for local inference.
> max_seq_length=2048 is enough for LSP completions and cuts training time vs 4096.
> seed=42 everywhere - your advisor specifically mentioned reproducibility.

---

## src/merge_convert.py (Phase 4)

```python
from unsloth import FastLanguageModel
import sys

LANG = sys.argv[1]  # "python" or "go"

model, tokenizer = FastLanguageModel.from_pretrained(
    f"adapters/{LANG}_lora", max_seq_length=2048, load_in_4bit=True)

model = model.merge_and_unload()
model.save_pretrained(f"merged/{LANG}_merged", safe_serialization=True)
tokenizer.save_pretrained(f"merged/{LANG}_merged")

print(f"Merged -> merged/{LANG}_merged")
print(f"Convert: cd llama.cpp && python convert_hf_to_gguf.py ../merged/{LANG}_merged "
      f"--outtype q4_k_m --outfile ../gguf/{LANG}_coder.gguf")
```

---

## ollama/Modelfile_python

```
FROM ./gguf/python_coder.gguf
PARAMETER num_ctx 2048
PARAMETER temperature 0.1
PARAMETER stop "<|im_end|>"
SYSTEM "You are an expert Python programmer. Respond only with complete, correct code."
```

## ollama/Modelfile_go

```
FROM ./gguf/go_coder.gguf
PARAMETER num_ctx 2048
PARAMETER temperature 0.1
PARAMETER stop "<|im_end|>"
SYSTEM "You are an expert Go programmer. Respond only with complete, correct code."
```

---

## Continue.dev config (~/.continue/config.json)

```json
{
  "models": [
    { "title": "Python LoRA (local)", "provider": "ollama", "model": "python-coder" },
    { "title": "Go LoRA (local)",     "provider": "ollama", "model": "go-coder"     }
  ],
  "tabAutocompleteModel": {
    "title": "Autocomplete",
    "provider": "ollama",
    "model": "python-coder"
  }
}
```

---

## Results Table (copy into paper)

| Model                           | pass@1 Python | pass@1 Go | Latency (ms) |
|---------------------------------|:-------------:|:---------:|:------------:|
| Qwen2.5-Coder-1.5B (base)       |      ?%       |     ?%    |      ?       |
| + Python LoRA (r=8, 2 epochs)   |    **?%**     |    N/A    |      ?       |
| + Go LoRA     (r=8, 2 epochs)   |      N/A      |  **?%**   |      ?       |

> Fill in after running evaluate_functional_correctness on each .jsonl file.
> Even a 5% improvement in pass@1 over baseline proves LoRA fine-tuning works.
> Include one side-by-side completion example (same prompt, base vs LoRA output) in the paper.

---

## Paper section map

| Paper section              | Source                               |
|----------------------------|--------------------------------------|
| Methodology diagram        | Screenshot from previous response    |
| Dataset stats table        | Printed output of eda.py             |
| EDA figures                | results/eda_python.png, eda_go.png   |
| Baseline results           | evaluate_functional_correctness on baseline.jsonl |
| Final results + comparison | Results table above                  |
| Pitfalls discussion        | Data leakage + class imbalance notes |
| Reproducibility            | seed=42 in all scripts               |
