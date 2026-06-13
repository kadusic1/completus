# Completus

> QLoRA fine-tuning of **Qwen2.5-Coder-1.5B** on CodeSearchNet (Python & Go),
> achieving **42.68% HumanEval pass@1** (+31.7 pp over baseline) on a single
> consumer GPU.

This repository contains the full end-to-end pipeline: data preparation,
QLoRA fine-tuning, adapter merging, GGUF export, and local serving via Ollama.
The accompanying conference paper (LaTeX source in `paper/`) documents the
methodology and results in full.

---

## Results

| Checkpoint | HumanEval Pass@1 | 95% Wilson CI | Correct / 164 |
|---|:---:|:---:|:---:|
| Qwen2.5-Coder-1.5B (baseline) | 10.98% | [7.1, 16.7] | 18 |
| Completus Python (fine-tuned) | **42.68%** | [35.4, 50.3] | **70** |
| **Improvement** | **+31.70 pp** | non-overlapping | **+52** |

Evaluation: zero-shot, greedy decoding, 164 problems, single GPU.

---

## Hardware

Trained and evaluated on an **NVIDIA RTX 4060 Laptop (8 GB VRAM)**.
No multi-GPU setup required.

| Run | Steps | Time |
|---|:---:|:---:|
| Python (31,966 samples) | 1,998 | ~2.2 h |
| Go (21,284 samples) | 1,331 | ~62 min |

---

## Dataset Statistics

Source: [`claudios/code_search_net`](https://huggingface.co/datasets/claudios/code_search_net) on HuggingFace Hub.

| Language | Full train split | Pipeline cap | After pipeline | Retained |
|---|:---:|:---:|:---:|:---:|
| Python | 412,178 | 50,000 | 31,966 | 63.9% |
| Go | 317,832 | 50,000 | 21,284 | 42.6% |
| **Total** | | **100,000** | **53,250** | **53.3%** |

### Preparation pipeline (5 stages)

1. **Near-deduplication** — MinHash LSH, 128 permutations, 5-token shingles, Jaccard threshold 0.8
2. **Code-smell filtering** — drop lines > 120 chars, functions < 30 tokens, bare `pass`/ellipsis bodies
3. **Docstring filtering** — require >= 15 tokens, no `TODO`/`FIXME`/`XXX`, docstring distinct from function name
4. **Fill-in-the-middle transform** — 80% of samples get PSM/SPM transform using FIM sentinel tokens; 20% kept causal
5. **Length management** — enforce 2048-token limit

---

## Model & Training

| Hyperparameter | Value |
|---|---|
| Base model | Qwen2.5-Coder-1.5B |
| Quantisation | 4-bit NF4 (QLoRA) |
| LoRA rank r | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down |
| Batch size (per device) | 2 |
| Gradient accumulation | 8 (effective batch 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Optimiser | AdamW 8-bit |
| Epochs | 1 |
| Max sequence length | 2048 |
| Seed | 3407 |

---

## Repository Structure

```
completus/
├── paper/                        # LaTeX source (multi-file)
│   ├── main.tex                  # Root file — preamble, title, \input{}
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── results.tex
│   ├── conclusion.tex
│   ├── references.tex
│   ├── fig_pipeline.pdf
│   ├── fig_dataset.pdf
│   └── fig_passatk.pdf
├── src/
│   ├── prepare_data.py           # Download + run data pipeline
│   ├── run_train.py              # QLoRA fine-tuning entry point
│   ├── run_eval.py               # HumanEval evaluation entry point
│   ├── run_merge.py              # Adapter merge + GGUF export
│   ├── eda.py                    # Exploratory data analysis
│   ├── data_pipeline/            # Dedup, filtering, FIM transform
│   ├── train/                    # Unsloth trainer wrapper
│   ├── eval/                     # HumanEval generator + evaluator
│   └── merge_export/             # Merge + GGUF export logic
├── ollama/
│   ├── Modelfile_python          # FIM template, temp=0.0, ctx=2048
│   └── Modelfile_go
├── adapters/
│   ├── python_lora/              # LoRA adapter weights (Python)
│   └── go_lora/                  # LoRA adapter weights (Go)
├── results/
│   ├── eval/
│   │   ├── pass_at_k.json        # Baseline: {"pass@1": 0.1098}
│   │   └── pass_at_k_merged.json # Fine-tuned: {"pass@1": 0.4268}
│   └── autocomplete_eval/        # Qualitative Ollama completions
└── data/
    └── eval/                     # HumanEval problem cache
```

---

## Setup

```bash
pip install unsloth trl transformers datasets torch matplotlib
pip install human-eval
```

Requires Python 3.10+ and a CUDA-capable GPU with at least 8 GB VRAM.
[Ollama](https://ollama.com) must be installed and running for the serving step.

---

## Running the Pipeline

### 1. Data preparation

```bash
python src/prepare_data.py
```

Downloads `claudios/code_search_net` from HuggingFace, runs the 5-stage
preparation pipeline for Python and Go, and writes the training files to
`data/`.

### 2. Fine-tuning

```bash
python src/run_train.py python
python src/run_train.py go
```

Saves LoRA adapters to `adapters/python_lora/` and `adapters/go_lora/`.

### 3. Merge and export to GGUF

```bash
python src/run_merge.py python
python src/run_merge.py go
```

Merges adapters into the base model and exports quantised GGUF binaries
(`q4_k_m`) via Unsloth's export routine.

### 4. Register with Ollama

```bash
ollama create python-coder -f ollama/Modelfile_python
ollama create go-coder     -f ollama/Modelfile_go
```

Models are then available at `http://localhost:11434`.

### 5. Evaluate on HumanEval

```bash
python src/run_eval.py
```

Runs zero-shot greedy evaluation on all 164 HumanEval problems and writes
`pass@1` to `results/eval/`.

---

## Paper

The conference paper is in `paper/`. To compile:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

Figure PDFs are already included. To regenerate them from source, run
`python paper/generate_figures.py` (requires `matplotlib`, `numpy`).

---

## Authors

**Nejra Smajlovic** — Polytechnic Faculty, University of Zenica
`nejra.smajlovic.22@size.ba`

**Adi Kadusic** — Polytechnic Faculty, University of Zenica
`adi.kadusic.22@size.ba`
