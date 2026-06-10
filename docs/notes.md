# Paper guide
## Introduction
## Theory (Literature Review)

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
Focus: `decoder-only`

Cite as:
```
@misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}
```

2. [LoRA](https://arxiv.org/pdf/2106.09685)

Focus: `low-rank decomposition`, `rank r`, `lora_alpha`, `target modules`, `zero inference latency nakon merge-a`

Cite as:
```
@misc{hu2021loralowrankadaptationlarge,
      title={LoRA: Low-Rank Adaptation of Large Language Models}, 
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2106.09685}, 
}
```

3. [QLoRA](https://arxiv.org/abs/2305.14314)

Focus: `4-bit NF4`, `zašto možeš trenirati na consumer GPU`

Cite as:
```
@misc{dettmers2023qloraefficientfinetuningquantized,
      title={QLoRA: Efficient Finetuning of Quantized LLMs}, 
      author={Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer},
      year={2023},
      eprint={2305.14314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.14314}, 
}
```

4. [PEFT](https://arxiv.org/pdf/2308.10462)

Focus: `LoRA improvements for code generation`, `pass@1`

Cite as:
```
@misc{weyssow2024exploringparameterefficientfinetuningtechniques,
      title={Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models}, 
      author={Martin Weyssow and Xin Zhou and Kisub Kim and David Lo and Houari Sahraoui},
      year={2024},
      eprint={2308.10462},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2308.10462}, 
}
```

5. [PEFT survey](https://arxiv.org/abs/2403.14608)

Focus: `LoRA within broader PEFT family`

Cite as:
```
@misc{han2024parameterefficientfinetuninglargemodels,
      title={Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey}, 
      author={Zeyu Han and Chao Gao and Jinyang Liu and Jeff Zhang and Sai Qian Zhang},
      year={2024},
      eprint={2403.14608},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.14608}, 
}
```

6. [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)

Focus: `current state of the model we are fine tuning`, `1.5B`

Cite as:
```
@misc{hui2024qwen25codertechnicalreport,
      title={Qwen2.5-Coder Technical Report}, 
      author={Binyuan Hui and Jian Yang and Zeyu Cui and Jiaxi Yang and Dayiheng Liu and Lei Zhang and Tianyu Liu and Jiajun Zhang and Bowen Yu and Keming Lu and Kai Dang and Yang Fan and Yichang Zhang and An Yang and Rui Men and Fei Huang and Bo Zheng and Yibo Miao and Shanghaoran Quan and Yunlong Feng and Xingzhang Ren and Xuancheng Ren and Jingren Zhou and Junyang Lin},
      year={2024},
      eprint={2409.12186},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.12186}, 
}
```

7. [CodeSearchNet](https://arxiv.org/abs/1909.09436)

Focus: `characteristics of the dataset`

Cite as:
```
@misc{husain2020codesearchnetchallengeevaluatingstate,
      title={CodeSearchNet Challenge: Evaluating the State of Semantic Code Search}, 
      author={Hamel Husain and Ho-Hsiang Wu and Tiferet Gazit and Miltiadis Allamanis and Marc Brockschmidt},
      year={2020},
      eprint={1909.09436},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1909.09436}, 
}
```

8. [HumanEval-XL](https://arxiv.org/abs/2402.16694)

Focus: `HumanEval`, `pass@k`

Cite as:
```
@misc{peng2024humanevalxlmultilingualcodegeneration,
      title={HumanEval-XL: A Multilingual Code Generation Benchmark for Cross-lingual Natural Language Generalization}, 
      author={Qiwei Peng and Yekun Chai and Xuhong Li},
      year={2024},
      eprint={2402.16694},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.16694}, 
}
```

9. [A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct](https://arxiv.org/abs/2601.14277)

Focus: `GGUF`

Cite as:
```
@misc{kurt2026quantizationiuseunified,
      title={Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct}, 
      author={Uygar Kurt},
      year={2026},
      eprint={2601.14277},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.14277}, 
}
```

10. [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)

Focus: use this as help
[Fine-Tuning Language Models with Fill-in-the-Middle: A Comprehensive Guide](https://blog.gopenai.com/fine-tuning-language-models-with-fill-in-the-middle-a-comprehensive-guide-58a022b8f8df)

Cite as:
```
@misc{bavarian2022efficienttraininglanguagemodels,
      title={Efficient Training of Language Models to Fill in the Middle},
      author={Mohammad Bavarian and Heewoo Jun and Nikolas Tezak and John Schulman and Christine McLeavey and Jerry Tworek and Mark Chen},
      year={2022},
      eprint={2207.14255},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2207.14255},
}
```

11. [Clean Code, Better Models](https://arxiv.org/pdf/2508.11958)

Focus: `a short mention of every technique`

Cite as:
```
@misc{xue2025cleancodebettermodels,
      title={Clean Code, Better Models: Enhancing LLM Performance with Smell-Cleaned Dataset},
      author={Zhipeng Xue and Xiaoting Zhang and Zhipeng Gao and Xing Hu and Shan Gao and Xin Xia and Shanping Li},
      year={2025},
      eprint={2508.11958},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2508.11958},
}
```

## Methodology

### Data preparation

- `JSONL`
- `Fill in the middle` mentioned in theory
- [`Deduplication`](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation)
- `Code smells` mentioned in theory: `Clean Code, Better Models`
- `Docstring quality filtering`
- `Manage Token Lengths`
- `Diversify Completion Granularity`

### Base model eval

- [Benchmark](https://huggingface.co/datasets/floatai/HumanEval-XL)

## Results

- Mention amount of data filtered out in data preparation and why
