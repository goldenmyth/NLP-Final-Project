# Positional Encoding Alignment for Efficient KV-Cache Reuse in Multi-Document QA

### [Read the Full Report](/Report.pdf)

This repository contains the code for the NLP final project on Positional Encoding Alignment (PEA) for efficient KV-cache reuse in Multi-Document Question Answering.

The project studies whether independently precomputed document KV-caches can be reused inside a larger multi-document prompt without fully recomputing the document prefill. The main idea is to correct the RoPE positional phase of cached keys before concatenating document caches into one global cache.

## Project idea

Large language models are often used in Retrieval-Augmented Generation and Multi-Document QA, where the prompt may contain several long documents. The expensive part of inference is the prefill stage, because the model has to process the full context before generation.

This project compares three strategies:

- **Baseline** - standard full-prefill inference over the complete prompt.
- **Aligned** - independent document prefill, RoPE-based key shifting, and KV-cache concatenation.
- **Naive** - independent document prefill and KV-cache concatenation without positional correction.

The goal is to check whether RoPE-aware positional alignment makes KV-cache reuse more stable than simple cache concatenation.

## Repository structure

```
NLP-Final-Project/
├── config.yaml
├── requirements.txt
├── main_benchmark.py
├── main_scaling.py
└── src/
    ├── config_loader.py
    ├── model_engine.py
    ├── utils_cache.py
    ├── utils_data.py
    ├── utils_metrics.py
    └── utils_rope.py
```

Main files:

- `main_benchmark.py` runs the MuSiQue Multi-hop QA benchmark.
- `main_scaling.py` runs the synthetic Needle in a Haystack scaling experiment.
- `config.yaml` stores model, dataset, and experiment settings.
- `src/utils_rope.py` implements RoPE key shifting.
- `src/utils_cache.py` handles KV-cache extraction, alignment, concatenation, and memory estimation.
- `src/utils_metrics.py` computes answer-level and internal comparison metrics.

## Environment

The reported experiments were run with:

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- CUDA-enabled GPU

The main model is:

```
Qwen/Qwen2.5-3B-Instruct
```

A GPU is strongly recommended. The reported timing results were obtained on a single NVIDIA V100 32GB GPU.

## Installation

Clone the repository:

```bash
git clone <https://github.com/goldenmyth/NLP-Final-Project.git>
cd NLP-Final-Project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If your environment requires Hugging Face authentication, log in before running the experiments:

```bash
huggingface-cli login
```

## Configuration

The main settings are stored in [config.yaml](/config.yaml)

`attn_implementation: "eager"` is used because the benchmark needs attention weights for the internal comparison metrics.

## Reproducing the results

Run all commands from the repository root.

### MuSiQue benchmark

```bash
python main_benchmark.py
```

This script loads the MuSiQue answerable validation split, selects examples with more than one supporting paragraph, and compares the Aligned and Naive cache reuse strategies against the full Baseline reference.

The output is saved to:

```
results/musique_results_3b.csv
```

The file contains per-example predictions and metrics, including Exact Match, F1, Top-1 agreement, KL divergence, gold-token rank, NLL, and attention correlation.

### Needle in a Haystack scaling test

```bash
python main_scaling.py
```

This script creates synthetic multi-document prompts with a hidden secret code and tests how Baseline, Aligned, and Naive behave as the number of documents increases.

The outputs are saved to:

```
results/scaling_results_3b.csv
results/scaling_plots.png
```

The CSV contains latency, speedup, context length, KV-cache memory, and Exact Match for each strategy and document count.

## Expected results

On MuSiQue, Aligned and Naive have similar answer-level quality: Aligned reaches about 37.38 EM and 49.80 F1, while Naive reaches about 37.12 EM and 49.77 F1. The main difference is internal behavior: Aligned preserves the Baseline attention structure better, with attention correlation around 0.9462 compared with 0.8378 for Naive.

On the synthetic Needle in a Haystack test, Baseline remains correct for all tested document counts. Naive works on shorter contexts but fails starting from 20 documents. Aligned remains correct up to 25 documents and fails only on longer contexts.

In the reported setup, Aligned achieves an average speedup of about 11.94x over full Baseline prefill, while also preserving better average Exact Match than Naive.

## Reproducibility notes

The exact timing values may differ depending on GPU type, CUDA version, library versions, and system load. Quality metrics should be close to the reported values when using the same model, dataset split, seed, and configuration.

The first run can take longer because the model and dataset need to be downloaded. If GPU memory is limited, reduce `num_samples_musique` or the maximum document count in `config.yaml`.

## Conclusion

The experiments show that RoPE-aware positional alignment is a cheap and useful correction for KV-cache reuse. Simple cache concatenation can work on short contexts, but it becomes unstable as the number of documents grows. Aligned cache reuse shifts the failure point to longer contexts and better preserves the model's internal attention structure.

The method does not fully replace full prefill, because independently cached documents cannot attend to each other during document precomputation. However, it is useful for RAG and Multi-Document QA settings where the same document collection is reused across multiple queries.
