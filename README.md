# Cautious Tree Search Decoding (CTSD) for vLLM

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-Supported-green.svg)](https://github.com/vllm-project/vllm)

This repository provides an official vLLM plugin implementation of **Cautious Tree Search Decoding (CTSD)**, introduced in the paper:
> *Cautious Tree Search Decoding: Perplexity-Guided Parallel Exploration for Language Model Generation* (Carlo Cetrone, 2026).

---

## 🔍 How It Works

Autoregressive language models typically commit to a single token at each step $t$ using greedy decoding or stochastic sampling. This myopic commitment often leads to compounding errors on multi-step reasoning tasks.

**Cautious Tree Search Decoding** replaces single-token commitment with lookahead tree exploration and perplexity-guided pruning:

1. **Frontier Expansion**: At each step, CTSD samples `BREADTH` ($B$) candidate tokens from the top-$k$ distribution for each frontier node.
2. **KV Cache Prefix Sharing**: Because all sequences share the prompt and intermediate branch prefixes, vLLM's **PagedAttention / Automatic Prefix Caching (APC)** reuses the KV cache across all branches in parallel forward passes.
3. **Cautious Evaluation & Pruning**:
   - The tree grows until reaching lookahead depth `DEPTH` ($D$), evaluating $B^D$ leaf candidate sequences (e.g., $B=3, D=3 \implies 27$ sequences).
   - For all root-to-leaf paths $p$, CTSD computes average path perplexity:
     $$\text{PPL}(p \mid x) = \exp\left(-\frac{1}{|p|}\sum_{t=1}^{|p|} \log p_\theta(y_t \mid x, y_{<t})\right)$$
   - It selects the path $p^*$ with the lowest perplexity.
   - It **commits** the first token $y_1^*$ of $p^*$ to the final generated output.
   - It **prunes** all branches that do not start with $y_1^*$, retaining the $B^{D-1}$ remaining candidate paths (e.g. $3^2 = 9$ sequences).
   - The $B^{D-1}$ remaining sequences are expanded again by factor $B$ in the next forward pass ($9 \times 3 = 27$ sequences), maintaining the tree depth in a rolling lookahead window.

---

## 🚀 Installation

### In Google Colab or Local Linux / GPU Environment:
```bash
# 1. Install vLLM
pip install vllm

# 2. Install cautious-decoding plugin from GitHub
pip install git+https://github.com/CarloCetrone/cautious-decoding.git
```

---

## 💡 Quickstart Example

```python
from vllm import LLM
from cautious_decoding import CautiousDecoder

# Initialize vLLM with Automatic Prefix Caching enabled
llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.85,
)

prompt = "Solve step by step: A farmer has 15 cows. All but 6 die. How many are left?"

# Method 1: Using the attached vLLM method (via plugin)
result = llm.cautious_generate(
    prompt=prompt,
    breadth=3,       # 3 tokens branched at each step
    depth=3,         # Tree evaluated at depth 3 (27 paths evaluated)
    max_tokens=256,
    temperature=0.7,
    verbose=True,
)
print("Generated Output:\n", result["generated_text"])

# Method 2: Using the CautiousDecoder class directly
decoder = CautiousDecoder(llm=llm, breadth=3, depth=3)
result = decoder.generate(prompt=prompt, verbose=True)
print("Generated Output:\n", result["generated_text"])
```

---

## 📊 Hyperparameters

- **`breadth` ($B$)**: Branching factor (default: `3`).
- **`depth` ($D$)**: Tree depth evaluated before committing a token (default: `3`).
- **`temperature`**: Sampling temperature for logit distribution (default: `0.7`).
- **`max_tokens`**: Maximum tokens committed to output (default: `512`).

---

## 📄 Citation

```bibtex
@article{cetrone2026cautious,
  title={Cautious Tree Search Decoding: Perplexity-Guided Parallel Exploration for Language Model Generation},
  author={Cetrone, Carlo},
  year={2026}
}
```
