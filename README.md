# Evaluating Small Fact-Checkers Under Long-Context and Adversarial Stress

**Course Project: Human + AI Scientific Discovery**

## Overview

This project evaluates MiniCheck-style fact-checkers under long-context and adversarial stress:

1. **Long-context evaluation**: Testing models on documents ranging from 57 to 74,488 tokens
2. **Adversarial injection**: Testing robustness against hallucinations injected at different positions

## Repository Structure

```
.
├── src/                    # Source code
│   ├── evaluate.py         # Main evaluation
│   ├── adversarial_injection.py  # Adversarial experiments
│   ├── final_analysis.py   # Analysis and aggregation
│   └── plot_*.py          # Visualization scripts
├── paper.pdf               # Final report (NeurIPS 2026 format)
└── README.md
```

## Setup

```bash
conda create -n minicheck-eval python=3.10
conda activate minicheck-eval
pip install pandas matplotlib seaborn numpy
```

## Reproducing Results

```bash
# Download datasets from original sources:
# - ExpertQA, RAGTruth, SciFact: LLM-AggreFact (HuggingFace)
# - SummHay, TofuEval: Contact dataset authors

# Run evaluation
cd src
python evaluate.py --model flan-t5-large --dataset ExpertQA

# Run adversarial injection
python adversarial_injection.py

# Generate analysis and figures
python final_analysis.py --results_dir /path/to/results --output_dir /path/to/output
```

## Key Findings

| Finding | Details |
|---------|---------|
| Lost in middle | 9-10% BAcc degradation at 1k-2k tokens on ExpertQA, recovery at 2k-4k |
| Throughput cost | 250x reduction on 74k-token docs (3.9 SPM vs 1000+ SPM) |
| Adversarial vulnerability | 57.5% fooling rate; entity hallucinations most effective (38.3% detection) |
| No positional rescue | Injection detection is position-invariant |

**Note**: The full U-shaped "lost in the middle" recovery curve is only observable on ExpertQA, which is the only dataset with samples across all 5 length bins. Other datasets lack samples in longer bins due to natural document length distributions.

## Models Evaluated

- **flan-t5-large** (770M): MiniCheck's default lightweight fact-checker
- **Bespoke-MiniCheck-7B** (7B): Larger specialized fact-checker
- OpenRouter API: Gemma-4-26B, Gemma-4-31B, GPT-OSS-120B, GPT-OSS-20B, Trinity-Large

## Author

Chenan Wang - Spring 2026
