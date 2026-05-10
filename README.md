# Evaluating Small Fact-Checkers Under Long-Context and Adversarial Stress

**Course Project: Human + AI Scientific Discovery**

This project evaluates MiniCheck-style small fact-checkers under two stress conditions:

1. **Long-context natural stress**: Testing on documents ranging from hundreds to 74k tokens
2. **Adversarial injection**: Injecting synthetic hallucinations at beginning, middle, and end positions

## Repository Structure

```
.
├── src/                    # Evaluation source code
│   ├── evaluate.py         # Main evaluation pipeline
│   ├── adversarial_injection.py  # Adversarial experiments
│   ├── final_analysis.py  # Analysis and aggregation
│   ├── plot_*.py          # Visualization scripts
│   └── ...
├── data/                  # Raw evaluation results (JSON)
│   ├── results_*.json     # Model evaluation results
│   └── adversarial_*.json # Adversarial injection results
├── final_results/         # Aggregated results and figures
│   ├── summary_*.csv      # Aggregated data tables
│   └── *.png              # Generated figures
└── paper/                 # Final report
    └── paper.tex          # NeurIPS 2026 format report
```

## Key Findings

| Finding | Details |
|---------|---------|
| Lost in middle | 9-10% BAcc degradation at 1k-2k tokens on ExpertQA, recovery at 2k-4k |
| Throughput cost | 250x reduction on 74k-token docs (3.9 SPM vs 1000+ SPM) |
| Adversarial vulnerability | 57.5% fooling rate; entity hallucinations most effective (38.3% detection) |
| No positional rescue | Injection detection is position-invariant |

## Quick Start

```bash
# Activate environment
conda activate minicheck-eval

# Run evaluation
python src/evaluate.py --model flan-t5-large --dataset ExpertQA

# Generate analysis and figures
python src/final_analysis.py --results_dir data --output_dir final_results
```

## Models Evaluated

- **flan-t5-large** (770M): MiniCheck's default lightweight fact-checker
- **Bespoke-MiniCheck-7B** (7B): Larger specialized fact-checker
- OpenRouter API: Gemma-4-26B, Gemma-4-31B, GPT-OSS-120B, GPT-OSS-20B, Trinity-Large

## Datasets

| Dataset | Avg Tokens | Samples |
|---------|-----------|---------|
| ExpertQA | 433 | 3,702 |
| RAGTruth | 412 | 16,371 |
| SummHay | 74,488 | 100 |
| SciFact | 57 | 188 |
| TofuEval-MediaS | 778 | 726 |
| TofuEval-MeetB | 792 | 100 |
| Lfqa | 320 | 100 |

## About the "Lost in the Middle" U-Shape

The "lost in the middle" phenomenon (U-shaped performance curve) is **only observable on ExpertQA** because it is the **only dataset with samples in all 5 length bins** (0-500, 500-1000, 1000-2000, 2000-4000, 4000+ tokens).

Other datasets lack sufficient samples in longer bins:
- **RAGTruth**: Only 3 bins (0-500, 500-1000, 1000-2000) - shows degradation, no recovery
- **Lfqa**: Only 2 bins (0-500, 500-1000)
- **TofuEval-MeetB**: Only 2 bins (500-1000, 1000-2000)
- **SciFact**: Only 1 bin (0-500)
- **SummHay**: Only 1 bin (4000+)

This is a **data limitation**, not a model limitation. The models do show the U-shape on ExpertQA.

## Requirements

- Python 3.10+
- pandas, matplotlib, seaborn, numpy
- CUDA-capable GPU for local model inference

## References

- Tang et al. MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents (EMNLP 2024)
- Laban et al. The Good, The Bad, and The Summary (arXiv 2024)
- Wang et al. AlignScore: Evaluating Factual Consistency (ACL 2023)

## Author

Chenan Wang - Spring 2026
