# Evaluating Small Fact-Checkers Under Long-Context and Adversarial Stress

**Course Project: Human + AI Scientific Discovery**
**Track: Open-Ended Research**

## Overview

This project evaluates MiniCheck-style fact-checkers under long-context and adversarial stress:
1. **Long-context evaluation**: Testing models on documents ranging from 57 to 74,488 tokens
2. **Adversarial injection**: Testing robustness against hallucinations at beginning, middle, and end positions

## Repository Structure

```
.
├── src/                    # Source code
│   ├── evaluate.py         # Main evaluation
│   ├── adversarial_injection.py  # Adversarial experiments
│   ├── final_analysis.py   # Analysis and aggregation
│   ├── data_loader.py      # Dataset loading
│   ├── analysis.py          # Result analysis
│   ├── openrouter_client.py # API benchmarks
│   ├── plot_*.py           # Visualization scripts
│   └── smoke_test.py       # Quick sanity check
└── README.md
```

\textbf{Note}: The final report (paper.pdf) should be submitted separately to the course system, not stored in this repo.

## Setup

```bash
conda create -n minicheck-eval python=3.10
conda activate minicheck-eval
pip install pandas matplotlib seaborn numpy
```

## Reproducing Results

### 1. Main Evaluation (Long-Context)

```bash
cd src

# Evaluate flan-t5-large on ExpertQA
python evaluate.py --model flan-t5-large --dataset ExpertQA

# Evaluate Bespoke-MiniCheck-7B on ExpertQA
python evaluate.py --model Bespoke-MiniCheck-7B --dataset ExpertQA

# Evaluate on all datasets (flan-t5-large)
for dataset in ExpertQA RAGTruth SciFact SummHay TofuEval-MediaS TofuEval-MeetB Lfqa; do
    python evaluate.py --model flan-t5-large --dataset $dataset
done

# Evaluate on all datasets (Bespoke-MiniCheck-7B)
for dataset in ExpertQA RAGTruth SciFact SummHay TofuEval-MediaS TofuEval-MeetB Lfqa; do
    python evaluate.py --model Bespoke-MiniCheck-7B --dataset $dataset
done
```

### 2. Adversarial Injection Experiments

```bash
cd src

# Run adversarial injection on ExpertQA (360 samples)
python adversarial_injection.py --dataset ExpertQA --output ../results_adversarial/

# Run adversarial injection on RAGTruth
python adversarial_injection.py --dataset RAGTruth --output ../results_adversarial/
```

### 3. OpenRouter API Benchmarks

```bash
cd src

# Test Gemma-4-26B on ExpertQA
python openrouter_client.py --model Gemma-4-26B --dataset ExpertQA --api_key YOUR_API_KEY

# Test other API models
python openrouter_client.py --model Gemma-4-31B --dataset SciFact --api_key YOUR_API_KEY
python openrouter_client.py --model GPT-OSS-120B --dataset ExpertQA --api_key YOUR_API_KEY
python openrouter_client.py --model GPT-OSS-20B --dataset SciFact --api_key YOUR_API_KEY
python openrouter_client.py --model Trinity-Large --dataset ExpertQA --api_key YOUR_API_KEY
```

### 4. Analysis and Figure Generation

```bash
cd src

# Aggregate all results and generate figures
python final_analysis.py \
    --results_dir ../results \
    --adv_results ../results_adversarial \
    --openrouter_results ../results_openrouter \
    --output_dir ../final_results

# Generate publication-ready paper figures
python plot_paper_figures.py --data_dir ../final_results --output_dir ../final_results

# Generate context degradation plots
python plot_context_degradation.py

# Generate U-shape visualization
python plot_lost_in_middle.py
```

### 5. Quick Smoke Test

```bash
cd src
python smoke_test.py
```

## Datasets

Download from original sources:
- **ExpertQA, RAGTruth, SciFact**: LLM-AggreFact benchmark (HuggingFace)
- **SummHay**: Contact dataset authors for access
- **TofuEval**: From original paper authors

## Key Findings

| Finding | Details |
|---------|---------|
| Lost in middle | 9-10% BAcc degradation at 1k-2k tokens on ExpertQA, recovery at 2k-4k |
| Throughput cost | 250x reduction on 74k-token docs (3.9 SPM vs 1000+ SPM) |
| Adversarial vulnerability | 57.5% fooling rate; entity hallucinations most effective (38.3% detection) |
| No positional rescue | Injection detection is position-invariant |

**Note**: The full U-shaped "lost in the middle" recovery curve is only observable on ExpertQA, which is the only dataset with samples across all 5 length bins (0-500, 500-1000, 1000-2000, 2000-4000, 4000+). Other datasets lack samples in longer bins due to natural document length distributions.

## Models Evaluated

- **flan-t5-large** (770M): MiniCheck's default lightweight fact-checker
- **Bespoke-MiniCheck-7B** (7B): Larger specialized fact-checker
- OpenRouter API: Gemma-4-26B, Gemma-4-31B, GPT-OSS-120B, GPT-OSS-20B, Trinity-Large

## Author

Chenan Wang - Spring 2026
