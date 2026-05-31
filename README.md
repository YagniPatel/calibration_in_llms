# LLM Calibration via Self-Evaluation

A framework for measuring and improving confidence calibration in small language models using self-evaluation. Applied to Qwen-2.5-1.5B and SmolLM-1.7B on TriviaQA, with temperature scaling post-processing to reduce Expected Calibration Error (ECE).

---

## 🧠 Problem Statement

Language models can be confidently wrong. **Calibration** measures how well a model's stated confidence matches its actual accuracy — a well-calibrated model that says "I'm 80% sure" should be correct 80% of the time.

This project asks: *Can small LLMs estimate their own confidence reliably through self-evaluation, and can we improve that calibration without retraining?*

---

## 🔬 Methodology

### Self-Evaluation Framework

Instead of relying on token probability as confidence, we prompt the model to evaluate its own answer:

```
Question: [TriviaQA question]
Model answer: [generated answer]
Is this answer correct? Respond with True or False.
```

We extract the **logit scores** for the `True` and `False` tokens from the final prediction step, then convert these to a confidence score via softmax:

```
confidence = softmax(logit_True) / (softmax(logit_True) + softmax(logit_False))
```

### Temperature Scaling

Post-hoc calibration without modifying the model weights:

```
calibrated_confidence = softmax(logits / T)
```

Temperature `T` is optimized on a held-out validation set to minimize ECE.

---

## 📊 Results

### ECE (Expected Calibration Error) — lower is better

| Model | ECE (Before) | ECE (After Temp. Scaling) | Improvement |
|---|---|---|---|
| **Qwen-2.5-1.5B** | 0.217 | **0.132** | −39.2% |
| **SmolLM-1.7B** | 0.078 | **0.074** | −5.1% |

### Discriminative Ability (AUROC) — higher is better

| Model | AUROC |
|---|---|
| Qwen-2.5-1.5B | **0.787** |
| SmolLM-1.7B | 0.626 |

### Key Finding

> **Stronger discriminative ability (AUROC) does not imply better initial calibration (ECE).**

Qwen-2.5-1.5B was better at *distinguishing* correct from incorrect answers (AUROC 0.787) but started *more miscalibrated* (ECE 0.217) than SmolLM-1.7B. This reveals a fundamental tradeoff in how small LLMs express confidence.

---

## 🗂️ Project Structure

```
llm-calibration-self-evaluation/
├── data/
│   └── triviaqa_sample.json       # 500 TriviaQA questions used for eval
├── models/
│   └── temperature_params.json    # Optimized temperature values per model
├── outputs/
│   ├── calibration_plots/         # Reliability diagrams before/after scaling
│   └── results_summary.csv        # Full results table
├── self_evaluate.py               # Self-evaluation prompting + logit extraction
├── temperature_scaling.py         # Temperature scaling optimization
├── evaluate_calibration.py        # ECE, AUROC, reliability diagram computation
└── run_pipeline.py                # End-to-end pipeline script
```

---

## 📦 Setup

```bash
git clone https://github.com/YagniPatel/llm-calibration-self-evaluation.git
cd llm-calibration-self-evaluation

pip install torch transformers datasets scikit-learn numpy matplotlib
```

Models used (downloaded via HuggingFace Hub automatically):
- `Qwen/Qwen2.5-1.5B-Instruct`
- `HuggingFaceTB/SmolLM-1.7B-Instruct`

---

## ▶️ How to Run

```bash
# Run self-evaluation on TriviaQA samples
python self_evaluate.py --model qwen2.5-1.5b --n_samples 500

# Fit temperature scaling on validation split
python temperature_scaling.py --model qwen2.5-1.5b

# Evaluate calibration (ECE, AUROC, reliability diagrams)
python evaluate_calibration.py --model qwen2.5-1.5b --plot

# Or run full pipeline for both models
python run_pipeline.py --models qwen2.5-1.5b smollm-1.7b
```

---

## 📉 Reliability Diagrams

Reliability diagrams show confidence bins vs. actual accuracy. A perfectly calibrated model follows the diagonal line.

- **Before temperature scaling:** Qwen-2.5-1.5B shows overconfidence at high confidence bins
- **After temperature scaling:** Distribution pulls toward the diagonal, reducing systematic overconfidence

*(See `outputs/calibration_plots/` for full plots)*

---

## 🔍 Key Takeaways

1. **Self-evaluation is a viable, lightweight confidence estimation technique** for small LLMs without requiring access to full vocabulary logits.
2. **Temperature scaling is effective even for small models** — reduces ECE by up to 39% without any weight updates.
3. **AUROC and ECE measure different things** — one measures discrimination, the other measures reliability. Optimizing one doesn't fix the other.

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `Qwen-2.5-1.5B` · `SmolLM-1.7B` · `TriviaQA` · `Scikit-learn` · `Matplotlib`

---

## 📚 References

- [TriviaQA Dataset](https://nlp.cs.washington.edu/triviaqa/)
- [On Calibration of Modern Neural Networks (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)
- [Can LLMs Express Their Uncertainty?](https://arxiv.org/abs/2306.13063)
