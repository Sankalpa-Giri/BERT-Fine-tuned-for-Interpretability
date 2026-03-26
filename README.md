# CSE39 NLP Assignment 2 — Part 5: Interpretability Gap
**Roll Number:** 2305154  
**Course:** CSE39 — Natural Language Processing  
**Instructor:** Dr. Sambit Praharaj, KIIT

---

## Problem Statement
BERT lacks interpretability — it is a "black box" that produces predictions without
explaining which parts of the input influenced the decision.

## Objective
Make BERT's predictions explainable by:
1. Extracting and visualizing **attention weights**
2. Computing **gradient-based token importance** (Gradient × Input)
3. Comparing both methods and identifying potential **model biases**

---

## Project Structure
```
bert_interpretability/
├── BERT_fine-tuned.py            ← Full pipeline (train → interpret → visualize)
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
└── outputs/           ← Generated heatmaps and plots (auto-created)
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

Requires Python 3.8+ and a GPU (optional but faster).

---

## Methods Implemented

### 1. Attention Rollout (Abnar & Zuidema, 2020)
- Extracts attention weights from all 12 BERT layers
- Propagates attention through layers (with residual connections)
- Produces a single importance score per token w.r.t. the [CLS] token
- **Output:** `attn_heatmap_*.png`, `attn_head_layer12_*.png`

### 2. Gradient × Input
- Registers a forward hook on BERT's embedding layer
- Backpropagates from the predicted class logit
- Computes element-wise product of gradients and embeddings
- Takes L2 norm over the hidden dimension per token
- **Output:** `grad_heatmap_*.png`

### 3. Comparison Visualization
- Side-by-side bar chart comparing both methods
- **Output:** `comparison_*.png`

### 4. Bias Analysis (Extension)
- Tests gendered pronoun pairs ("he" vs "she", "his" vs "her")
- Checks if the model assigns different importance to gender tokens
- **Output:** `bias_analysis.png`

---

## Output Files

| File | Description |
|------|-------------|
| `attn_heatmap_N.png`       | Token importance via attention rollout |
| `attn_head_layer12_N.png`  | 2D attention matrix for Layer 12, Head 1 |
| `grad_heatmap_N.png`       | Token importance via Gradient × Input |
| `comparison_N.png`         | Side-by-side attention vs gradient |
| `bias_analysis.png`        | Bias detection with gender pairs |

---

## Results Table

| Model | Accuracy | F1 Score | Training Time | Parameters |
|-------|----------|----------|---------------|------------|
| Baseline BERT (no fine-tune) | ~0.50 | ~0.50 | — | 109M |
| Fine-tuned BERT (SST-2) | ~0.91+ | ~0.91+ | ~5–10 min | 109M |

*(Exact values printed at end of run)*

---

## Key Findings
- **Attention rollout** highlights contextually important tokens across all layers
- **Gradient × Input** is more faithful to the actual decision boundary
- **Bias analysis** reveals whether the model treats gendered pronouns differently
- Sentiment words (e.g., *wonderful*, *terrible*) consistently receive high scores in both methods

---

## References
1. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
2. Abnar & Zuidema (2020). Quantifying Attention Flow in Transformers.
3. Simonyan et al. (2014). Deep Inside Convolutional Networks (gradient saliency).
4. Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks.
