 
# Tashkees-AI at AbjadMed 2026: Flat vs. Hierarchical Classification for Fine-Grained Arabic Medical QA

<p align="center">
<img src="https://placehold.co/800x250/fee2e2/ef4444?text=Tashkees-AI+Medical+QA+Classification" alt="Tashkees-AI Architecture Header">
</p>

This repository contains the official implementation, models, and evaluation framework for **Tashkees-AI**, developed for the **AbjadMed 2026 Shared Task** on Fine-Grained Arabic Medical Question Classification, colocated with the **2nd Workshop on NLP for Languages Using Arabic Script (AbjadNLP @ EACL 2026)** in Rabat, Morocco.

Our system evaluates multiple architectural paradigms—**Flat Fine-Tuned Encoders**, **Two-Stage Hierarchical Classification**, and **Lexical Ensembles**—to map highly imbalanced, noisy, and conversationally rich patient queries into 82 distinct medical specialties. Tashkees-AI achieved a peak **Macro $F_1$-score of 0.3659** on the blind test set, ranking **26th on the official leaderboard**.

#### By: [Fatimah Mohamed Emad Eldin](https://scholar.google.com/citations?user=CfX6eA8AAAAJ&hl=ar)
#### *Cairo University*

[![Paper](https://img.shields.io/badge/ACL_Anthology-2026.abjadnlp--1.20-b31b1b.svg)](https://aclanthology.org/2026.abjadnlp-1.20/)
[![Code](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/astral-fate/Tashkees-AI-at-AbjadMed)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 System Overview & Core Finding

Medical question classification over specialized domains features deep morphological complexities, severe class imbalance, and significant semantic overlap (e.g., *Dental health* vs. *Dentistry*). 

Our extensive empirical study yields a critical architectural insight: **Flat sequence classification via fine-tuned domain-specific encoders significantly outperforms complex hierarchical structures.** While hierarchical models narrower down decision spaces, they suffer from irreversible **error propagation cascades** at early stages, creating an insurmountable performance ceiling.

### The Core Architectural Dual-Pipeline
1. **Flat Strategy (Best Performer):** Text Preprocessing & Normalization $\rightarrow$ Pre-trained Arabic Encoder Baseline $\rightarrow$ Weighted Cross-Entropy Sequence Classification Head.
2. **Hierarchical Strategy:** 12-way Coarse Domain Router $\rightarrow$ 12 Independent Domain-Specific Expert Classifiers $\rightarrow$ 82 Fine-Grained Leaves (Highly susceptible to error propagation).

---

## 🚀 Quick Start & Usage

You can fine-tune or run inference directly using our preprocessed pipeline with `transformers`.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the best performing cleaned MARBERTv2 model
model_name = "FatimahEmadEldin/Tashkees-AI-MARBERTv2-AbjadMed"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sample Egyptian/Saudi conversational medical query
patient_query = "مرحبا يا دكتور، أعاني من ألم شديد في المفاصل وخاصة الركبة عند الاستيقاظ صباحاً مع تورم خفيف."

# Preprocessing / Tokenization
inputs = tokenizer(patient_query, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

# Inference
model.eval()
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

print(f"Predicted Specialty ID: {predicted_class_id}")

```

---

## ⚙️ Experimental Configurations

We exhaustively benchmarked three distinct setups to balance performance across major and minority clinical sectors:

### 1. Loss Optimization Strategy

To counter the extreme **85:1 class imbalance** (ranging from 7 to 600 samples per class), models utilize a **Weighted Cross-Entropy Loss** formulation:

$$\mathcal{L} = -\sum_{i=1}^{N} w_{y_i} \log(p_{y_i})$$

Where weights are calculated inversely proportional to class frequencies to prevent the model from ignoring low-resource clinical fields.

### 2. Hierarchical Taxonomy Mapping

We aggregated the 82 fine-grained labels into 12 macro-domains to isolate routing errors:

* **Surgical Specialties** (General, Orthopedic, Plastic, etc.)
* **Internal Medicine & Chronic** (Cardiovascular, Diabetes, Gastrointestinal, etc.)
* **Women's & Reproductive** (Women's health, Pregnancy, IVF, etc.)
* **Children's Health** (Child health, Pediatric diseases)
* *...[See paper text for the complete 12-macro mapping mapping]*

### Hyperparameters Summary

| Model / Experiment | Learning Rate | Batch Size | Epochs | Max Length | Optimization Notes |
| --- | --- | --- | --- | --- | --- |
| **MARBERTv2 (Baseline)** | $2\times 10^{-5}$ | 16 | 4 | 128 | AdamW |
| **MARBERTv2 (Optimized)** | $6\times 10^{-5}$ | 32 | 15 | 128 | Weight Decay = 0.1 |
| **Hierarchical (Stage 1)** | $2\times 10^{-5}$ | 32 | 10 | 128 | Weight Decay = 0.01 |
| **Hierarchical (Stage 2)** | $2\times 10^{-5}$ | 32 | 15 | 128 | Weight Decay = 0.01 |
| **CAMeLBERT-DA** | $6\times 10^{-5}$ | 32 | 15 | 128 | Weight Decay = 0.1 |
| **LightGBM** | 0.05 | — | 3000 | — | Num Leaves = 100 |

---

## 📊 Empirical Evaluation Results

Models were trained on 25,156 samples, validated on 2,795 records, and evaluated on an official blind test set of 18,634 rows using **Macro $F_1$-Score** as the ranking benchmark.

### Main Results Tracker

| Paradigm / Architecture | Validation $F_1$ | Test (Blind) $F_1$ | Leaf Coverage (Out of 82) |
| --- | --- | --- | --- |
| 🏆 **MARBERTv2 (Cleaned Text)** | **0.392** | **0.3659** | **79 / 82** |
| 🥈 AraBERTv2 | 0.361 | 0.358 | 82 / 82 |
| 🥉 MARBERTv2 (Raw Text) | 0.360 | 0.357 | 82 / 82 |
| Ensembles: MARBERT + LightGBM (0.7) | 0.358 | 0.358 | 79 / 82 |
| Two-Stage Hierarchical (Full Pipeline) | 0.336 | 0.332 | 68 / 82 |
| LightGBM Baseline (TF-IDF) | 0.315 | — | 82 / 82 |
| CAMeLBERT-DA | 0.293 | 0.293 | 82 / 82 |
| RAG Retrieval ($k$-NN, $k=20$) | 0.233 | 0.230 | 81 / 82 |

### Key Diagnostic Takeaways

* **Data Cleansing Dividend:** Normalizing dialectal variants and removing structural conversational noise (greetings/pleasantries) yielded a **+0.032 absolute gain** (8.9% relative increase) for MARBERTv2.
* **The Hierarchical Cascade Failure:** Despite a high Stage-1 Macro Domain Accuracy of **65.4%**, errors cascade irreversibly. Missing early routing means **30.2% of entries were permanently doomed**, restricting final leaves to only 68 active categories.
* **The Data Sparsity Boundary:** Per-category evaluation demonstrates a harsh tracking ceiling: high-resource features (e.g., *Hematological diseases*) achieved **0.83 $F_1$**, whereas extreme low-resource leaves ($<10$ samples) dropped directly to **0.00 $F_1$**.

---

## 📜 Citation

If you build upon this architecture or utilize the Tashkees-AI weights, please cite our official ACL paper:

```bibtex
@inproceedings{emad-eldin-2026-tashkees,
    title = "Tashkees-{AI} at {A}bjad{M}ed 2026: Flat vs. Hierarchical Classification for Fine-Grained {A}rabic Medical {QA}",
    author = "Emad Eldin, Fatimah Mohamed",
    editor = "El-Haj, Mo  and
      Rayson, Paul  and
      Jarrar, Mustafa  and
      Ezeani, Ignatius  and
      Ezzini, Saad  and
      Ahmadi, Sina  and
      Haddad Haddad, Amal  and
      Amol, Cynthia  and
      Abdelali, Ahmad  and
      Abudalfa, Shadi",
    booktitle = "Proceedings of the 2nd Workshop on {NLP} for Languages Using {A}rabic Script",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "[https://aclanthology.org/2026.abjadnlp-1.20/](https://aclanthology.org/2026.abjadnlp-1.20/)",
    doi = "10.18653/v1/2026.abjadnlp-1.20",
    pages = "137--143",
    abstract = "This paper describes Tashkees-AI, a system developed for the AbjadMed 2026 Shared Task on Arabic Medical Question Classification. A comprehensive empirical study was conducted across 82 fine-grained categories, investigating three paradigms: fine-tuned encoder models, hierarchical classification, and ensemble methods. Leveraging a dataset of 27k Arabic medical question-answer pairs, an extensive ablation studies was conducted, comparing MARBERTv2, CAMeLBERT, two-stage hierarchical classifiers, and RAG-based approaches. The findings reveal that fine-tuned MARBERTv2 with data cleaning yields the best performance, achieving a macro F1-score of 0.3659 on the blind test set. In contrast, hierarchical methods surprisingly underperformed (0.332 F1) due to error propagation. The system ranked 26th on the official leaderboard."
}

```
 
 
