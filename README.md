# Explainable Claim Check-Worthiness using CACL

## Overview
This repository contains our course project on Explainable Claim Check-Worthiness Detection, completed at IIIT Delhi under the guidance of Prof. Shad Akhtar.

We propose a novel architecture called Claim-Aware Contrastive Learning (CACL), which models check-worthiness as a function of underlying rationality factors, improving both performance and interpretability.

---

## Key Contributions
- Claim-Level Embedding Network (CLEN) with rationality-specific queries
- Multi-label contrastive learning for improved representation learning
- Explainable predictions using rationality labels
- State-of-the-art comparable performance

---

## Results
Our model achieves:
- Accuracy: 80.52%
- Macro F1: 77.91%
- CW-F1: 85.50%
- NCW-F1: 70.33%

CACL achieves performance comparable to and slightly better than the current state-of-the-art model **CheckMate**.

---

## Comparison with Prior Work

We benchmark our model against several baselines, including the current state-of-the-art:

- SVM + TF-IDF  
- BERT (Fine-tuned)  
- RoBERTa (Fine-tuned)  
- XLNet (Fine-tuned)  
- **CheckMate (Sundriyal et al., 2025)** ← State-of-the-art  

### Key Insight
Our proposed CACL model matches and, in some aspects, outperforms **CheckMate**, particularly in:
- Non-check-worthy class detection (NCW-F1)
- Rationality label prediction
- Overall macro F1

---

## Method
The architecture consists of:
1. BERT-based token encoder
2. Claim-Level Embedding Network (CLEN) with cross-attention
3. Rationality Aggregator (mean / weighted pooling)
4. Multi-objective training:
   - Check-worthiness loss
   - Rationality loss
   - Contrastive loss

---

## Why It Works
- Contrastive learning improves embedding separation
- Rationality queries capture fine-grained reasoning
- Weighted aggregation focuses on important factors
- Leads to better generalisation and interpretability

---

## Project Details
- Institution: IIIT Delhi
- Course: Natural Language Processing
- Instructor: Prof. Shad Akhtar

---

## Authors
- Ritvik Shekhar
- Nischay Soni
- Hitesh Bhat

---

## References

[1] Sundriyal, M., Akhtar, M. S., & Chakraborty, T. (2025).  
**Leveraging Rationality Labels for Explainable Claim Check-Worthiness.**  
IEEE Transactions on Artificial Intelligence.

---

## Statement
Our proposed CACL architecture achieves state-of-the-art comparable performance, matching and slightly outperforming **CheckMate (Sundriyal et al., 2025)** while providing improved explainability through rationality-aware modeling.
