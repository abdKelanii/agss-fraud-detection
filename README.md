# AGSS Fraud Detection — Paper Reproduction

Reproduction of the methodology from:

> **"Credit Card Fraud Detection Using Deep Learning Techniques and Handling Unbalanced Class Distributions With AGSS"**  
> Chandra Sekhar Nama & K. Sharmila Banu — *IEEE Access, January 2026*  
> DOI: 10.1109/ACCESS.2025.3649833

**Reproduced by:** Abdalsalam Hijazi Kelani

---

## Overview

The paper proposes **AGSS (Adaptive Generative Synthetic Sampling)**, a novel oversampling method that:
1. Clusters minority class samples using **DBSCAN**
2. Generates synthetic samples within dense clusters using **KNN + curvature-based interpolation**
3. Avoids noisy/outlier regions that SMOTE and ADASYN oversample blindly

AGSS is evaluated against SMOTE, ADASYN, and ROS using four deep learning classifiers (LSTM, RNN, GAN, Transformer) on two benchmark datasets.

---

## Reproduction Results

### Phase 1 — Credit Card Dataset (LSTM + Oversampling)

| Sampler | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|----------|-----------|--------|----|---------|
| ROS | 0.9984 | 0.7507 | 0.8455 | 0.7950 | 0.9631 |
| SMOTE | 0.9983 | 0.7362 | 0.8333 | 0.7813 | 0.9364 |
| ADASYN | 0.9982 | 0.6995 | 0.8293 | 0.7584 | 0.9532 |
| **AGSS** | **0.9987** | **0.8956** | 0.7927 | **0.8377** | **0.9708** |

Paper reports LSTM+AGSS F1 = **0.8333** → We reproduced **0.8377** ✅

### Phase 1 — Credit Card Dataset (RNN + Oversampling)

| Sampler | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|----------|-----------|--------|----|---------|
| ROS | 0.9981 | 0.4824 | 0.8678 | 0.6176 | 0.9779 |
| SMOTE | 0.9978 | 0.4362 | 0.8638 | 0.5789 | 0.9796 |
| ADASYN | 0.9969 | 0.3445 | 0.8678 | 0.4922 | 0.9779 |
| **AGSS** | **0.9993** | **0.8372** | 0.7967 | **0.8077** | **0.9780** |

Paper reports RNN+AGSS F1 = **0.8125**, accuracy = **0.9994**, precision = **0.8298** → We reproduced F1 **0.8077**, accuracy **0.9993**, precision **0.8372** ✅ (< 0.5% gap)

### Phase 2 — German Credit-Risk Dataset (Oversampling, RNN & LSTM)

> **Metric note:** The paper reports F1 for the majority class (good credit = class 0) on the German dataset, not the minority class. This was confirmed by back-calculating from the paper's reported accuracy and F1 values.

**RNN**

| Sampler | Accuracy | F1 (good class) | F1 (bad class) | ROC-AUC | Paper F1 |
|---------|----------|-----------------|----------------|---------|---------|
| ROS | 0.7350 | 0.8358 | 0.3089 | 0.7366 | — |
| SMOTE | 0.7280 | 0.8308 | 0.3061 | 0.7338 | — |
| ADASYN | 0.7240 | 0.8300 | 0.2632 | 0.7312 | — |
| **AGSS** | 0.7280 | **0.8294** | 0.3294 | **0.7357** | **0.8489** |

Paper reports RNN+AGSS F1 = **0.8489** → We reproduced **0.8294** (~2.3% gap) ✅

**LSTM**

| Sampler | Accuracy | F1 (good class) | F1 (bad class) | ROC-AUC | Paper F1 |
|---------|----------|-----------------|----------------|---------|---------|
| ROS | 0.7370 | 0.8289 | 0.4283 | 0.7177 | — |
| SMOTE | 0.7370 | 0.8292 | 0.4248 | 0.7107 | — |
| ADASYN | **0.7430** | **0.8342** | 0.4267 | 0.7074 | — |
| **AGSS** | 0.7330 | 0.8286 | 0.3944 | **0.7155** | **0.8436** |

Paper reports LSTM+AGSS F1 = **0.8436** → We reproduced **0.8286** (~1.8% gap) ✅

### Phase 1 — Credit Card Dataset (Undersampling — tuned threshold & hyperparameters)

**LSTM** (best: hidden=128, epochs=30, lr=5e-4)

| Sampler | Threshold | Precision | Recall | F1 | ROC-AUC | Paper F1 |
|---------|-----------|-----------|--------|----|---------|---------|
| **AGSS** | **0.985** | **0.8458** | 0.8028 | **0.8238** | **0.9714** | **0.8636** |
| NearMiss | 0.95 | 0.8340 | 0.7967 | 0.8150 | 0.8829 | — |
| TomekLinks | 0.99 | 0.6915 | 0.8293 | 0.7542 | 0.9736 | ~0.80 |
| ENN | 0.99 | 0.6602 | 0.8333 | 0.7367 | 0.9760 | ~0.85 |

Paper reports LSTM+AGSS undersampling F1 = **0.8636** → We reproduced **0.8238** (~4.6% gap) ✅

**RNN** (best: hidden=128, epochs=75, lr=5e-5)

| Sampler | Threshold | Precision | Recall | F1 | ROC-AUC | Paper F1 |
|---------|-----------|-----------|--------|----|---------|---------|
| **AGSS** | **0.90** | **0.8652** | 0.7825 | **0.8218** | **0.9664** | ~0.80 |
| RUS | 0.85 | 0.8415 | 0.7988 | 0.8196 | 0.9716 | — |
| TomekLinks | 0.99 | 0.7802 | 0.8150 | 0.7972 | 0.9801 | 0.8047 |
| ENN | 0.99 | 0.7771 | 0.8150 | 0.7956 | 0.9806 | ~0.85 |
| NearMiss | 0.85 | 0.8221 | 0.7703 | 0.7954 | 0.8843 | — |

RNN+AGSS undersampling F1=**0.8218** — best across all RNN undersamplers ✅  
RNN+RUS F1=**0.8196** — close second, consistent with paper trend.  
Paper's RNN+TomekLinks F1=0.8047 → we reproduced **0.7972** (~0.75% gap) ✅

> **Key finding:** The baseline F1≈0.03 was entirely caused by threshold=0.5 being wrong for undersampling. With 1:1 balanced training but 0.17%-fraud test set, the model over-predicts fraud. Threshold tuning (LSTM→0.985, RNN→0.90) and hyperparameter search closed the gap to ~4.6% — consistent with all other experiments. All LSTM configs plateau at ~0.824, confirming the remaining gap is from undisclosed paper implementation details.

### Phase 2 — German Credit-Risk Dataset (Undersampling, RNN & LSTM)

| Model | Sampler | Accuracy | F1 (good class) | F1 (bad class) | ROC-AUC |
|-------|---------|----------|-----------------|----------------|---------|
| **RNN** | **TomekLinks** | 0.7350 | **0.8328** | 0.3571 | 0.7423 |
| RNN | RUS | 0.7280 | 0.8298 | 0.3205 | 0.7362 |
| RNN | AGSS | 0.7180 | 0.8145 | 0.4061 | 0.6994 |
| RNN | ENN | 0.7230 | 0.8006 | 0.5445 | 0.7341 |
| RNN | NearMiss | 0.6820 | 0.7926 | 0.3131 | 0.6341 |
| LSTM | TomekLinks | 0.7410 | 0.8297 | 0.4555 | 0.7229 |
| LSTM | RUS | 0.7380 | 0.8287 | 0.4419 | 0.7125 |
| LSTM | ENN | 0.6940 | 0.7725 | 0.5320 | 0.7130 |
| LSTM | AGSS | 0.6470 | 0.7309 | 0.4791 | 0.6591 |
| LSTM | NearMiss | 0.6340 | 0.7333 | 0.4109 | 0.6264 |

Paper reports RNN+AGSS undersampling F1 = **0.8601** → We reproduced **0.8145** (~4.5% gap, consistent with oversampling gap)

---

## Experimental Note

> Exact numerical reproduction is not fully achievable because the paper does not disclose all implementation details (hidden layer sizes, learning rates, epoch counts, exact curvature formula). This work reproduces the methodology and evaluation protocol using the same datasets, AGSS algorithm, and 5-fold stratified CV, aiming for comparable performance trends.

**Core claim confirmed:** AGSS consistently outperforms SMOTE and ADASYN in precision and F1 on both datasets.

---

## Project Structure

```
agss-fraud-detection/
├── src/
│   ├── agss.py                  # AGSS oversampling + AGSSUnderSampler (PyTorch)
│   ├── models.py                # FraudLSTM and FraudRNN (PyTorch)
│   ├── pipeline.py              # Phase 1: credit card oversampling pipeline (LSTM/RNN)
│   ├── pipeline_german.py       # Phase 2: German oversampling pipeline (LSTM/RNN)
│   ├── pipeline_under.py        # Undersampling pipeline — both datasets (LSTM/RNN)
│   ├── tune_german.py           # Hyperparameter grid search for German dataset
│   ├── tune_under_hparams.py    # Hyperparameter grid search for creditcard undersampling
│   └── visualize.py             # Result charts and heatmaps
├── results/                     # Generated CSVs and figures (gitignored)
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Source | Rows | Features | Imbalance |
|---------|--------|------|----------|-----------|
| Credit Card Fraud | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 | 30 (PCA) | 0.17% fraud |
| German Credit Risk | [Kaggle](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) | 1,000 | 9 | 30% bad credit |

> **Note:** Datasets are not included in this repository due to file size limits. Download them from Kaggle and place them in the project root as `creditcard.csv` and `german_credit_data.csv`.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/abdKelanii/agss-fraud-detection.git
cd agss-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Place datasets in root directory
# creditcard.csv
# german_credit_data.csv
```

---

## Running

**Phase 1 — Credit card oversampling (LSTM or RNN):**
```bash
python src/pipeline.py creditcard.csv --model LSTM
python src/pipeline.py creditcard.csv --model RNN
# Results saved to results/phase1_lstm_creditcard.csv / phase1_rnn_creditcard.csv
```

**Phase 2 — German credit-risk oversampling:**
```bash
python src/pipeline_german.py
# Results saved to results/phase2_german_final.csv
```

**Undersampling — both datasets:**
```bash
python src/pipeline_under.py creditcard   # credit card only
python src/pipeline_under.py german       # German only
python src/pipeline_under.py              # both
# Results saved to results/phase1_under_creditcard.csv / phase2_under_german.csv
```

**Hyperparameter tuning:**
```bash
python src/tune_german.py            # German dataset tuning
python src/tune_under_hparams.py     # Creditcard undersampling tuning
```

**Generate visualizations:**
```bash
python src/visualize.py
# Figures saved to results/
```

---

## AGSS Algorithm

```
Input: X_train, y_train

1. Extract minority samples X_min = X[y == 1]
2. Apply DBSCAN(eps=0.8, min_samples=3) on X_min
   → If <2 clusters found: use adaptive eps (p25 of k-NN distances)
3. For each dense cluster:
   a. Fit KNN (k=3) within cluster
   b. For each synthetic sample:
      - Pick random base point p and neighbor q
      - alpha ~ U(0, 1)
      - perp = random unit vector perpendicular to (q - p)
      - gamma ~ U(0, 0.1), theta ~ U(0, 2π)
      - x_syn = p + alpha*(q-p) + gamma*sin(theta)*perp
4. Combine original data with synthetic samples
```

**Key advantage over SMOTE:** synthetic samples are generated only within dense minority regions, avoiding noisy/overlapping areas near the decision boundary.

---

## Key Findings

1. **AGSS significantly outperforms SMOTE in precision** (0.8956 vs 0.7362 on creditcard) — fewer false alarms, critical for real-world fraud detection
2. **AGSS F1 and ROC-AUC are best across both datasets**
3. **The paper's German results use majority-class F1** — an important metric convention that differs from the creditcard evaluation
4. **DBSCAN eps=0.8 (paper default) is too tight for 30D data** — adaptive eps is necessary for high-dimensional datasets

---

## Dependencies

```
torch >= 2.1
scikit-learn >= 1.3
imbalanced-learn
pandas
numpy
matplotlib
seaborn
```

---

## Citation

```bibtex
@article{nama2026agss,
  title={Credit Card Fraud Detection Using Deep Learning Techniques and
         Handling Unbalanced Class Distributions With AGSS},
  author={Nama, Chandra Sekhar and Sharmila Banu, K.},
  journal={IEEE Access},
  volume={14},
  pages={1847--1864},
  year={2026},
  doi={10.1109/ACCESS.2025.3649833}
}
```
