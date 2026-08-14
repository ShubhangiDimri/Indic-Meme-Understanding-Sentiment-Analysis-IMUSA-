# 🧩 Indic Meme Understanding & Sentiment Analysis (IMUSA) — FIRE 2026

A multimodal machine learning system for **Punjabi meme sentiment classification**, developed for the **Indic Meme Understanding & Sentiment Analysis (IMUSA)**.

The project investigates whether combining **Punjabi textual information with visual meme features** improves sentiment classification compared with text-only and image-only approaches.

---

## 🎯 Task Overview

Punjabi memes communicate sentiment through a combination of **textual content and visual context**. Understanding either modality independently may not be sufficient, particularly for sarcasm and culturally contextual content.

The task is formulated as a **4-class classification problem**:

- 😏 **Sarcasm**
- 😐 **Neutral**
- ⚠️ **Offensive**
- 💪 **Motivational**

---

## 📊 Dataset

The IMUSA dataset contains Punjabi memes along with their textual content.

| Dataset | Samples |
|---|---:|
| Training Dataset | 3,002 |
| Official Test Dataset | 500 |

For model development, the labeled training dataset was divided using a **stratified 80/20 train-validation split**.

| Development Split | Samples |
|---|---:|
| Training | 2,402 |
| Validation | 600 |

A fixed random seed of `42` was used for reproducibility.

> **Note:** The original IMUSA dataset, meme images, and official test data are not redistributed through this repository.

---

## ⚖️ Class Distribution

The training split was highly imbalanced.

| Class | Training Samples |
|---|---:|
| Sarcasm | 1,063 |
| Motivational | 696 |
| Neutral | 601 |
| Offensive | 42 |

To address this imbalance, **class-weighted cross-entropy loss** was used during training.

Approximate class weights:

```text
Motivational : 0.8628
Neutral      : 0.9992
Offensive    : 14.2976
Sarcasm      : 0.5649
```

---

## 🧠 Methodology

We developed and compared three primary systems:

1. **MuRIL — Text-only baseline**
2. **CLIP — Image-only baseline**
3. **MuRIL + CLIP — Multimodal fusion**

A probability-level ensemble was also evaluated.

---

## 1. MuRIL — Text-Only Baseline

The Punjabi meme text is encoded using `google/muril-base-cased`.

### Architecture

```text
Punjabi Text
     │
     ▼
MuRIL Tokenizer
     │
     ▼
   MuRIL
     │
     ▼
Text Representation
     │
     ▼
Classification Head
     │
     ▼
4 Sentiment Classes
```

### Training Configuration

- **Model:** `google/muril-base-cased`
- **Maximum sequence length:** `128`
- **Optimizer:** AdamW
- **Learning rate:** `2e-5`
- **Batch size:** `16`
- **Epochs:** `4`
- **Loss:** Weighted Cross-Entropy
- **Primary metric:** Macro-F1

### Best Validation Result

| Metric | Score |
|---|---:|
| Accuracy | 0.6267 |
| Macro-F1 | **0.5336** |

---

## 2. CLIP — Image-Only Baseline

The image-only baseline uses the vision encoder from `openai/clip-vit-base-patch32`.

### Architecture

```text
Meme Image
    │
    ▼
CLIP Processor
    │
    ▼
CLIP ViT-B/32
    │
    ▼
768-D Visual Representation
    │
    ▼
Classification Head
    │
    ▼
4 Sentiment Classes
```

The pretrained CLIP vision encoder was frozen and a lightweight classification head was trained on top of its visual representation.

Approximately **198K parameters** were trainable during this experiment.

### Best Validation Result

| Metric | Score |
|---|---:|
| Accuracy | 0.6050 |
| Macro-F1 | **0.4991** |

---

## 3. MuRIL + CLIP Multimodal Fusion

Our primary system combines **Punjabi textual representations from MuRIL** with **visual representations from CLIP**.

### Architecture

```text
                 Punjabi Meme
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
          Text                Image
            │                   │
            ▼                   ▼
         MuRIL               CLIP ViT
            │                   │
            ▼                   ▼
    Text Embedding       Image Embedding
       (768-D)              (768-D)
            │                   │
            ▼                   ▼
    Text Projection      Image Projection
       (256-D)              (256-D)
            │                   │
            └─────────┬─────────┘
                      │
                      ▼
                Concatenation
                    (512-D)
                      │
                      ▼
                  Fusion MLP
                      │
                      ▼
                   4 Logits
                      │
                      ▼
             Sentiment Prediction
```

Both pretrained encoders were initially frozen while the projection and fusion layers were trained.

Approximately **526K parameters** were trainable in the multimodal fusion experiment.

### Best Validation Result

| Metric | Score |
|---|---:|
| Macro-F1 | **0.5743** |

The multimodal system outperformed both unimodal baselines.

---

## 🔀 Probability Ensemble

A probability-level ensemble was evaluated using the multimodal model and the MuRIL text-only model.

The final probability distribution was calculated as:

```text
P(final) = 0.95 × P(multimodal) + 0.05 × P(MuRIL)
```

The ensemble weight was selected using **validation data only**.

### Best Ensemble Result

| Metric | Score |
|---|---:|
| Accuracy | 0.6300 |
| Macro-F1 | **0.5762** |

This configuration achieved the highest validation Macro-F1 among the evaluated systems.

---

## 📈 Validation Results

| Model | Modality | Validation Macro-F1 |
|---|---|---:|
| CLIP ViT-B/32 | Image | 0.4991 |
| MuRIL | Text | 0.5336 |
| **MuRIL + CLIP Fusion** | **Text + Image** | **0.5743** |
| **95% Multimodal + 5% MuRIL Ensemble** | **Ensemble** | **0.5762** |

### Performance Progression

```text
CLIP Image-Only
Macro-F1 = 0.4991
        │
        ▼
MuRIL Text-Only
Macro-F1 = 0.5336
        │
        ▼
MuRIL + CLIP Multimodal
Macro-F1 = 0.5743
        │
        ▼
Probability Ensemble
Macro-F1 = 0.5762
```

The multimodal fusion model improved Macro-F1 by approximately **4.1 percentage points** over the MuRIL text-only baseline.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- MuRIL
- CLIP ViT-B/32
- scikit-learn
- pandas
- NumPy
- Pillow
- Google Colab
- NVIDIA T4 GPU








## ⚠️ Disclaimer

This repository is intended for **research and educational purposes**.

The models may make incorrect predictions, particularly for underrepresented categories such as **Offensive**, due to severe class imbalance in the available training data.
