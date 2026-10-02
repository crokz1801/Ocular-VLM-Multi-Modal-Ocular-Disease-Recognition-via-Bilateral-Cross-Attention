# 🩺 Dual-Eye Ocular Disease Recognition

A patient-level **multi-label ocular disease classification system** using **dual-eye retinal images**, **EfficientNet-B4**, **cross-eye attention**, **clinical metadata**, **image-quality features**, and **CLIP-guided training**.

The system is designed for the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset and predicts eight ocular disease categories from left/right eye images at the **patient level**.

---

## 🚀 Overview

Most retinal classification pipelines process an individual eye independently. This project instead treats both eyes as a **single patient-level sample**, allowing information from one eye to influence the representation of the other.

### Key ideas

- 👁️ **Dual-eye patient-level modeling**
- 🧠 **EfficientNet-B4** shared backbone
- 🔄 **Cross-eye attention** between left and right eyes
- 📊 **Clinical metadata + image-quality assessment (IQA)**
- 🎯 **CLIP-based semantic smoothing**
- ⚖️ **Class-imbalance aware loss**
- 🔥 **Fragility-aware training**
- 📉 **SWA (Stochastic Weight Averaging)**
- 🎚️ **Per-class threshold optimization**
- 🔍 **CLIP-based failure analysis**
- 🧪 **Strict classifier retraining (cRT)**
- 🔄 **8-view Test-Time Augmentation (TTA)**
- 🏥 **Multi-label prediction for 8 ocular conditions**

---

# 📌 Dataset

This project uses the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset.

### Dataset statistics

| Property | Value |
|---|---:|
| Patients | 6,392 |
| Training patients | 5,439 |
| Validation patients | 953 |
| Missing eye information | 324 |
| Number of classes | 8 |
| Input | Left + Right retinal images |
| Task | Multi-label classification |

The train/validation split is performed at the **patient level**, ensuring that the same patient does not appear in both splits.

```text
6,392 patients
      │
      ├── 5,439 Train
      │
      └──   953 Validation
```

### Classes

| Code | Disease |
|---|---|
| N | Normal |
| D | Diabetes |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertension |
| M | Myopia |
| O | Other |

---

# 🏗️ Model Architecture

```text
                   PATIENT
                      │
             ┌────────┴────────┐
             │                 │
        LEFT EYE           RIGHT EYE
             │                 │
             ▼                 ▼
     Ben Graham Prep     Ben Graham Prep
             │                 │
             ▼                 ▼
       EfficientNet-B4   EfficientNet-B4
          (shared)          (shared)
             │                 │
             ▼                 ▼
        100 tokens         100 tokens
             │                 │
             └────────┬────────┘
                      │
              Cross-Eye Attention
              ┌───────┴───────┐
              │               │
        L queries R      R queries L
              │               │
              └───────┬───────┘
                      │
                   512-d
                      │
        ┌─────────────┼─────────────┐
        │             │             │
    Metadata         IQA       CLIP features
      64-d           32-d       / semantic
        │             │          information
        └─────────────┴─────────────┘
                      │
                   Fusion
                   608-d
                      │
                      ▼
              Classification Head
                      │
                      ▼
              8 Disease Outputs
```

### Backbone

**EfficientNet-B4**

- Shared weights for left/right eyes
- Input resolution: **320 × 320**
- Feature representation converted into **100 patch tokens**
- Approximately **20.6M trainable parameters**

### Cross-Eye Attention

Instead of independently classifying each eye:

```text
Left representation  ─────► attends to ─────► Right
Right representation ─────► attends to ─────► Left
```

This allows the model to capture relationships between the two eyes of the same patient.

The resulting cross-eye representation is **512-dimensional**.

---

# 🧩 Additional Features

## Clinical Metadata

Patient-level metadata is incorporated into the final representation.

```text
Metadata → 64 dimensions
```

Examples include demographic information such as:

- Age
- Sex

---

## Image Quality Assessment

Image-quality features are included to provide additional information about image reliability.

```text
IQA → 32 dimensions
```

This helps the model account for differences in image quality such as visibility and image degradation.

---

# 🤖 CLIP Precomputation

CLIP embeddings are precomputed at the **patient level**.

For patients with both eyes:

```text
Left CLIP embedding
        +
Right CLIP embedding
        ↓
Average patient representation
```

The embeddings are then used for:

- Semantic smoothing during training
- Prototype construction
- Class similarity analysis
- Failure analysis
- Identifying potential failure factors

### Class similarity

The precomputed CLIP representations revealed strong semantic similarity between some classes.

For example:

```text
S(G, O) = 0.863
S(N, A) = 0.876
```

---

# 🔥 Fragility-Aware Learning

A major component of the training pipeline is identifying classes where the model is particularly fragile.

After warm-up training, class-wise recall and a fragility score are computed.

### Initial analysis

| Class | Recall | Fragility |
|---|---:|---:|
| N | 0.871 | 0.1990 |
| D | 0.478 | 0.5158 |
| G | 0.671 | 0.3870 |
| C | 0.852 | 0.3356 |
| A | 0.725 | 0.4152 |
| H | 0.270 | 0.7122 |
| M | 0.842 | 0.3352 |
| O | 0.416 | 0.5197 |

The initially identified fragile classes were:

```text
D, H, O
```

During later training, the fragility analysis converged to:

```text
H
```

as the remaining
