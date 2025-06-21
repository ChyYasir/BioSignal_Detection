# 🧠 Stacked Sparse Autoencoder (SSAE) for Feature Extraction and Classification

This repository contains a modular implementation of a **Stacked Sparse Autoencoder (SSAE)** with four encoding layers, designed for dimensionality reduction and classification tasks in Keras.

## 📌 Overview

The `build_ssae_model()` function builds a complete SSAE architecture that:

- Learns **compressed, sparse representations** of high-dimensional data
- Trains each layerwise **autoencoder** to reconstruct the input
- Stacks **only the encoder parts** to form a deep network
- Adds a **softmax classifier** on top for final prediction

---

## 🧱 Architecture Details

Each autoencoder in the stack compresses the input further:

| Autoencoder | Input Shape | Hidden Layer | Output Shape |
|-------------|-------------|---------------|---------------|
| AE1         | `input_dim` | 18 units      | `input_dim`   |
| AE2         | 18           | 12 units      | 18            |
| AE3         | 12           | 8 units       | 12            |
| AE4         | 8            | 6 units       | 8             |

The final classifier takes the 6-dimensional encoded vector and passes it through a **Dense softmax** layer to classify into `num_classes`.

---

## 🔍 Function: `build_ssae_model(input_dim, num_classes)`

### Parameters:
- `input_dim` *(int)*: Number of input features
- `num_classes` *(int)*: Number of output classes for classification

### Returns:
```python
final_model, autoencoder1, encoder1, autoencoder2, encoder2,
autoencoder3, encoder3, autoencoder4, encoder4
