# EEG Alcoholism Detection 🧠⚡

A Deep Learning project that classifies EEG signals to distinguish between **Alcoholic** and **Control** subjects. This repository implements hybrid neural network architectures (**CNN-LSTM** and **CNN-GRU**) to analyze multi-channel sensor data.

## 📂 Dataset
The dataset used in this project is the **EEG Database Data** sourced from Kaggle.
* **Link:** [Alcoholics EEG Dataset (Kaggle)](https://www.kaggle.com/datasets/nnair25/Alcoholics)
* **Source:** The original data contains measurements from 64 electrodes placed on the scalp sampled at 256 Hz.

## 🛠️ Project Workflow

### 1. Data Preprocessing
The raw CSV files are processed to create a 3D feature matrix suitable for time-series classification.
* **Grouping:** Data is grouped by unique `trial_id` (Subject Name + Trial Number).
* **Pivoting:** Each trial is transformed into a matrix of shape `(256, 64)`.
    * **Rows:** 256 Time Steps (Sample Num)
    * **Columns:** 64 Channels (Sensor Positions)
* **Labels:** * `0`: Control Group
    * `1`: Alcoholic Group

### 2. Model Architectures
Two hybrid deep learning models were trained to capture both spatial features (using CNNs) and temporal dynamics (using RNNs).

#### **Model A: CNN-LSTM**
* **Input:** (256 time steps, 64 channels)
* **Conv1D Layer:** 64 filters, Kernel size 3 (Feature Extraction)
* **Batch Normalization & Max Pooling**
* **LSTM Layer:** 64 units (Sequence Learning)
* **Dense Layers:** 32 units (ReLU) -> 1 unit (Sigmoid)

#### **Model B: CNN-GRU**
* **Input:** (256 time steps, 64 channels)
* **Conv1D Layer:** 64 filters, Kernel size 3
* **Batch Normalization & Max Pooling**
* **GRU Layer:** 64 units (Efficient Temporal Processing)
* **Dense Layers:** 32 units (ReLU) -> 1 unit (Sigmoid)

### 3. Performance
Both models were trained for 50 epochs using the `Adam` optimizer and `binary_crossentropy` loss.

| Model | Training Accuracy | Validation Accuracy |
|-------|-------------------|---------------------|
| **CNN-LSTM** | ~98% | ~84% |
| **CNN-GRU** | ~86% | ~72% |

*Note: The CNN-LSTM model showed superior performance in capturing the complex patterns of the EEG signals.*

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/EEG-Alcoholism-Detection.git](https://github.com/your-username/EEG-Alcoholism-Detection.git)
   cd EEG-Alcoholism-Detection
