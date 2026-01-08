# EEG Alcoholism Detection using Deep Learning 🧠⚡

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project designed to classify EEG signals to distinguish between **Alcoholic** and **Control** subjects. This repository implements and compares hybrid neural network architectures (**CNN-LSTM** and **CNN-GRU**) to analyze multi-channel brainwave sensor data.

## 📂 Dataset

The dataset used in this project is the **EEG Database Data**, which contains measurements from 64 electrodes placed on the scalp, sampled at 256 Hz.

- **Dataset Link:** [Alcoholics EEG Dataset (Kaggle)](https://www.kaggle.com/datasets/nnair25/Alcoholics)
- **Source:** Large Scale Movie Description Challenge (SMNI_CMI_TRAIN/TEST).
- **Structure:**
  - **Subjects:** Categorized into 'a' (Alcoholic) and 'c' (Control).
  - **Sensors:** 64 channels (e.g., FP1, FP2, F7, F8).
  - **Time Steps:** 256 samples per trial (1 second at 256 Hz).

## 🛠️ Project Workflow

### 1. Data Preprocessing
Raw EEG CSV data is transformed into a structured 3D format suitable for time-series classification models.
- **Grouping:** Data is grouped by `trial_id` (a combination of Subject Name and Trial Number).
- **Pivoting:** Each trial is converted into a matrix of shape `(256, 64)`.
  - **Rows:** 256 Time Steps.
  - **Columns:** 64 Sensor Channels.
- **Label Encoding:**
  - `1`: Alcoholic Group
  - `0`: Control Group

### 2. Model Architectures
We explore two hybrid architectures that combine **Convolutional Neural Networks (CNN)** for spatial feature extraction with **Recurrent Neural Networks (RNN)** for temporal sequence learning.

#### **Model A: CNN-LSTM**
This model uses Long Short-Term Memory units to capture long-range dependencies in the EEG signals.
- **Input Shape:** `(256, 64)`
- **Layers:** Conv1D (64 filters) → BatchNormalization → MaxPooling → LSTM (64 units) → Dense (32 units) → Output (Sigmoid).

#### **Model B: CNN-GRU**
This model utilizes Gated Recurrent Units, which are computationally more efficient than LSTMs while effectively handling temporal data.
- **Input Shape:** `(256, 64)`
- **Layers:** Conv1D (64 filters) → BatchNormalization → MaxPooling → GRU (64 units) → Dense (32 units) → Output (Sigmoid).

## 📊 Results & Performance

Both models were trained for **50 epochs** using the `Adam` optimizer and `binary_crossentropy` loss function.

| Model Architecture | Training Accuracy | Validation Accuracy |
|-------------------|-------------------|---------------------|
| **CNN-LSTM** | ~98%              | ~83%                |
| **CNN-GRU** | ~86%              | ~72%                |

*Observation: The CNN-LSTM architecture generally outperformed the GRU variant in detecting the complex patterns associated with alcoholism in EEG scans.*

## 🚀 Installation & Usage

### Prerequisites
* Python 3.x
* Jupyter Notebook

### Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/EEG-Alcoholism-Detection.git](https://github.com/your-username/EEG-Alcoholism-Detection.git)
   cd EEG-Alcoholism-Detection

```

2. **Install dependencies:**
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib

```


3. **Prepare the Data:**
* Download the dataset from the Kaggle link above.
* Extract the files into a folder named `SMNI_CMI_TRAIN` (or update the path in the notebook).


4. **Run the analysis:**
* Open `main.ipynb` in Jupyter Notebook.
* Execute the cells to load data, train models, and view the visualization plots.



## 📈 Visualizations

The notebook includes plotting functions to visualize:

* **Model Accuracy:** Comparison of training vs. validation accuracy over epochs.
* **Model Loss:** Tracking loss reduction over time to check for overfitting.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
