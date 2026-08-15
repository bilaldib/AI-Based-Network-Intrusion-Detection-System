# AI-Based Network Intrusion Detection System using Machine Learning

An AI-based Network Intrusion Detection System (NIDS) designed to detect and classify network traffic as normal or malicious using Machine Learning and Deep Learning techniques.

The project explores multiple machine learning approaches and compares their performance for network intrusion detection, with particular attention to imbalanced attack classes such as **R2L** and **U2R**.

## 🎯 Project Objectives

The main objectives of this project are to:

* Detect malicious network traffic.
* Classify network connections into different attack categories.
* Compare classical Machine Learning algorithms with Deep Learning.
* Analyze the impact of class imbalance on intrusion detection.
* Improve minority-class detection using techniques such as SMOTE.
* Perform detailed error analysis.
* Compare model performance using F1-score and other classification metrics.

## 🧠 Machine Learning Approaches

Several approaches were implemented and evaluated:

| Approach                 |  F1-Macro | Description                                     |
| ------------------------ | --------: | ----------------------------------------------- |
| Random Forest / Baseline |         — | Initial tree-based approach                     |
| XGBoost Baseline         |     0.543 | Gradient boosting baseline                      |
| XGBoost + Class Weights  |     0.587 | Addresses class imbalance using class weighting |
| XGBoost + SMOTE          | **0.608** | 🏆 Best overall result                          |
| MLP / Deep Learning      |     0.579 | Neural network implemented with TensorFlow      |

The results show that **SMOTE provided the best overall F1-macro score (0.608)**.

The MLP achieved an F1-macro score of **0.579** and showed different behavior across attack categories. In particular, it performed relatively better on **R2L**, while its performance on **U2R** remained challenging.

## 📊 Dataset

The project uses the **NSL-KDD** dataset for network intrusion detection.

The dataset contains network connections labeled as normal traffic or different types of attacks.

Main files:

```text
data/
├── KDDTrain+.txt
└── KDDTest+.txt
```

The attack categories analyzed in the project include:

* Normal
* DoS
* Probe
* R2L
* U2R

## 🔬 Methodology

The project follows the following workflow:

```text
NSL-KDD Dataset
       │
       ▼
Data Loading
       │
       ▼
Data Preprocessing
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering / Encoding
       │
       ▼
Train / Test
       │
       ├───────────────┐
       ▼               ▼
Machine Learning   Deep Learning
       │               │
       ▼               ▼
Random Forest      MLP
XGBoost            TensorFlow
       │               │
       └───────┬───────┘
               ▼
        Model Evaluation
               │
               ▼
        Error Analysis
               │
               ▼
       Model Comparison
```

## ⚖️ Class Imbalance

One of the main challenges of intrusion detection is the imbalance between attack classes.

Some attack categories, especially **R2L and U2R**, contain significantly fewer samples than majority classes.

To address this problem, the project evaluates:

* Class weighting
* SMOTE (Synthetic Minority Over-sampling Technique)
* Per-class F1-score
* Confusion matrices
* Error analysis

## 🏆 Results

The main comparison obtained the following results:

| Model / Technique       |     F1-Macro |
| ----------------------- | -----------: |
| XGBoost Baseline        |        0.543 |
| XGBoost + Class Weights |        0.587 |
| XGBoost + SMOTE         | **0.608** 🏆 |
| MLP (TensorFlow)        |        0.579 |

### Key Findings

**XGBoost + SMOTE** achieved the best overall F1-macro score.

The experiments also demonstrated that different algorithms behave differently depending on the attack category.

The MLP showed relatively strong behavior on **R2L**, while **U2R** remained difficult to classify accurately.

This highlights the importance of evaluating intrusion detection models not only using global metrics, but also using **per-class performance and error analysis**.

## 📈 Experimental Analysis

The repository contains visualizations and analysis results including:

* Class distribution
* Protocol/service analysis
* Correlation matrix
* Random Forest confusion matrix
* Feature importance
* Random Forest vs XGBoost comparison
* Multi-class confusion matrices
* Normalized confusion matrices
* F1-score per class
* R2L train/test analysis
* R2L confidence errors
* Improvement comparison
* MLP training curves
* MLP confusion matrix
* Final model comparison

These results are available in:

```text
notebooks/
```

## 📁 Project Structure

```text
AI-Based-Network-Intrusion-Detection-System/
│
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── notebooks/
│   ├── 01_class_distribution.png
│   ├── 02_protocol_service.png
│   ├── 03_correlation_matrix.png
│   ├── ...
│   ├── 16_final_comparison_all_models.png
│   ├── final_model_comparison.csv
│   ├── improvement_comparison.csv
│   └── model_comparison.csv
│
├── src/
│   ├── app.py
│   ├── eda.py
│   ├── error_analysis.py
│   ├── improve_model.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_mlp.py
│   ├── train_model.py
│   ├── train_multiclass.py
│   └── train_xgboost.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-tf.txt
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/bilaldib/AI-Based-Network-Intrusion-Detection-System.git
cd AI-Based-Network-Intrusion-Detection-System
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install the main dependencies

```bash
pip install -r requirements.txt
```

## 🧠 TensorFlow / MLP Environment

The MLP experiment uses TensorFlow and is maintained in a separate environment.

Create the environment:

```powershell
py -3.13 -m venv venv_tf
```

Activate it:

```powershell
venv_tf\Scripts\Activate.ps1
```

Install TensorFlow dependencies:

```powershell
pip install -r requirements-tf.txt
```

## 🚀 Running the Project

### Exploratory Data Analysis

```bash
python src/eda.py
```

### Train the baseline model

```bash
python src/train_model.py
```

### Train XGBoost

```bash
python src/train_xgboost.py
```

### Multi-class classification

```bash
python src/train_multiclass.py
```

### Model improvement

```bash
python src/improve_model.py
```

### Error analysis

```bash
python src/error_analysis.py
```

### Train the MLP

Activate the TensorFlow environment first:

```powershell
venv_tf\Scripts\Activate.ps1
```

Then:

```bash
python src/train_mlp.py
```

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* TensorFlow
* Keras
* Matplotlib
* Seaborn
* Git
* GitHub

## 🔐 Cybersecurity Focus

This project demonstrates the application of Artificial Intelligence to:

* Network Security
* Intrusion Detection
* Attack Classification
* Anomaly Detection
* Imbalanced Learning
* Security Data Analysis

## 🔮 Future Improvements

Potential future improvements include:

* Hyperparameter optimization
* Advanced feature selection
* More advanced neural network architectures
* Ensemble learning
* Real-time network traffic analysis
* Integration with packet capture tools such as Wireshark or Zeek
* Deployment as a real-time NIDS application
* Explainable AI for intrusion detection
* Evaluation on additional cybersecurity datasets

## 👨‍💻 Author

**Bilal Dib**

GitHub: https://github.com/bilaldib

## 📄 License

This project is intended for educational, research, and cybersecurity learning purposes.
