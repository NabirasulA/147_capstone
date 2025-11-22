# 📌 **Misbehavior detection in VANET using Deep Learning**

This project implements a complete deep learning–based misbehavior detection pipeline for the VEREMI dataset, used in VANET (Vehicular Ad Hoc Network) security analysis.
The system uses neural networks — MLP and CNN-1D — along with an additional XGBoost baseline model for comparison.

Advanced interpretability methods such as SHAP and LIME are used to provide Explainable AI (XAI) insights into how the deep learning models make decisions.

---

## 🚀 **Project Highlights**

* Full preprocessing pipeline
  ✔ Data Cleaning
  ✔ Data Normalization
  ✔ Data Transformation
* Parallel training of three models:

  * **MLP**
  * **CNN-1D**
  * **XGBoost**
* Evaluation Metrics:

  * Accuracy
  * 
  * Confusion Matrix
  * ROC Curve
  
* Explainability (XAI):

  * SHAP Summary Plots
  * LIME Instance-Level Explanations
* All results automatically saved to the `/results` directory

---

## 📂 **Project Structure**

```
Capstone-147/
├── main.py
├── main_improved.py
├── npz.py
├── npz_reduce.py
├── train_raw_models.py
│
├── src/
│   ├── data_preprocessing/
│   ├── evaluation/
│   ├── explainability/
│   ├── models_raw/
│
├── results/
│   ├── logs/
│   ├── models/
│   ├── confusion_matrices/
│   ├── shap_output/
│
└── README.md
```

---

## 📊 **Dataset: VEREMI**

The **VEREMI dataset** is a well-known benchmark for detecting malicious behavior in Vehicular Ad Hoc Networks (VANETs).
It contains labeled entries:

* **0 → Legitimate node**
* **1 → Misbehaving node**

Large `.csv` files are converted into efficient `.npz` format using:

```
python npz.py
```

Dataset files are **excluded from GitHub using `.gitignore`**.

---

## 🛠️ **Installation**

### 1️⃣ Create a Conda environment

```bash
conda create -n vanet python=3.10
conda activate vanet
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Install XGBoost (if not included):

```bash
pip install xgboost
```

---

## 🧹 **Data Preprocessing**

Preprocessing includes:

* Removing invalid entries
* Normalizing features
* Transforming dataset into trainable format

Generate `.npz`:

```
python npz.py
```

---

## 🤖 **Training the Models**

### **Train MLP + CNN-1D (Raw Models)**

```
python main.py --cache_npz veremi_binary_1m.npz --epochs 20 --batch_size 512
```

### **Train XGBoost**

```
python main_improved.py --model xgb --cache_npz veremi_binary_1m.npz
```

All outputs are saved to:

```
results/
```

---

## 📈 **Evaluation Metrics**

For each model, the following are generated:

* Accuracy score
* Macro F1 & Weighted F1
* Precision/Recall
* Confusion Matrix (PNG)
* ROC Curve (PNG)
* sklearn classification report

Example files:

```
results/confusion_matrices/mlp_cm.png
results/confusion_matrices/cnn1d_cm.png
results/confusion_matrices/xgb_cm.png
```

---

## 🧠 **Explainability (XAI)**

### SHAP

Generates global feature importance using SHAP values:

```
python explain_raw_models.py
```

### LIME

Explains predictions for a specific test instance:

```
python run_lime.py
```

Output files include:

* `shap_mlp.png`
* `lime_output.png`

---

## 🧩 **System Architecture Overview**

The architecture includes:

* Dataset ingestion
* Preprocessing
* Parallel training of MLP, CNN-1D, XGBoost
* Unified evaluation pipeline
* Explainability layer (SHAP + LIME)

(Architecture diagram included separately in repo.)

---

## ⚙️ **Tech Stack**

| Component      | Technology                           |
| -------------- | ------------------------------------ |
| Language       | Python 3.10                          |
| ML Frameworks  | TensorFlow, XGBoost                  |
| Explainability | SHAP, LIME                           |
| Visualization  | Matplotlib, Seaborn                  |
| Environment    | Conda                                |
| Dataset        | VEREMI (VANET Misbehavior Detection) |

---

## 🧪 **Results Summary**

| Model   | Accuracy (Approx.) | Notes                 |
| ------- | ------------------ | --------------------- |
| MLP     | ~52–53%            | Balanced baseline     |
| CNN-1D  | ~52–53%            | Similar to MLP        |
| XGBoost | ~58%               | Best performing model |

---

## 👨‍💻 **Developed By**

**Nabirasul A**
B.Tech – Computer Science Engineering
Capstone Project 147

---

## 📄 **License**

This repository is for **academic and research use only**.

---

If you want, I can also provide:

✔ Enhanced GitHub banner
✔ Shields badges (Python version, last commit, stars)
✔ A better architecture diagram (HD horizontal)
✔ A professional project PDF for submission

Just tell me!

