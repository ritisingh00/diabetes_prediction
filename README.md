# 🩺 Diabetes Prediction System

A machine learning project to predict the onset of diabetes using clinical data. Built during a **45-day Data Science Internship at R3 Systems, Nashik**.

---

## 📌 Problem Statement

Diabetes is one of the most prevalent chronic diseases globally. Early prediction using patient data can help in timely medical intervention. This project builds and compares multiple ML classification models to predict whether a patient is diabetic based on diagnostic measurements.

---

## 📊 Dataset

- **Source:** [PIMA Indians Diabetes Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 patients
- **Features:** 8 medical attributes (Glucose, BMI, Age, Blood Pressure, etc.)
- **Target:** Outcome (1 = Diabetic, 0 = Non-Diabetic)

---

## ⚙️ Models Compared

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~72% |
| Support Vector Machine (SVM) | ~73% |
| **Random Forest** ✅ | **74.04%** |

> **Random Forest** achieved the best performance and was selected as the final model.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat)

---

## 📁 Project Structure

```
diabetes-prediction/
│
├── diabetes.csv               # Dataset
├── diabetes_prediction.ipynb  # Main Jupyter Notebook
├── requirements.txt           # Dependencies
└── README.md
```

---

## 🔍 Project Workflow

1. **Data Loading & Exploration** — shape, dtypes, null values, statistical summary
2. **EDA** — distribution plots, correlation heatmap, class balance check
3. **Data Preprocessing** — handling zero values, feature scaling
4. **Model Training** — Logistic Regression, SVM, Random Forest
5. **Model Evaluation** — Accuracy, Confusion Matrix, Classification Report
6. **Comparison & Conclusion** — Random Forest selected as best model

---

## 📈 Key Visualizations

- Correlation heatmap of all features
- Distribution plots for Glucose, BMI, Age
- Confusion matrices for all 3 models
- Model accuracy comparison bar chart

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/ritisingh00/diabetes-prediction.git
cd diabetes-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook diabetes_prediction.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 👩‍💻 Author

**Riti Singh**  
B.E. Computer Science | Matoshri College of Engineering, Nashik  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/riti-singh-54167a293)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/ritisingh00)

---

> 💡 *Built during 45-day internship at R3 Systems, Nashik (2024)*
