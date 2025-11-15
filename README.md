# Data-Driven Analysis & Prediction of Singapore Resale HDB Prices

## 📌 Project Overview
This project provides a complete machine learning workflow (in **Project.ipynb**) for predicting Singapore HDB resale prices.  
It covers:

- Data loading & cleaning  
- Feature engineering using **custom transformers + scikit-learn pipelines**  
- Model training: **Linear Regression, Random Forest, MLP Regressor**  
- Hyperparameter tuning with **GridSearchCV**  
- Evaluation using **RMSE** and **R²**  
- Learning curve visualizations for all models  

The target variable is log-transformed to improve prediction stability.

---

## ⚙️ Environment Setup (Google Colab)
This project is optimized for **Google Colab**, where all essential libraries are already available.

### **Prerequisites**
- A Google account (for Colab)

### **Setup Steps**
1. **Open the Notebook**
   - Upload `Project.ipynb` to Colab or Google Drive  
   - OR open Colab → *File → Upload notebook*

2. **Install/Update Dependencies**  
   Add this to the first cell if needed:
   ```python
   !pip install pandas numpy scikit-learn matplotlib seaborn
   ```

---

## 📦 Dependencies
| Package       | Version (Recommended) | Purpose |
|---------------|------------------------|---------|
| pandas        | ≥ 2.0.0               | Data manipulation |
| numpy         | ≥ 1.25.0              | Numerical operations |
| scikit-learn  | ≥ 1.3.0               | Modeling, preprocessing, evaluation |
| matplotlib    | ≥ 3.7.0               | Visualizations |
| seaborn       | ≥ 0.12.0              | Statistical plots |

---

## 🗂️ Data Preparation
The dataset is obtained from the Singapore Government Data Portal.

### **Steps**
1. Download the dataset from:  
   *(Housing category on data.gov.sg)*

2. Upload the CSV to Colab using the left sidebar (**Files** → upload).  
   Rename it to:

```
ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv
```

3. Load it in the notebook:
```python
import pandas as pd
data = pd.read_csv("ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
```

---

## ▶️ Reproduction Steps
1. Prepare and upload the dataset  
2. Open **Project.ipynb** in Google Colab  
3. Click: **Runtime → Run all**

Everything including models, performance tables, and learning curves will be generated automatically.

---

## 📊 Key Results

### **1. Model Performance Table**
Comparison metrics for:
- **Linear Regression**  
- **Random Forest**  
- **MLP Regressor**

Metrics shown:
- **RMSE** (lower = better accuracy)  
- **R² Score** (how much variance is explained)

### **2. Learning Curves**
Three figures will be generated:
- Linear Regression  
- Random Forest  
- MLP Regressor  

Each plot displays:
- Training RMSE  
- Validation RMSE  
Used for diagnosing models’ **bias & variance**.

---

## 📁 Repository Structure
```
Project.ipynb     # Full ML workflow
README.md         # Project documentation
```

---

## 📝 Notes
- Files uploaded to Colab runtime are temporary unless saved to Google Drive  
- Ensure dataset name matches exactly for the notebook to load it properly

---

## 🎉 End
You’re ready to run and explore the ML workflow for predicting resale HDB prices!
