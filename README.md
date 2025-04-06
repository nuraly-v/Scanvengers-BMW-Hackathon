# Scanvengers-BMW-Hackathon

Team Scanvenger's work for creating the prediction model using Causal machine learning.

# BMW 2024 Causal Machine Learning Hackathon

Welcome to our repository for the BMW 2024 Causal Machine Learning Hackathon! This project tackles quality prediction and causal inference on production line sensor data, aiming to optimize automotive part manufacturing using advanced machine learning techniques.

## 📁 Project Structure

### 1. `BMWHackathon.ipynb`

#### 🔍 Description:

This is the main exploration and preprocessing notebook. It contains:

- Data loading and cleaning
- Visualization of distributions and correlations
- Initial feature engineering

#### 📌 Highlights:

- Focuses on sensor readings, part types, and timestamps
- Heavy use of `pandas`, `matplotlib`, and `seaborn` for EDA
- 75+ code cells diving deep into the structure of the dataset

---

### 2. `modelpred.ipynb`

#### 🔍 Description:

A quick notebook for running statistical summaries and basic model predictions on the training set.

#### 📌 Highlights:

- Loads the `train.csv` file
- Outputs summary statistics
- Initial steps toward modeling

---

### 3. `newrock.ipynb`

#### 🔍 Description:

A powerful modeling pipeline with feature engineering, dimensionality reduction, and ensemble learning.

#### 📌 Highlights:

- Outlier detection using IQR
- Feature engineering with PCA and aggregation
- Model training with `RandomForest`, `LightGBM`, and `XGBoost`
- Stacked classifier for robust performance
- Uses `SMOTETomek` for handling class imbalance

#### 📈 Output:

- Classification report
- Feature importance
- ROC-AUC score
- Cleaned dataset saved as `cleaned_dataset4.csv`

---

### 4. `scanvenge2.ipynb`

#### 🔍 Description:

Focuses on causal inference using the `DoWhy` library to explore treatment effects in the manufacturing process.

#### 📌 Highlights:

- Timestamp feature extraction (hour, shift, day)
- Integration with the `dowhy` package for causal modeling
- Preprocessing part of a causal analysis pipeline

---

### 5. `scanvenger.ipynb`

#### 🏆 "Winner Notebook" 🏆

#### 🔍 Description:

Final refined model with the best performance — this was the top submission.

#### 📌 Highlights:

- Intuitive feature engineering with timestamp-based logic
- One-hot encoding of categorical features
- RandomForest classification with feature importance visualization
- Simple and effective model pipeline for real-time prediction

#### 📈 Output:

- Classification report and accuracy
- Feature importance plot

---

## 🧪 Technologies Used

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- xgboost, lightgbm
- dowhy (for causal inference)

## 📦 Dataset

The dataset includes:

- Sensor measurements for car parts
- Timestamps and part metadata
- Labels indicating part status (`OK` vs `NOK`)
