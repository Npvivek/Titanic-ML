# Titanic Survival Prediction

## Project Overview

This project predicts survival on the Titanic using the **Kaggle Titanic Dataset**. Various machine learning models are employed to classify passengers as either **Survived** or **Not Survived** based on their features like age, sex, passenger class, and family size.

## Key Features

- **Data Preprocessing**:
  - Handled missing values using median/mode imputation.
  - Outliers were detected and removed using the **Interquartile Range (IQR)** method.
  - Feature extraction included creating new features like `Title`, `Family_Size_Grouped`, and `Ticket_Frequency`.
  - Categorical features were label-encoded and one-hot encoded for compatibility with machine learning algorithms.

- **Exploratory Data Analysis (EDA)**:
  - Visualized key features using **histograms**, **bar plots**, and **box plots** to understand their relationship with the target (`Survived`).
  - Highlighted correlations between numerical features and survival rates.

- **Modeling**:
  - Implemented and compared multiple models, including:
    - **Logistic Regression**
    - **Decision Tree Classifier**
    - **Random Forest Classifier**
    - **Support Vector Machine (SVM)**
    - **K-Nearest Neighbors (KNN)**
    - **XGBoost Classifier**

- **Model Evaluation**:
  - Models were evaluated based on **accuracy**.
  - **Logistic Regression** was used for the final predictions due to its balanced performance.

## Dependencies

- **Programming Language**: Python 3.x
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `seaborn`, `matplotlib`
  - Machine Learning: `scikit-learn`, `xgboost`
