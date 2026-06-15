# Drug Discovery AI using Machine Learning

## Project Overview

This project was developed as part of the Master's Program in Artificial Intelligence.

The objective is to apply Machine Learning techniques to Drug Discovery by predicting interactions between chemical compounds and biological targets.

Two benchmark datasets were used:

* DAVIS Dataset (Drug-Target Affinity Regression)
* BACE Dataset (Binary Molecular Activity Classification)

The project includes a complete Machine Learning pipeline covering data preprocessing, feature engineering, model training, evaluation, and deployment through a prediction interface.


# Project Structure

```text
ML_PROJECT/

├── Data/
│   ├── raw/
│   └── processed/
│       ├── bace.csv
│       └── davis.csv
│
├── notebooks/
│   ├── drug_discovery_preprocessing.ipynb
│   └── drug_discovery_modeling.ipynb
│
├── models/
│   ├── bace_classifier.joblib
│   ├── davis_regressor.joblib
│   ├── scaler_bace.joblib
│   ├── scaler_davis.joblib
│   └── davis_target_encoder.joblib
│
├── src/
│   ├── train_models.py
│   └── app.py
│
├── docs/
│   ├── Rapport_Drug_Discovery_ML_versionfinal.pdf
│   └── poster.pdf
│
├── environment.yml
├── requirements.txt
├── README.md
└── LICENSE
```


# Datasets

## DAVIS Dataset

The DAVIS dataset is used for Drug-Target Affinity prediction.

Main characteristics:

* 68 unique drugs
* 379 protein targets
* 25,772 drug-target interactions
* Continuous affinity values (Kd)

Task:

Regression

Target variable:

* Y (Binding Affinity)

---

## BACE Dataset

The BACE dataset is used for molecular activity prediction.

Main characteristics:

* 1,513 molecules
* Binary activity labels

Task:

Binary Classification

Target variable:

* Class

  * 1 = Active
  * 0 = Inactive


# Project Workflow

The project follows the following pipeline:

1. Data Loading
2. Data Exploration
3. Data Cleaning
4. Missing Values Analysis
5. Duplicate Detection
6. SMILES Validation using RDKit
7. Feature Engineering
8. Descriptor Extraction
9. Standardization
10. Class Balancing using SMOTE
11. Model Training
12. Model Evaluation
13. Prediction Interface Development


# Molecular Descriptors

The following molecular descriptors were extracted using RDKit:

* Molecular Weight (MW)
* LogP
* TPSA
* Number of Hydrogen Donors
* Number of Hydrogen Acceptors
* Rotatable Bonds
* Ring Count
* Heavy Atom Count


# Machine Learning Models

## Classification Models (BACE)

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest Classifier
* Gradient Boosting Classifier

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC AUC


## Regression Models (DAVIS)

* Ridge Regression
* Support Vector Regression (SVR)
* Random Forest Regressor
* Gradient Boosting Regressor

Evaluation Metrics:

* MSE
* RMSE
* MAE
* R² Score


# Best Results

## BACE Classification

Best Model:

Gradient Boosting Classifier

Performance:

* Accuracy ≈ 0.78
* F1-Score ≈ 0.76
* ROC AUC ≈ 0.85


## DAVIS Regression

Best Model:

Gradient Boosting Regressor

Performance:

* RMSE ≈ 3075
* MAE ≈ 2304
* R² ≈ 0.39


# Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate drug_discovery_env
```

Install additional dependencies:

```bash
pip install -r requirements.txt
```


# Run the Application

Launch the Streamlit interface:

```bash
streamlit run src/app.py
```


# Technologies Used

* Python
* Pandas
* NumPy
* RDKit
* Scikit-Learn
* Imbalanced-Learn
* Streamlit
* Matplotlib
* Seaborn


# Authors

Hajar Bentaib / Hanane Ayar

Master of Excellence in Artificial Intelligence

Faculty of Sciences Ben M'Sick

Casablanca, Morocco

Academic Year: 2025-2026



```

```
