# Employee Attrition Prediction Model

<div align="center">
  <img src="https://img.shields.io/badge/MLOps-Hub-blue?style=for-the-badge" alt="MLOps Hub" />
  <img src="https://img.shields.io/badge/python-3.12.x-blue?style=for-the-badge" alt="Python Version" />
  <img src="https://img.shields.io/badge/status-Active-green?style=for-the-badge" alt="Status" />
</div>

<hr />


## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
    - [Create Virtual Environment](#step-1-create-virtual-environment)
    - [Install Dependencies](#step-2-install-dependencies)
    - [Training the Model](#step-3-training-the-model)
    - [Testing/Prediction](#step-4-testingprediction)
- [ML Pipeline](#ml-pipeline)
- [Frontend](#frontend)
- [Contribution](#contribution)


## Overview

This project is an **Employee Attrition Prediction System** built using machine learning. It predicts whether an employee will leave (attrition) or stay at a company based on various employee and workplace factors. The model uses classification algorithms to identify at-risk employees, helping organizations implement targeted retention strategies.

**Business Value:**
- Identify employees likely to leave the organization
- Enable proactive retention interventions
- Reduce turnover costs and improve workforce planning
- Support HR decision-making with data-driven insights


## Datasets

The main dataset used is [employee_attrition.csv](./datasets/employee_attrition.csv), containing 74,500 employee records with both training and test data.

**Dataset Features:**

| Feature | Description |
|---------|-------------|
| Employee ID | Unique identifier for each employee |
| Age | Age of the employee |
| Gender | Gender (Male/Female) |
| Years at Company | Tenure in years |
| Job Role | Position/Department |
| Monthly Income | Salary in monthly terms |
| Work-Life Balance | Rating (Poor/Good/Excellent) |
| Job Satisfaction | Level (Low/Medium/High) |
| Performance Rating | Employee performance (Low/Average/High) |
| Number of Promotions | Career progression count |
| Overtime | Whether employee works overtime (Yes/No) |
| Distance from Home | Commute distance in km |
| Education Level | Highest education attained |
| Marital Status | Single/Married/Divorced |
| Number of Dependents | Family dependents count |
| Job Level | Position hierarchy level |
| Company Size | Organization size (Small/Medium/Large) |
| Company Tenure | Time at current company |
| Remote Work | Remote work eligibility (Yes/No) |
| Leadership Opportunities | Career growth potential |
| Innovation Opportunities | Innovation involvement |
| Company Reputation | Market reputation |
| Employee Recognition | Recognition programs |
| **Attrition** | **Target: Stayed/Left** |
| dataset_type | Train/Test split indicator |

## Features

**Key Factors Influencing Employee Attrition:**
- **Compensation**: Monthly income, job level
- **Work Environment**: Work-life balance, overtime, remote work, distance from home
- **Career Development**: Promotions, leadership opportunities, innovation participation
- **Job Satisfaction**: Overall satisfaction, performance ratings, recognition
- **Demographics**: Age, tenure, marital status, education level
- **Company Factors**: Company size, reputation, tenure at organization


## Project Structure

```
employee-attrition-model/
├── datasets/
│   ├── employee_attrition.csv          # Main dataset (74,500 records)
│   └── data-engg/                      # Processed datasets at each pipeline stage
│       ├── 01_ingestion.csv
│       ├── 02_validation.csv
│       ├── 03_eda_df.csv
│       ├── 04_cleaned_df.csv
│       ├── 05_feature_engg_df.csv
│       └── 06_preprocess_train/test_df.csv
├── artifacts/                          # Trained models and artifacts
│   ├── model_v1/
│   └── model_v2/
├── src/
│   ├── data-pipeline/                  # Data engineering pipeline
│   │   ├── _01_ingestion.py            # Data loading and exploration
│   │   ├── _02_validation.py           # Data validation with Pandera
│   │   ├── _03_eda.py                  # Exploratory data analysis
│   │   ├── _04_cleaning.py             # Data cleaning
│   │   ├── _05_feature_engg.py         # Feature engineering
│   │   └── _06_preprocessing.py        # Preprocessing & scaling
│   └── model-pipeline/                 # Model development pipeline
│       ├── _07_training.py             # Model training
│       ├── _08_evaluation.py           # Model evaluation
│       ├── _09_cross_validation.py     # Cross-validation
│       ├── _10_tuning.py               # Hyperparameter tuning
├── frontend/
│   ├── app.py                          # Flask web application
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/script.js
│   └── templates/
│       └── index.html
├── notebook/
│   └── test.ipynb                      # Jupyter notebook for experimentation
├── requirements.txt
└── README.md
```


## Tech Stack

- **Python Version**: 3.12+
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Data Validation**: pandera
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Flask, Flask-CORS
- **Model Format**: Pickle (.pkl)


## ML Model Pipeline

### 1. Data Engineering Pipeline
- **Ingestion**: Load raw data from CSV
- **Validation**: Schema validation using Pandera
- **EDA**: Exploratory Data Analysis - understand data patterns
- **Cleaning**: Handle missing values, outliers, inconsistencies
- **Feature Engineering**: Create new features, encoding categorical variables
- **Preprocessing**: Scaling, normalization, train-test split

### 2. Model Development Pipeline
- **Training**: Train classification models (Logistic Regression, Random Forest, etc.)
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Cross-Validation**: K-fold validation for robust assessment
- **Hyperparameter Tuning**: GridSearch/RandomSearch for optimal parameters

### 3. Frontend
- Flask-based web application for model inference
- Real-time predictions on new employee data


## Setup & Installation

#### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Run Data Pipeline

Execute each data engineering step sequentially:

```bash
cd src/data-pipeline
python _01_ingestion.py
python _02_validation.py
python _03_eda.py
python _04_cleaning.py
python _05_feature_engg.py
python _06_preprocessing.py
```

#### Step 4: Train & Evaluate Model

Once data is preprocessed, run model development pipeline:

```bash
cd ../model-pipeline
python _07_training.py       # Train the model
python _08_evaluation.py     # Evaluate performance
python _09_cross_validation.py  # Validate robustness
python _10_tuning.py         # Hyperparameter optimization
```

#### Step 5: Run Web Application

```bash
cd ../../frontend
python app.py
```

Access the application at `http://localhost:5000`


## Usage

**For Prediction on New Data:**

Load the trained model and make predictions:

```python
import pickle
import pandas as pd

# Load model
with open('artifacts/model_v1/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
with open('artifacts/model_v1/preprocessor.pkl', 'rb') as f:
    features = pickle.load(f)

# Prepare new employee data
new_employee = pd.DataFrame({
    'Age': [35],
    'Monthly Income': [6000],
    # ... other features
})

# Preprocess and predict
processed = preprocessor.transform(new_employee)
prediction = model.predict(processed)
probability = model.predict_proba(processed)

print(f"Attrition Risk: {prediction[0]}")
print(f"Probability: {probability[0]}")
```


## Key Findings

- Identify primary drivers of employee attrition
- Understand demographic and workplace patterns
- Data-driven recommendations for retention strategies


## Contributing

Please read our [Contributing Guidelines](CONTRIBUTION.md) before submitting pull requests.

Contributions are welcome! Please follow standard Git workflow:
1. Create a feature branch
2. Make your changes
3. Submit a pull request


## License
This project is under [MIT Licence](LICENCE) support.
