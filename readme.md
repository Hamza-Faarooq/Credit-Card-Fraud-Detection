# Credit Card Fraud Detection

## Project Overview
This project aims to detect fraudulent transactions in a credit card dataset using various machine learning algorithms. It includes data preprocessing, feature engineering, handling class imbalance, model training, evaluation, hyperparameter tuning, and deploying the best model using a Flask web application.

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Deployment](#model-deployment)
- [Installation](#installation)


## Dataset
The dataset used is the "creditcard.csv" file, which contains credit card transactions labeled as fraudulent (1) or non-fraudulent (0). If the file is not found, a sample dataset with similar structure is generated.

### Features
- `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: Principal components obtained with PCA.
- `Amount`: Transaction amount.
- `Class`: 1 for fraudulent transactions, 0 for non-fraudulent transactions.

## Data Preprocessing
- Load the dataset and handle missing values (if any).
- Standardize the `Amount` and `Time` columns using `StandardScaler`.
- Drop the original `Amount` and `Time` columns after scaling.

## Feature Engineering
- Extract additional features from the `Time` column: `hour` and `day_of_week`.
- Apply a log transformation to the `scaled_amount`.
- Create an interaction term between `log_amount` and `hour`.

## Model Training and Evaluation
### Machine Learning Models Used
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix

### Handling Class Imbalance
SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset.

### Training and Evaluation
Split the dataset into training and testing sets, train each model, and evaluate their performance using the metrics mentioned above.

## Hyperparameter Tuning
GridSearchCV is used to perform hyperparameter tuning for the Gradient Boosting model to find the best parameters.

### Parameter Grid
- `n_estimators`: [50, 100, 200]
- `max_depth`: [3, 6, 9]
- `learning_rate`: [0.01, 0.1, 0.2]

### Best Model
Evaluate the best model found by GridSearchCV and save it using `joblib`.

## Model Deployment
A Flask web application is created to deploy the best model for making predictions.

### Flask Application
- **Route**: `/predict`
- **Method**: `POST`
- **Input**: JSON containing feature values
- **Output**: JSON containing the prediction result (0 for non-fraudulent, 1 for fraudulent)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/credit_card_fraud_detection/credit-card-fraud-detection.git
   cd credit-card-fraud-detection