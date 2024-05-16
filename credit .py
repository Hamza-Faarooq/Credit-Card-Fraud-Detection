# Import necessary libraries for data manipulation, visualization, and modeling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to load the dataset; if not found, generate a sample dataset
try:
    # Load the dataset from a CSV file
    data = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    # If the file is not found, create a sample dataset with a similar structure
    data_dict = {
        'Time': np.random.randint(0, 172800, size=1000),  # Random time in seconds within two days
        'V1': np.random.randn(1000),
        'V2': np.random.randn(1000),
        'V3': np.random.randn(1000),
        'V4': np.random.randn(1000),
        'V5': np.random.randn(1000),
        'V6': np.random.randn(1000),
        'V7': np.random.randn(1000),
        'V8': np.random.randn(1000),
        'V9': np.random.randn(1000),
        'V10': np.random.randn(1000),
        'V11': np.random.randn(1000),
        'V12': np.random.randn(1000),
        'V13': np.random.randn(1000),
        'V14': np.random.randn(1000),
        'V15': np.random.randn(1000),
        'V16': np.random.randn(1000),
        'V17': np.random.randn(1000),
        'V18': np.random.randn(1000),
        'V19': np.random.randn(1000),
        'V20': np.random.randn(1000),
        'V21': np.random.randn(1000),
        'V22': np.random.randn(1000),
        'V23': np.random.randn(1000),
        'V24': np.random.randn(1000),
        'V25': np.random.randn(1000),
        'V26': np.random.randn(1000),
        'V27': np.random.randn(1000),
        'V28': np.random.randn(1000),
        'Amount': np.random.uniform(0, 5000, size=1000),  # Random amount between 0 and 5000
        'Class': np.random.randint(0, 2, size=1000)  # Random class, either 0 (non-fraud) or 1 (fraud)
    }

    # Convert the dictionary to a DataFrame
    data = pd.DataFrame(data_dict)
    # Save the DataFrame to a CSV file
    data.to_csv('creditcard.csv', index=False)
    print("Sample dataset created and saved as creditcard.csv")

# Print the first few rows of the dataset to understand its structure
print(data.head())

# Display information about the dataset, such as number of columns, non-null counts, and data types
print(data.info())

# Get descriptive statistics for numerical columns in the dataset
print(data.describe())

# Check for any missing values in the dataset
print(data.isnull().sum())

# Visualize the distribution of the 'Class' column (fraud vs. non-fraud)
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# Standardize the 'Amount' and 'Time' columns for better model performance
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns as they are now scaled
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Rearrange columns to have scaled features at the beginning and 'Class' at the end
data = data[['scaled_amount', 'scaled_time'] + [col for col in data.columns if col not in ['scaled_amount', 'scaled_time', 'Class']] + ['Class']]

# Extract additional features from 'scaled_time' (original 'Time' in seconds)
data['hour'] = (data['scaled_time'] // 3600) % 24  # Extract hour of the day
data['day_of_week'] = (data['scaled_time'] // (3600 * 24)) % 7  # Extract day of the week

# Apply log transformation to 'scaled_amount' to handle skewness
data['log_amount'] = np.log1p(data['scaled_amount'])

# Create an interaction term between 'log_amount' and 'hour'
data['amount_time_interaction'] = data['log_amount'] * data['hour']

# Drop 'scaled_time' and 'scaled_amount' as they have been transformed into new features
data.drop('scaled_time', axis=1, inplace=True)
data.drop('scaled_amount', axis=1, inplace=True)

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Initialize machine learning models to be evaluated
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate each model
for model_name, model in models.items():
    # Fit the model on the training data
    model.fit(X_train, y_train)
    # Predict on the testing data
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics for the model
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")

# Hyperparameter tuning for the Gradient Boosting model using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid.best_params_)

# Evaluate the best estimator found by GridSearchCV
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Best Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the best model to a file using joblib
joblib.dump(best_model, 'best_model.pkl')

# Deployment of the model using a Flask web application
# Import Flask and joblib for creating the web application
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create a Flask application
app = Flask(__name__)

# Load the saved model
model = joblib.load('best_model.pkl')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    # Convert the data to a numpy array and predict using the loaded model
    prediction = model.predict(np.array([data['features']]))
    # Return the prediction result as JSON
    output = int(prediction[0])
    return jsonify(result=output)

# Run the Flask application on port 5000
if __name__ == '__main__':
    app.run(port=5000, debug=True)