import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/home/mems/Downloads/adult.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Data Preprocessing

# Replace '?' with NaN and handle missing values
df = df.replace('?', np.nan)

# Drop rows with missing values (alternative: impute)
df = df.dropna()

print(f"Dataset shape after handling missing values: {df.shape}")

# Encode categorical variables
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 
                      'relationship', 'race', 'sex', 'native.country', 'income']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target variable
X = df.drop('income', axis=1)
y = df['income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train logistic regression model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(logreg.coef_[0])
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(15, 5))

# Confusion Matrix Heatmap
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['<=50K', '>50K'], 
            yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Feature Importance Plot
plt.subplot(1, 3, 2)
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.title('Top 10 Feature Importances')
plt.xlabel('Absolute Coefficient Value')
plt.gca().invert_yaxis()

# Probability Distribution
plt.subplot(1, 3, 3)
plt.hist(y_pred_proba[:, 1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Predicted Probability Distribution')
plt.xlabel('Probability of Income >50K')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Function to make predictions on new data
def predict_income(model, scaler, new_data, label_encoders):
    """
    Predict income for new data
    new_data should be a dictionary with the same features as the training data
    """
    # Create a DataFrame with the same structure as training data
    new_df = pd.DataFrame([new_data])
    
    # Encode categorical variables using the same encoders
    for col in categorical_columns:
        if col in new_df.columns and col != 'income':
            try:
                new_df[col] = label_encoders[col].transform([new_data[col]])[0]
            except ValueError:
                # Handle unseen labels by using the most common class
                new_df[col] = 0
    
    # Scale the features
    new_scaled = scaler.transform(new_df)
    
    # Make prediction
    prediction = model.predict(new_scaled)[0]
    probability = model.predict_proba(new_scaled)[0]
    
    # Decode prediction
    income_pred = label_encoders['income'].inverse_transform([prediction])[0]
    
    return income_pred, probability[1]

# Example prediction
example_data = {
    'age': 35,
    'workclass': 'Private',
    'fnlwgt': 200000,
    'education': 'Bachelors',
    'education.num': 13,
    'marital.status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'capital.gain': 0,
    'capital.loss': 0,
    'hours.per.week': 50,
    'native.country': 'United-States'
}

predicted_income, probability = predict_income(logreg, scaler, example_data, label_encoders)
print(f"\nExample Prediction:")
print(f"Predicted Income: {predicted_income}")
print(f"Probability of Income >50K: {probability:.4f}")

# Model performance metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAdditional Metrics:")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save the model and preprocessing objects
import joblib

model_artifacts = {
    'model': logreg,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': list(X.columns)
}

joblib.dump(model_artifacts, 'income_prediction_model.pkl')
print("\nModel saved as 'income_prediction_model.pkl'")