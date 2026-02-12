import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------- 1. Data Loading and Cleaning -------------------------
file_path = r'C:\Users\ACER\Desktop\diabetes_binary\dataset\DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv'

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Clean the target column (diabetic)
data['diabetic'] = data['diabetic'].str.strip()
data = data[data['diabetic'].isin(['Yes', 'No'])]

# Select relevant features for training
selected_features = ['age', 'gender', 'bmi',
                     'glucose', 'family_diabetes', 'hypertensive']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(data[selected_features], drop_first=True)
X = X.fillna(X.mean())  # Fill missing values with column mean

# Map target column to binary values (0 and 1)
y = data['diabetic'].map({'No': 0, 'Yes': 1})

# Save feature column names for future prediction consistency
columns = X.columns

# ------------------------- 2. Data Preprocessing (Scaling & SMOTE) -------------------------
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=44)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=44)

# ------------------------- 3. Model Building and Training (Random Forest) -------------------------
best_rf_model = RandomForestClassifier(
    n_estimators=200,      # Number of trees in the forest
    max_depth=None,        # Maximum depth of the tree
    min_samples_split=2,   # Minimum samples required to split an internal node
    random_state=44,
    n_jobs=-1              # Use all available processors for faster training
)

best_rf_model.fit(X_train, y_train)

# ------------------------- 4. Model Evaluation -------------------------
y_pred = best_rf_model.predict(X_test)

print("\n" + "="*30)
print("Model Evaluation Results")
print("="*30)
print(f"Accuracy: {round(accuracy_score(y_test, y_pred)*100, 2)} %")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------- 5. Saving Models for Deployment -------------------------
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained model
with open('models/rf_diabetes_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)

# Save the scaler (essential for processing new input data)
with open('models/rf_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the feature columns
with open('models/rf_columns.pkl', 'wb') as f:
    pickle.dump(columns, f)

print("\n" + "="*60)
print("Model and artifacts saved successfully in the 'models' folder.")
print("="*60)

# ------------------------- 6. Plotting Confusion Matrix -------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Random Forest Model')
plt.savefig('final_confusion_matrix_en.png', dpi=300)
print("Confusion matrix plot saved as 'final_confusion_matrix_en.png'")
