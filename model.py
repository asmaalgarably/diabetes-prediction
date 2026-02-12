import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------- Read the dataset -------------------------
data = pd.read_csv(
    r'C:\Users\ACER\Desktop\diabetes_binary\dataset\DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv'
)

# Clean the target column
data['diabetic'] = data['diabetic'].str.strip()
data = data[data['diabetic'].isin(['Yes', 'No'])]

# Select relevant features (use BMI, not height/weight for training)
selected_features = ['age', 'gender', 'bmi',
                     'glucose', 'family_diabetes', 'hypertensive']

# Convert categorical variables to dummies
X = pd.get_dummies(data[selected_features], drop_first=True)
X = X.fillna(X.mean())  # Fill missing values with column mean

# Map target column to 0 and 1
y = data['diabetic'].map({'No': 0, 'Yes': 1})

# Save columns for future use with new data
columns = X.columns

# ------------------------- Scaling features -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------- Handle imbalanced data using SMOTE -------------------------
smote = SMOTE(random_state=44)
X_res, y_res = smote.fit_resample(X_scaled, y)

# ------------------------- Split data into train and test sets -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=44)

# ------------------------- Grid Search to find best hyperparameters -------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=44)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best RandomForest parameters:", grid_search.best_params_)

# ------------------------- Evaluate the model -------------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------- Save the model, scaler, and columns -------------------------
with open('models/rf_diabetes_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/rf_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/rf_columns.pkl', 'wb') as f:
    pickle.dump(columns, f)

# ------------------------- Predict on new patient data -------------------------
new_data = pd.DataFrame({
    'age': [50, 35],
    'gender': ['Male', 'Female'],
    'weight': [70, 50],
    'height': [170, 160],
    'glucose': [140, 110],
    'family_diabetes': [0, 1],
    'hypertensive': [0, 0]
})

# Calculate BMI from weight and height
new_data['bmi'] = new_data['weight'] / ((new_data['height']/100) ** 2)

# Drop weight and height as model uses BMI
new_data = new_data.drop(['weight', 'height'], axis=1)

# Convert categorical variables
new_data = pd.get_dummies(new_data, drop_first=True)

# Add missing columns if any
for col in columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder columns to match training data
new_data = new_data[columns]

# Scale new data
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = best_model.predict(new_data_scaled)
probabilities = best_model.predict_proba(new_data_scaled)[:, 1]

# Display results
result = pd.DataFrame({
    'Predicted': ['Diabetic' if val == 1 else 'Not Diabetic' for val in predictions],
    'Probability (%)': [round(prob*100, 2) for prob in probabilities]
})

print("\nPrediction results on new data:")
print(result)
cm = confusion_matrix(y_test, y_pred)

# 2. إعداد الرسم البياني
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)   

# رسم المصفوفة باستخدام Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])

# إضافة العناوين
plt.xlabel('Predicted Label (توقع النظام)')
plt.ylabel('True Label (الحالة الحقيقية)')
plt.title('Confusion Matrix - Diabetes Prediction Model')

# حفظ الصورة لاستخدامها في المشروع
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
