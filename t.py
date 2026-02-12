import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Algorithms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ------------------------- 1. Data Preparation -------------------------
# Note: Using the same logic as your original code
try:
    data = pd.read_csv(
        r'C:\Users\ACER\Desktop\diabetes_binary\dataset\DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv')
    data['diabetic'] = data['diabetic'].str.strip()
    data = data[data['diabetic'].isin(['Yes', 'No'])]
    selected_features = ['age', 'gender', 'bmi',
                         'glucose', 'family_diabetes', 'hypertensive']
    X = pd.get_dummies(data[selected_features], drop_first=True)
    X = X.fillna(X.mean())
    y = data['diabetic'].map({'No': 0, 'Yes': 1})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=44)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=44)
except Exception as e:
    print(
        f"Error loading data: {e}. (This is expected if running outside your local environment)")
    # Creating dummy data for demonstration if file not found
    from sklearn.datasets import make_classification
    X_res, y_res = make_classification(
        n_samples=1000, n_features=6, random_state=44)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=44)

# ------------------------- 2. Define Models -------------------------
models = {
    "SVM (Support Vector Machine)": SVC(kernel='rbf', C=1.0, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=44),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=44)
}

# ------------------------- 3. Train and Evaluate -------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Algorithm": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2)
    })

# ------------------------- 4. Display & Save Results -------------------------
df_results = pd.DataFrame(results).sort_values(
    by="Accuracy (%)", ascending=False)
print("\n--- Comparison Table ---")
print(df_results.to_string(index=False))

# Save to CSV for user
df_results.to_csv('algorithms_comparison_results.csv', index=False)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x="Accuracy (%)", y="Algorithm",
            data=df_results, palette="viridis")
plt.xlim(0, 100)
plt.title("Comparison of ML Algorithms Accuracy")
for index, value in enumerate(df_results["Accuracy (%)"]):
    plt.text(value + 1, index, str(value) + "%")

plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=300)
print("\nComparison chart saved as 'comparison_chart.png'")
