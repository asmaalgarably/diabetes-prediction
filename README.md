# 💉 Smart Diabetes Risk Assessment System

## 📖 Overview

The Smart Diabetes Risk Assessment System uses Artificial Intelligence and Machine Learning to analyze patient health data and predict diabetes risk levels, providing medical recommendations to support decision-making.

### 🟢 Features

- Health Risk Prediction
- PDF Medical Reports
- Batch Data Analysis
- Medical Image OCR Analysis

> ⚠️ This system supports decision-making but does not replace professional medical consultation.

---

## 🧑‍⚕️ Patient Data Entry

Users can enter the following patient data:

- Patient Name
- Age
- Gender
- Weight & Height
- Glucose Level
- Hypertension
- Family Diabetes History

BMI (Body Mass Index) is calculated from weight and height for model input.

---

## 🧠 Diabetes Risk Prediction

1. **Data Preparation**:

   - Convert categorical variables to numeric (One-Hot Encoding)
   - Fill missing values with column mean
   - Align columns with the trained model
   - Scale features using `StandardScaler`

2. **Prediction Model | RandomForestClassifier**

   - Trained on a diabetes dataset from Bangladesh.
   - Handled imbalanced data with **SMOTE**.
   - Hyperparameters optimized using `GridSearchCV`.
   - Outputs prediction (`Diabetic` or `Non-Diabetic`) and probability.

3. **Risk Interpretation**
   - **Low**: probability < 0.33  
     ✅ Regular monitoring, maintain healthy lifestyle, balanced diet, regular physical activity.
   - **Medium**: 0.33 ≤ probability < 0.66  
     ⚠️ Monitor glucose weekly, reduce simple carbs, increase healthy protein & vegetables, consult a dietitian.
   - **High**: probability ≥ 0.66  
     🚨 Consult a doctor immediately, follow diabetic diet & exercise plan, monitor glucose daily.

---

## 📄 Patient Medical Report

- Generates a PDF report including:
  - Patient basic information
  - BMI and glucose level
  - Risk level
  - Medical recommendations
- The report can be downloaded via the `Download PDF` button.

---

## 📊 Batch Data Analysis

- Upload `.csv` or `.xlsx` files with multiple patients.
- Calculate BMI for each patient.
- Use the model to predict each patient's risk.
- Display results in a **Status** table.
- Plot patient distribution using a bar chart.

---

## 🩸 Medical Image Analysis

- Upload glucose test image (PNG, JPG, JPEG)
- Use **OCR (EasyOCR)** to extract glucose value.
- Simple diagnosis:
  - Low Glucose (<70)
  - Normal (70–140)
  - High Glucose (>140)
- Provides medical recommendations for each case.

---

## ⚙️ Model Training

1. Read dataset: `DiaBD_A Diabetes Dataset`
2. Clean data and remove invalid values
3. Select features: `age`, `gender`, `bmi`, `glucose`, `family_diabetes`, `hypertensive`
4. Convert categorical variables using One-Hot Encoding
5. Balance the dataset using SMOTE
6. Split dataset: 80% train / 20% test
7. Optimize RandomForest hyperparameters using `GridSearchCV`
8. Save the trained model, scaler, and columns for future predictions

---

## 🖥️ Predicting New Patient Data

- Enter patient data (age, gender, weight, height, glucose, family diabetes, hypertension)
- Calculate BMI and prepare features
- Make predictions and probability
- Example result:

| Predicted    | Probability (%) |
| ------------ | --------------- |
| Diabetic     | 85.23           |
| Not Diabetic | 12.50           |

---

## 🛠️ Requirements

- Python 3.9+
- Libraries:
  - streamlit
  - pandas, numpy
  - scikit-learn, imblearn
  - easyocr, opencv-python
  - fpdf, arabic-reshaper, bidi
  - matplotlib
  - pillow
  - pickle

---

## 🚀 How to Run

1. Open the project in a Python environment
2. Install required libraries
3. Run the Streamlit app:

```bash
cd app
streamlit run main.py
```
