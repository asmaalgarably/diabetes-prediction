import pandas as pd

# Sample data in English
data = {
    "id": [1, 2, 3, 4],
    "name": ["Ahmed", "Sarah", "Mohamed", "Laila"],
    "age": [45, 32, 50, 28],
    "gender": ["Male", "Female", "Male", "Female"],
    "weight": [80, 60, 90, 55],   
    "height": [170, 160, 175, 165],   
    "glucose": [120, 110, 140, 100],  
    "hypertensive": ["Yes", "No", "Yes", "No"],
    "family_diabetes": ["Yes", "Yes", "No", "No"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("patients_sample_english.csv", index=False)
print("File created: patients_sample_english.csv")
