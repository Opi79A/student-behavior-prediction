import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# --- Data Preparation ---
# Creating a synthetic dataset for student performance and behavior
student = ['Asma', 'Bob', 'Charlie', 'David', 'Eve', 'sara', 'Ali', 'Mona', 'Omar', 'Lina']
grade_math = [85, 78, 92, 60, 95, 66, 42, 90, 87, 91]
grade_science = [90, 40, 58, 92, 22, 56, 91, 93, 87, 95]
grade_english = [80, 86, 23, 90, 33, 66, 84, 91, 89, 94]
behavior = ['Good', 'Bad','Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Good', 'Good'] 
absences = [2, 5, 1, 0, 0, 3, 4, 1, 2, 0]
issues = [1, 0, 3, 2, 0, 1, 2, 0, 1, 0]

df = pd.DataFrame({
    'student': student,
    'grade_math': grade_math,
    'grade_science': grade_science,
    'grade_english': grade_english,
    'behavior': behavior,
    'absences': absences,
    'issues': issues
})

# --- Preprocessing ---
# Encode the target labels (Good/Bad) into 0 and 1
le = LabelEncoder()
df['behavior_encoded'] = le.fit_transform(df['behavior'])

# Selecting features and target variable
X = df[['grade_math', 'grade_science', 'grade_english', 'absences', 'issues']]
y = df['behavior_encoded']

# Split data: 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling: Normalize numerical values for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# --- Prediction for a New Student ---
new_student = {
    'grade_math': 100, 
    'grade_science': 100,
    'grade_english': 100,    
    'absences': 7,
    'issues': 7
}
new_student_df = pd.DataFrame([new_student])
new_student_scaled = scaler.transform(new_student_df)

# Get prediction and probability
prediction = model.predict(new_student_scaled)
probabilities = model.predict_proba(new_student_scaled)

# Final Output
result = le.inverse_transform(prediction)
print(f"The predicted behavior for the new student is: {result[0]}")
print(f"Confidence: {np.max(probabilities) * 100:.2f}%")
