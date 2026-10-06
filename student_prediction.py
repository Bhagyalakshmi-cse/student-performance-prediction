# Student Performance Prediction
# Author: Bhagyalakshmi
# Machine Learning Project

from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
# Features:
# Study Hours, Attendance (%), Previous Score
X = np.array([
    [2, 60, 50],
    [3, 65, 55],
    [4, 70, 60],
    [5, 75, 65],
    [6, 80, 70],
    [7, 85, 75],
    [8, 90, 80],
    [9, 95, 85],
    [1, 55, 45],
    [10, 98, 90]
])

# Target: Final exam score
y = np.array([52, 57, 62, 67, 72, 77, 82, 87, 47, 92])

# Create the Machine Learning model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Get student information
print("===================================")
print("   STUDENT PERFORMANCE PREDICTION")
print("===================================")

study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous exam score: "))

# Predict final score
student_data = np.array([[study_hours, attendance, previous_score]])
prediction = model.predict(student_data)

# Keep score between 0 and 100
predicted_score = max(0, min(100, prediction[0]))

print("\n-----------------------------------")
print(f"Predicted Final Score: {predicted_score:.2f}")
print("-----------------------------------")

if predicted_score >= 90:
    print("Performance: Excellent 🌟")
elif predicted_score >= 75:
    print("Performance: Very Good 👍")
elif predicted_score >= 60:
    print("Performance: Good 🙂")
else:
    print("Performance: Needs Improvement 📚")
