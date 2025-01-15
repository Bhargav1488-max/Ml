import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Example dataset
data = {
    "subject1": [80, 70, 60, 90, 85],
    "subject2": [75, 65, 55, 85, 80],
    "subject3": [78, 68, 58, 88, 83],
    "subject4": [72, 62, 52, 82, 77],
    "next_semester": [85, 75, 65, 95, 90],
}
df = pd.DataFrame(data)

# Prepare features and target
X = df[["subject1", "subject2", "subject3", "subject4"]]
y = df["next_semester"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open("marks_predictor_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
