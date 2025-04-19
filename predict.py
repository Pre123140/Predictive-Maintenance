import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/failure_prediction_model.pkl")

# Load dataset
df = pd.read_csv("data/predictions_with_timestamp.csv")

# Define features for prediction
features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
X = df[features]

# Make predictions
df["Predicted Failure"] = model.predict(X)

# Save predictions
df.to_csv("data/predictions_with_rf.csv", index=False)
print("Predictions saved successfully!")
