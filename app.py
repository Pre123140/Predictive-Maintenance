import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Define File Path ---
data_folder = os.path.join(os.path.dirname(__file__), "data") # Adjust path
predictions_rf_file = "data/predictions_with_rf.csv"



# --- Check if File Exists ---
if not os.path.exists(predictions_rf_file):
    raise FileNotFoundError(f"❌ ERROR: File not found: {predictions_rf_file}\nCheck if the file exists in 'data/' folder.")

# --- Load Dataset ---
data = pd.read_csv(predictions_rf_file)
print("✅ File loaded successfully! First 5 rows:")
print(data.head())

# --- Feature Selection ---
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = data[features]
y = data['Machine failure']  # Target variable

# --- Anomaly Detection ---
iso_forest = IsolationForest(contamination=0.1, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(X)

# --- Feature Importance Analysis ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\n🔹 Classification Report:\n")
print(classification_report(y_test, y_pred))

# --- Visualizations ---
# Anomaly Detection Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Anomaly', palette={1: 'blue', -1: 'red'})
plt.title("Anomaly Detection in Machine Data")
plt.show()

# Feature Importance Plot
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance Analysis")
plt.show()
