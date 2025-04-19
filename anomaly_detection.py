import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import plotly.express as px
import matplotlib

# Load datasets
original_file = 'predictive_maintenance.csv'
processed_file = 'processed.csv'
predictions_file = 'predictions.csv'
predictions_ts_file = 'predictions_with_timestamp.csv'
predictions_rf_file = r"D:\PERSONAL\PORTFOLIO\TECHNICAL PROJECTS\PREDICTIVE MAINTENANCE\predictive_maintenance_project\data\predictions_with_rf.csv"

data = pd.read_csv(predictions_rf_file)

# Selecting relevant features
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = data[features]
y = data['Machine failure']

# --- ANOMALY DETECTION ---
iso_forest = IsolationForest(contamination=0.2, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(X)

# Save anomaly detection results
data.to_csv('data/anomaly_results.csv', index=False)

# --- FEATURE IMPORTANCE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- VISUALIZATIONS ---
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

# --- Additional Insights ---
# Count of anomalies per failure mode
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
anomaly_counts = data.groupby('Anomaly')[failure_modes].sum()
anomaly_counts.plot(kind='bar', figsize=(10, 5))
plt.title("Anomalies per Failure Mode")
plt.ylabel("Count")
plt.xlabel("Anomaly (-1: Outlier, 1: Normal)")
plt.legend(title="Failure Mode")
plt.show()

# Overlay failure mode on anomaly plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Machine failure', style='Anomaly', palette='coolwarm')
plt.title("Anomaly Detection with Failure Modes")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(data[features + ['Machine failure']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Machine Parameters")
plt.show()

# 3D Scatter Plot for Anomalies
fig = px.scatter_3d(data, x='Rotational speed [rpm]', y='Torque [Nm]', z='Air temperature [K]',
                     color='Anomaly', color_discrete_map={1: 'blue', -1: 'red'}, title="3D Anomaly Detection")
fig.show()

# Failure Rate by Type
failure_rate = data.groupby('Type')['Machine failure'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(x='Type', y='Machine failure', data=failure_rate, palette='viridis')
plt.title("Failure Rate by Machine Type")
plt.show()

# Boxplot for Torque Anomalies
plt.figure(figsize=(10, 5))
sns.boxplot(x='Anomaly', y='Torque [Nm]', data=data, palette='coolwarm')
plt.title("Torque Distribution in Anomalies")
plt.show()
