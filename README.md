# 🔧 AI-Powered Predictive Maintenance for Manufacturing

## 📌 Overview
This project leverages machine learning and anomaly detection to forecast machine failures using real-world manufacturing sensor data. By combining supervised classification and unsupervised anomaly detection, it enables smarter maintenance planning and minimizes unexpected downtimes.

---

## 🎯 Objective
- Predict machine failure using operational features like torque, tool wear, and temperature.
- Detect anomalies using Isolation Forest.
- Visualize patterns to guide data-informed maintenance decisions.

---

## 💼 Business Impact
Predictive maintenance reduces unplanned downtimes, prevents costly repairs, improves workplace safety, and supports proactive monitoring across production environments.

---

## 🧠 Algorithms Used
- **Random Forest Classifier** – For machine failure prediction
- **Isolation Forest** – For unsupervised anomaly detection

---

## 🧪 Dataset
- **Source**: `predictive_maintenance.csv`
- **Key Features**:
  - Machine Type, Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min]
  - Failure Modes: TWF, HDF, PWF, OSF, RNF
  - Target: `Machine failure`

---

## 🔄 Project Flow
1. **Preprocessing** – Cleaned raw data, removed noise, and encoded features
2. **Training** – Random Forest Classifier trained on core operational variables
3. **Anomaly Detection** – Isolation Forest used to flag unusual conditions
4. **Visualization** – Created rich visual outputs to analyze trends and root causes
5. **Streamlit Dashboard (Optional)** – For real-time failure prediction insights

---

## 📂 Folder Structure
```
predictive_maintenance_project/
├── data/
│   ├── predictive_maintenance.csv
│   ├── processed_data.csv
│   ├── predictions.csv
│   ├── predictions_with_timestamp.csv
│   ├── predictions_with_rf.csv
│   ├── anomaly_results.csv
│
├── models/
│   ├── failure_prediction_model.pkl
│   ├── predictive_maintenance_model.pkl
│   ├── feature_names.pkl
│
├── output/
│   ├── Figure_1.png to Picture_8.png   # Visualizations
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   ├── anomaly_detection.py
│   ├── visualize_results.py
│
├── requirements.txt
```

---

## 📊 Visual Outputs
1. **Anomaly Detection (Torque vs RPM)** – Scatterplot with outliers highlighted
2. **Feature Importance** – Top predictors of machine failure
3. **Failure Mode Distribution** – Anomaly count by failure type
4. **Failure vs Anomaly Overlay** – Comparative visualization
5. **Correlation Heatmap** – Relationship between all features
6. **Failure Rate by Machine Type** – Grouped bar chart
7. **Torque Boxplot by Anomaly Class** – Distribution insights
8. **3D Anomaly Detection Plot** – Torque, Temp, RPM analysis

---

## 🛠 Requirements
```text
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
joblib
streamlit
```
Save the above into a `requirements.txt` and install using:
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess the Data
```bash
python src/preprocess.py
```

### 3. Train and Predict
```bash
python src/train_model.py
python src/predict.py
```

### 4. Run Anomaly Detection & Visualize Results
```bash
python src/anomaly_detection.py
python src/visualize_results.py
```

---

## 📘 Conceptual Study
👉 [Click here to read the full conceptual study](./conceptual_study_predictive_maintenance.pdf)

Includes:
- ML Theory for Predictive Maintenance
- Business Use Cases
- Feature Engineering & Model Design
- Deployment Strategy

---
## 📜 License

This project is open for educational use only. For commercial deployment, contact the author.

---

## 📬 Contact
If you'd like to learn more or collaborate on projects or other initiatives, feel free to connect on [LinkedIn](https://www.linkedin.com/in/prerna-burande-99678a1bb/) or check out my [portfolio site](https://youtheleader.com/).
