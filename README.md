# 🔧 AI-Powered Predictive Maintenance for Manufacturing

## 📌 Overview
This project leverages machine learning and anomaly detection to forecast machine failures using real-world manufacturing sensor data. By combining supervised classification and unsupervised anomaly detection, it enables smarter maintenance planning and minimizes unexpected downtimes.

## 🚀 Objective
- Predict machine failure using operational features like torque, tool wear, and temperature.
- Detect outliers using anomaly detection techniques.
- Visualize patterns to help maintenance teams make data-informed decisions.

## 💼 Business Impact
Predictive maintenance reduces unscheduled downtimes, cuts repair costs, enhances safety, and supports real-time monitoring across production floors.

---

## 🧠 Algorithms Used
- **Random Forest Classifier** for machine failure prediction
- **Isolation Forest** for anomaly detection

---

## 🧪 Dataset
- **Source**: `predictive_maintenance.csv`
- **Key Features**:
  - Type, Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min]
  - Failure Modes: TWF, HDF, PWF, OSF, RNF
  - Target: `Machine failure`

---

## 🔄 Project Flow
1. **Data Preprocessing**: Dropped irrelevant columns, encoded categorical fields
2. **Model Training**: Trained Random Forest on core operational features
3. **Anomaly Detection**: Applied Isolation Forest to identify outliers
4. **Evaluation & Visualization**: Visual analytics using Seaborn, Matplotlib, Plotly
5. **Dashboard Prototype**: Created using `streamlit` for failure insights *(optional preview only)*

---

## 📂 Folder Structure
---

## 🧠 Techniques Used
- **Anomaly Detection**: Isolation Forest
- **Failure Prediction**: Random Forest Classifier
- **Feature Importance Analysis**
- **Interactive Visualizations**
- **Dashboard (Streamlit Prototype)**

---

## 📂 Folder Structure
predictive_maintenance_project/
│
├── data/
│   ├── predictive_maintenance.csv             # Original raw dataset
│   ├── processed_data.csv                     # Preprocessed clean dataset
│   ├── predictions.csv                        # Initial prediction results
│   ├── predictions_with_timestamp.csv         # Time-aware predictions
│   ├── predictions_with_rf.csv                # Final output from Random Forest
│   ├── anomaly_results.csv                    # Output from anomaly detection
│
├── models/
│   ├── failure_prediction_model.pkl           # Trained Random Forest model
│   ├── predictive_maintenance_model.pkl       # Optional model object
│   ├── feature_names.pkl                      # Feature label encoder object
│
├── output/
│   ├── Figure_1.png       # Anomaly Detection Plot (Torque vs RPM)
│   ├── Figure_2.png       # Feature Importance Bar Plot
│   ├── Figure_3.png       # Anomalies per Failure Mode (Grouped Bar)
│   ├── Figure_4.png       # Combined Anomaly + Failure Mode Scatterplot
│   ├── Figure_5.png       # Correlation Heatmap of Parameters
│   ├── Figure_6.png       # Failure Rate by Machine Type (Bar Chart)
│   ├── Figure_7.png       # Torque Boxplot by Anomaly Class
│   ├── Picture_8.png      # 3D Anomaly Detection Plot
│
├── src/
│   ├── preprocess.py                # Clean and encode raw data
│   ├── train_model.py               # Build & evaluate Random Forest model
│   ├── predict.py                   # Generate predictions on test data
│   ├── anomaly_detection.py        # Run Isolation Forest & plot insights
│   ├── visualize_results.py        # Streamlit-based dashboard visuals
│
├── conceptual_study_predictive_maintenance.pdf   # 📘 Full conceptual write-up
│
├── README.md                        # Project overview + how to run
├── requirements.txt                 # All required Python libraries











---

## 📊 Key Visual Outputs

- Anomaly Detection (Torque vs RPM)
- Feature Importance by Random Forest
- Failure Modes vs Anomalies
- 3D Anomaly Plot (Torque + Temp + RPM)
- Heatmap of Sensor Correlations
- Failure Rates by Machine Type

---

## 📦 Dependencies

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`, `plotly`
- `joblib`
- `streamlit`

Install all at once:
```bash
pip install -r requirements.txt


🚀 How to Run
Preprocess the dataset - python preprocess.py
Train model and generate predictions - python train_model.py; python predict.py
Run anomaly detection and visualizations : python anomaly_detection.py, python visualize_results.py

📘 Resources Included
predictive_maintenance.csv – Original dataset
processed_data.csv, predictions_with_rf.csv, anomaly_results.csv – Outputs

Visual insights & plots - Streamlit dashboard prototype

---

## 📊 Key Visual Outputs (8 Plots)

1. **Anomaly Detection Plot** – Torque vs RPM with outliers in red  
2. **Feature Importance Plot** – Top features contributing to machine failure  
3. **Anomalies per Failure Mode** – Bar plot showing anomaly distribution by failure type  
4. **Overlay Plot of Anomaly + Machine Failure** – Failure vs Anomaly scatterplot  
5. **Correlation Heatmap** – Sensor features vs failure correlation  
6. **Failure Rate by Machine Type** – Bar plot grouped by machine type  
7. **Torque Boxplot by Anomaly** – Distribution of torque across anomaly classes  
8. **3D**


✅ Highlights
Predictive analytics with real-world relevance
Explainable visual outputs
Fully modular and extensible code
Practical application in smart manufacturing setups

## 📘 In-Depth Conceptual Study

→ [Click to Read the Full Conceptual Study PDF](./Conceptual_Study_Predictive_Maintenance.pdf)

This companion study includes:
- Theoretical background on predictive maintenance
- Intuition behind Random Forests and Isolation Forests
- Industry use cases and extensions
- Visual interpretations and modeling decisions


🧠 Author
Prerna Burande
🔗 LinkedIn | 🌐 Portfolio Website

⚠️ Disclaimer
This project is for educational and illustrative purposes only. It is based on publicly available datasets and open-source tools. Not intended for commercial use without prior written permission.

