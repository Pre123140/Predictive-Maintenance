import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset

import pandas as pd
import glob

# Search for any CSV file in the data folder
file_list = glob.glob(r"D:\PERSONAL\PORTFOLIO\TECHNICAL PROJECTS\PREDICTIVE MAINTENANCE\predictive_maintenance_project\data\*.csv")

if not file_list:
    raise FileNotFoundError("No CSV file found in the data folder!")

# Use the first CSV found
file_path = file_list[0]
print(f"Using file: {file_path}")

df = pd.read_csv(file_path)
print(df.head())


# Drop unnecessary columns
df = df.drop(columns=["UDI", "Product ID"])

# Encode categorical variable 'Type'
encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])  # Convert L, M, H → 0, 1, 2

# Save preprocessed data
df.to_csv("data/processed_data.csv", index=False)
print("✅ Data Preprocessing Completed!")
