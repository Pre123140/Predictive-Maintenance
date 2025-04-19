import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_csv("data/predictions.csv")

st.title("Predictive Maintenance Dashboard")

# Visualization 1: Machine failure distribution
st.subheader("Failure Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Machine failure", data=df, ax=ax)
st.pyplot(fig)

# Add more visualizations as needed...
