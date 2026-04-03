import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Load data
df = pd.read_csv(r"C:\Users\mikes\Documents\DDM\T06 Data-mining\gym_membership.csv")

#Clean data
cleanDF = df.drop(["id","birthday","name_personal_trainer"], axis=1)

#Split and label
X = cleanDF.drop("personal_training", axis=1)
y = cleanDF["personal_training"]

#Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

#Normalize numerical features
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

num_cols = X.select_dtypes(include=["int64","float64"]).columns
X[num_cols] = X[num_cols].apply(normalize, axis=0)

#Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Streamlit UI
st.title("🏋️ Gym Personal Training Predictor")

st.write("Enter member details:")

#Input fields
age = st.slider("Age", 16, 70, 30)
visits = st.slider("Visits per week", 0, 7, 3)
time = st.slider("Average time in gym (minutes)", 30, 180, 60)

#Create input
input_data = pd.DataFrame({
    "Age": [age],
    "visit_per_week": [visits],
    "avg_time_in_gym": [time]
})

#Match columns
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

#Predict
prediction = knn.predict(input_data)

#Output
st.subheader("Prediction:")

if prediction[0]:
    st.success("✅ This member is likely to use personal training.")
else:
    st.error("❌ This member is unlikely to use personal training.")