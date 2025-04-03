import streamlit as st
import pandas as pd
import numpy as np
import pickle

#LOad saved model and scaler
with open ("iris_svm_model.pkl", "rb") as f: #Read binary
    model = pickle.load(f)

with open ("iris_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

#App UI
st.title("Iris Flower Classifier Usign Support Vector Machines")
st.write("Input sepal and petal measurements below: ")

sepal_length = st.slider("Sepal Length (cm)",4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal with (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider ("PEtal length (cm)", 1.0, 7.0, 4.2)
petal_width = st.slider("Petal width (cm)", 0.1, 3.0, 1.3)

#Predift
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)
prediciton = model.predict(input_scaled)[0]

labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
st.subheader("Prediciton: ")
st.success(f"This flower is likely **{labels[prediction]}**")