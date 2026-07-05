import streamlit as st
import joblib

model = joblib.load("model/model.pkl")

st.title("🌸 Iris Flower Classification")

st.write("Predict Iris Flower Species")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):

    prediction = model.predict([[sl, sw, pl, pw]])

    labels = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(labels[prediction[0]])