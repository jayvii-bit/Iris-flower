import pandas as pd
import numpy as np 
import pickle
import streamlit as st

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

st.title('Iris Flower Classifier')

# Input fields
sepal_length = st.text_input('Sepal Length')
sepal_width = st.text_input('Sepal Width')
petal_length = st.text_input('Petal Length')
petal_width = st.text_input('Petal Width')


# Function to Make Prediction 

def predict(sepal_length, sepal_width, petal_length, petal_width):
    try:
        # COnvert inputs to float and create dataframe 
        input_data = {
            'sepal_length': float(sepal_length),
            'sepal_width' : float(sepal_width),
            'petal_length': float(petal_length),
            'petal_width': float(petal_width)
            }


        X = pd.DataFrame([input_data])

        # Make Prediction
        prediction = model.predict(X)[0]

        # Display result
        flower_types = {0: "setosa", 1: "versicolor", 2: "virginica"}
        flower_name = flower_types.get(prediction, 'unknown')
        st.success(f'This Iris Flower Specie is {flower_name}')

    except ValueError: 
        st.error("Please enter a valid number")


# Button to make prediction
if st.button('predict'):
    predict(sepal_length, sepal_width, petal_length, petal_width)