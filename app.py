import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_edu.pkl', 'rb') as file:
    onehot_encoder_edu = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Resume Shortlist')

# User input
Education_Level = st.selectbox('Education Level', onehot_encoder_edu.categories_[0])
Experience_Years = st.number_input('Experience Years')
Skills_Score = st.number_input('Skills Score')
Certifications = st.number_input('Certifications')
Previous_Companies = st.number_input('Previous Companies')
Job_Match_Score = st.number_input('Job Match Score')

# Prepare input data with consistent column names
input_data = pd.DataFrame({
    'Experience_Years': [Experience_Years],
    'Skills_Score': [Skills_Score],
    'Certifications': [Certifications],
    'Previous_Companies': [Previous_Companies],
    'Job_Match_Score': [Job_Match_Score]
})

# One-hot encode the education level
edu_encoded = onehot_encoder_edu.transform([[Education_Level]]).toarray()
edu_encoded_df = pd.DataFrame(edu_encoded, columns=onehot_encoder_edu.get_feature_names_out(['Education_Level']))

# Concatenate input data with encoded education level
input_data = pd.concat([input_data.reset_index(drop=True), edu_encoded_df.reset_index(drop=True)], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict when the button is clicked
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    if prediction_probability > 0.5:
        st.success("You are suitable for the job!")
    else:
        st.error("You are not suitable for the job.")
