import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
# Load the trained model and preprocessing objects
model = tf.keras.models.load_model('churn_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehotencoder.pkl', 'rb') as f:
    onehotencoder = pickle.load(f)  
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# Streamlit app
st.title('Customer Churn Prediction')
# Input fields  
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
tenure = st.slider('Tenure', 0,10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
estimated_salary = st.number_input('Estimated Salary')
geography = st.selectbox('Geography', onehotencoder.categories_[0])
credit_score = st.slider('Credit Score', 350, 850)

input_data = pd.DataFrame({
    'CreditScore': [credit_score],  # Placeholder value
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

# one hot encode geography column using loaded onehotencoder.pkl file
geography_encoded = onehotencoder.transform([[geography]])
geography_df = pd.DataFrame(geography_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))
# combine one hot encoded geography columns with input dataframe
input_data = pd.concat([input_data, geography_df], axis=1)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
predicted_class = (prediction > 0.5).astype(int)

if predicted_class[0][0] == 1:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is unlikely to churn.')
    