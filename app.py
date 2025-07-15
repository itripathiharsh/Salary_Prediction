import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# Load model files from Google Drive if not present
def download_model_file(gdrive_id, output_name):
    if not os.path.exists(output_name):
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, output_name, quiet=False)

# Google Drive File IDs
download_model_file("1iqNBtQFpmAF_BkQsnFPmxiw2vRfaQKAO", "income_rf_model.pkl")
download_model_file("1vPgZw4QNS3nCggI2yrGbOkoXeS7R8CrF", "scaler.pkl")
download_model_file("1yTBC7n8rD1zJrwNkIie5jErJxn9PfVfH", "full_columns.pkl")

# Load model and preprocessing tools
model = joblib.load('income_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
full_columns = joblib.load('full_columns.pkl')

# Streamlit page config
st.set_page_config(page_title="Income Prediction", layout="centered", page_icon="💰")
st.title("💼 Income Prediction App")


def user_input_features():
    st.subheader("Enter Employee Details:")

    age = st.slider('Age', 17, 90, 30)

    workclass = st.selectbox('Workclass', ['Private', 'Government', 'Self-employed', 'Unemployed'])

    education = st.selectbox('Education Level', [
    '10th', 'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc',
    'Bachelors', 'Masters', 'Doctorate'
    ])

    gender = st.radio('Gender', ['Male', 'Female', 'Others'])


    marital_status = st.selectbox('Marital Status', [
        'Married', 'Divorced', 'Never married', 'Separated', 'Widowed'
    ])

    occupation = st.selectbox('Occupation', [
        'Tech support', 'Craft repair', 'Sales', 'Exec-managerial', 'Professional',
        'Handlers/cleaners', 'Machine operator', 'Admin/clerical',
        'Farming/fishing', 'Transport', 'Protective services', 'Armed Forces', 'Other'
    ])

    relationship = st.selectbox('Relationship Status', [
        'Husband', 'Wife', 'Not-in-family', 'Own-child', 'Unmarried', 'Other-relative'
    ])

    race = st.selectbox('Race', [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ])


    capital_gain = st.number_input('Capital Gain', min_value=0, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, value=0)
    hours_per_week = st.slider('Hours per Week', 1, 99, 40)

    native_country = st.selectbox('Country of Origin', [
        'United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada',
        'China', 'England', 'Japan', 'France', 'Other'
    ])

    data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    return pd.DataFrame(data, index=[0])


# Run app
input_df = user_input_features()
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=full_columns, fill_value=0)
input_scaled = scaler.transform(input_encoded)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display results
st.subheader("🧾 Prediction Result")
label = '>50K' if prediction[0] == 1 else '<=50K'
probability = prediction_proba[0][1]

if prediction[0] == 1:
    st.success(f"Prediction: **{label}** ✅")
else:
    st.warning(f"Prediction: **{label}** ⚠️")

st.info(f"🔢 Probability of >50K Income: **{probability:.2f}**")
