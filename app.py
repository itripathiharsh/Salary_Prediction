import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load('income_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
full_columns = joblib.load('full_columns.pkl')  

st.set_page_config(page_title="Income Prediction", layout="centered", page_icon="💰")
st.title("💼 Income Prediction App")


def user_input_features():
    st.subheader("Enter Employee Details:")
    
    age = st.number_input('Age', min_value=17, max_value=90, value=30)
    
    workclass = st.selectbox('Workclass', [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ])
    
    education = st.selectbox('Education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool'
    ])
    
    marital_status = st.selectbox('Marital Status', [
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
        'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ])
    
    occupation = st.selectbox('Occupation', [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces'
    ])
    
    relationship = st.selectbox('Relationship', [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    
    race = st.selectbox('Race', [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ])
    
    gender = st.selectbox('Gender', ['Male', 'Female'])
    
    capital_gain = st.number_input('Capital Gain', min_value=0, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, value=0)
    hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=99, value=40)
    
    native_country = st.selectbox('Native Country', [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
        'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
        'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines',
        'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
        'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
        'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
        'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
        'Peru', 'Hong', 'Holand-Netherlands'
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

input_df = user_input_features()


input_encoded = pd.get_dummies(input_df)


input_encoded = input_encoded.reindex(columns=full_columns, fill_value=0)

input_scaled = scaler.transform(input_encoded)

prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("🧾 Prediction Result")
label = '>50K' if prediction[0] == 1 else '<=50K'
probability = prediction_proba[0][1]

if prediction[0] == 1:
    st.success(f"Prediction: **{label}** ✅")
else:
    st.warning(f"Prediction: **{label}** ⚠️")

st.info(f"🔢 Probability of >50K Income: **{probability:.2f}**")
