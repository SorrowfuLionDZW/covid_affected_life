import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

filename = 'decision_tree_model_covid.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Covid-19 Psychological Effects Prediction App')
st.subheader('Please enter your data:')

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Displaying the first few rows of the uploaded file
    st.write("Uploaded file preview:")
    st.write(df.head())

    # Check for the 'age' column
    if 'age' in df.columns:
        age_mapping = {'19-25': 22, 'Dec-18': 18, '33-40': 36.5, '60+': 65, '26-32': 29, '40-50': 45, '50-60': 55}
        df['age'] = df['age'].map(age_mapping)
    else:
        st.write("Warning: 'age' column not found in the uploaded file!")

    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    object_columns = df.select_dtypes(include=['object']).columns

    # Preprocess numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    df_numerical = pd.DataFrame(numerical_transformer.fit_transform(df[numerical_columns]), columns=numerical_columns)
    df_categorical = pd.get_dummies(df[object_columns], columns=object_columns)

    # Combine processed data and reindex to match original model input
    df_preprocessed = df_numerical.join(df_categorical)
    df_preprocessed = df_preprocessed.reindex(columns=columns_list, fill_value=0)

    # Make prediction using the preprocessed data
    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    
    st.subheader('Lifestyle Change Prediction:')
    st.write(prediction_text)