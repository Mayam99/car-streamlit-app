import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the dataset
car_prices_df = pd.read_csv("CarPricesPrediction.csv")

# Drop unnecessary column and prepare features and target
car_prices_df = car_prices_df.drop(columns=['Unnamed: 0'])
X = car_prices_df.drop(columns=['Price'])
y = car_prices_df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing: OneHotEncode categorical variables
categorical_features = ['Make', 'Model', 'Condition']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Streamlit app
st.title("Car Price Prediction App")

# User inputs
make = st.selectbox('Make', car_prices_df['Make'].unique())
model = st.selectbox('Model', car_prices_df['Model'].unique())
year = st.number_input('Year', min_value=1990, max_value=2023, value=2015, step=1)
mileage = st.number_input('Mileage', min_value=0, max_value=300000, value=50000, step=1000)
condition = st.selectbox('Condition', car_prices_df['Condition'].unique())

# Predict button
if st.button('Predict Price'):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[make, model, year, mileage, condition]],
                              columns=['Make', 'Model', 'Year', 'Mileage', 'Condition'])
    # Predict the price
    predicted_price = pipeline.predict(input_data)
    st.write(f"The estimated price of the car is: ${predicted_price[0]:,.2f}")
