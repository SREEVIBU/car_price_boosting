import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load trained model and encoders ---
with open("gbr_model.pkl", "rb") as f:
    gbr = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("categorical_options.pkl", "rb") as f:
    options_dict = pickle.load(f)

# --- Column definitions ---
numeric_cols = [
    'Year', 'Length', 'Width', 'Height', 'Seating Capacity',
    'Fuel Tank Capacity', 'Max_Power_Cleaned', 'Max_Torque_Cleaned', 
    'Engine_Cleaned', 'Kilometer'
]
label_cols = ['Make', 'Model', 'Location']
onehot_cols = ['Fuel Type', 'Transmission', 'Color', 'Owner', 'Seller Type', 'Drivetrain']

# --- Prediction function ---
def predict_price(
    Make, Model, Location,
    Year, Length, Width, Height, Seating_Capacity,
    Fuel_Tank_Capacity, Max_Power_Cleaned, Max_Torque_Cleaned, Engine_Cleaned, Kilometer,
    Fuel_Type, Transmission, Color, Owner, Seller_Type, Drivetrain
):
    data = {}

    # Label Encoded columns
    for col, val in zip(label_cols, [Make, Model, Location]):
        le = encoders[f'le_{col.lower()}']
        data[col] = [le.transform([val])[0]]

    # Numeric columns
    num_values = [Year, Length, Width, Height, Seating_Capacity, Fuel_Tank_Capacity,
                  Max_Power_Cleaned, Max_Torque_Cleaned, Engine_Cleaned, Kilometer]
    for col, val in zip(numeric_cols, num_values):
        data[col] = [val]

    df_input = pd.DataFrame(data)

    # One-hot encoded columns
    for col, val in zip(onehot_cols, [Fuel_Type, Transmission, Color, Owner, Seller_Type, Drivetrain]):
        for option in options_dict[col]:
            col_name = f"{col}_{option}" if option != options_dict[col][0] else col
            df_input[col_name] = [1 if val == option else 0]

    # Align with model columns
    model_cols = gbr.feature_names_in_
    for col in model_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[model_cols]

    # Predict
    predicted_price = gbr.predict(df_input)[0]
    return predicted_price

# --- Streamlit UI ---
st.set_page_config(page_title="🚗 Ultimate Car Price Predictor", layout="wide", page_icon="🚗")

st.markdown("""
<h1 style='text-align:center; color:#1f2937; font-family:Arial Black; font-size:48px;'>
🚗 Ultimate Car Price Predictor
</h1>
<p style='text-align:center; color:#4b5563; font-size:20px;'>
Instantly predict car prices with our advanced Gradient Boosting model! ✨
</p>
<hr style='border:2px solid #e5e7eb; margin-bottom:30px'>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# --- Left Column: Vehicle Details ---
with col1:
    st.subheader("🏎 Vehicle Details")
    Make = st.selectbox("Make", options_dict['Make'])
    Model = st.selectbox("Model", options_dict['Model'])
    Location = st.selectbox("Location", options_dict['Location'])
    Year = st.number_input("Year", min_value=1990, max_value=2025, value=2020)
    Kilometer = st.number_input("Kilometers Driven", min_value=0)
    Length = st.number_input("Length (mm)", min_value=0)
    Width = st.number_input("Width (mm)", min_value=0)
    Height = st.number_input("Height (mm)", min_value=0)
    Seating_Capacity = st.number_input("Seating Capacity", min_value=1, max_value=10, value=5)

# --- Right Column: Engine & Fuel Details ---
with col2:
    st.subheader("⚙️ Engine & Fuel Details")
    Fuel_Tank_Capacity = st.number_input("Fuel Tank Capacity (L)", min_value=0)
    Max_Power_Cleaned = st.number_input("Max Power (bhp)", min_value=0)
    Max_Torque_Cleaned = st.number_input("Max Torque (Nm)", min_value=0)
    Engine_Cleaned = st.number_input("Engine Capacity (cc)", min_value=0)
    Fuel_Type = st.selectbox("Fuel Type", options_dict['Fuel Type'])
    Transmission = st.selectbox("Transmission", options_dict['Transmission'])
    Color = st.selectbox("Color", options_dict['Color'])
    Owner = st.selectbox("Owner Type", options_dict['Owner'])
    Seller_Type = st.selectbox("Seller Type", options_dict['Seller Type'])
    Drivetrain = st.selectbox("Drivetrain", options_dict['Drivetrain'])

# --- Predict Button ---
if st.button("🚀 Predict Price!"):
    try:
        predicted_price = predict_price(
            Make, Model, Location, Year, Length, Width, Height, Seating_Capacity,
            Fuel_Tank_Capacity, Max_Power_Cleaned, Max_Torque_Cleaned,
            Engine_Cleaned, Kilometer, Fuel_Type, Transmission, Color,
            Owner, Seller_Type, Drivetrain
        )
        st.success(f"💰 **Predicted Price:** ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
