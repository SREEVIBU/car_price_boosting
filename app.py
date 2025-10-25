import gradio as gr
import pandas as pd
import numpy as np
import pickle

with open("gbr_model.pkl", "rb") as f:
    gbr = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("categorical_options.pkl", "rb") as f:
    options_dict = pickle.load(f)

numeric_cols = [
    'Year', 'Length', 'Width', 'Height', 'Seating Capacity',
    'Fuel Tank Capacity', 'Max_Power_Cleaned', 'Max_Torque_Cleaned', 
    'Engine_Cleaned', 'Kilometer'
]
label_cols = ['Make', 'Model', 'Location']
onehot_cols = ['Fuel Type', 'Transmission', 'Color', 'Owner', 'Seller Type', 'Drivetrain']

def predict_price(
    Make, Model, Location,
    Year, Length, Width, Height, Seating_Capacity,
    Fuel_Tank_Capacity, Max_Power_Cleaned, Max_Torque_Cleaned, Engine_Cleaned, Kilometer,
    Fuel_Type, Transmission, Color, Owner, Seller_Type, Drivetrain
):
    data = {}
    for col, val in zip(label_cols, [Make, Model, Location]):
        le = encoders[f'le_{col.lower()}']
        data[col] = [le.transform([val])[0]]

    num_values = [Year, Length, Width, Height, Seating_Capacity, Fuel_Tank_Capacity,
                  Max_Power_Cleaned, Max_Torque_Cleaned, Engine_Cleaned, Kilometer]
    for col, val in zip(numeric_cols, num_values):
        data[col] = [val]

    df_input = pd.DataFrame(data)

    for col, val in zip(onehot_cols, [Fuel_Type, Transmission, Color, Owner, Seller_Type, Drivetrain]):
        for option in options_dict[col]:
            col_name = f"{col}_{option}" if option != options_dict[col][0] else col
            df_input[col_name] = [1 if val == option else 0]

    model_cols = gbr.feature_names_in_
    for col in model_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[model_cols]

    predicted_price = gbr.predict(df_input)[0]
    return f"💰 Predicted Price: ₹ {predicted_price:,.2f}"

with gr.Blocks(title="🚗 Ultimate Car Price Predictor") as demo:

    gr.HTML("""
    <h1 style='text-align:center; color:#1f2937; font-family:Arial Black; font-size:48px;'>
        🚗 Ultimate Car Price Predictor
    </h1>
    <p style='text-align:center; color:#4b5563; font-size:20px;'>
        Instantly predict car prices with our advanced Gradient Boosting model! ✨
    </p>
    <hr style='border:2px solid #e5e7eb; margin-bottom:30px'>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🏎 Vehicle Details")
            Make_input = gr.Dropdown(choices=options_dict['Make'], label="Make", value=options_dict['Make'][0])
            Model_input = gr.Dropdown(choices=options_dict['Model'], label="Model", value=options_dict['Model'][0])
            Location_input = gr.Dropdown(choices=options_dict['Location'], label="Location", value=options_dict['Location'][0])
            Year_input = gr.Number(label="Year", value=2020)
            Kilometer_input = gr.Number(label="Kilometers")
            Length_input = gr.Number(label="Length (mm)")
            Width_input = gr.Number(label="Width (mm)")
            Height_input = gr.Number(label="Height (mm)")
            Seating_input = gr.Number(label="Seating Capacity")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Engine & Fuel Details")
            Fuel_Tank_input = gr.Number(label="Fuel Tank Capacity (L)")
            Max_Power_input = gr.Number(label="Max Power (bhp)")
            Max_Torque_input = gr.Number(label="Max Torque (Nm)")
            Engine_input = gr.Number(label="Engine Capacity (cc)")
            Fuel_Type_input = gr.Dropdown(choices=options_dict['Fuel Type'], label="Fuel Type")
            Transmission_input = gr.Dropdown(choices=options_dict['Transmission'], label="Transmission")
            Color_input = gr.Dropdown(choices=options_dict['Color'], label="Color")
            Owner_input = gr.Dropdown(choices=options_dict['Owner'], label="Owner Type")
            Seller_input = gr.Dropdown(choices=options_dict['Seller Type'], label="Seller Type")
            Drivetrain_input = gr.Dropdown(choices=options_dict['Drivetrain'], label="Drivetrain")

    predict_button = gr.Button("🚀 Predict Price!", variant="primary")

    output_price = gr.Textbox(label="Predicted Price", interactive=False)

    predict_button.click(
        fn=predict_price,
        inputs=[
            Make_input, Model_input, Location_input, Year_input, Length_input, Width_input,
            Height_input, Seating_input, Fuel_Tank_input, Max_Power_input, Max_Torque_input,
            Engine_input, Kilometer_input, Fuel_Type_input, Transmission_input, Color_input,
            Owner_input, Seller_input, Drivetrain_input
        ],
        outputs=output_price
    )

demo.launch(share=True)
