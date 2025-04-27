# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import uuid
from sqlalchemy import create_engine, text

# Database connection
DATABASE_URL = "postgresql://shewak:RsCYFIh2R75LmN2T7t6ZpfE20xhRO5F3@dpg-d076cdpr0fns73891uig-a.oregon-postgres.render.com/liverdb"  # Change this

engine = create_engine(DATABASE_URL)

# Create table (if not exists)
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS liver_user_predictions (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(100),
            name VARCHAR(100),
            phone VARCHAR(20),
            age INTEGER,
            gender INTEGER,
            total_bilirubin FLOAT,
            direct_bilirubin FLOAT,
            total_proteins FLOAT,
            albumin FLOAT,
            ag_ratio FLOAT,
            result VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))

# Load model, scaler, feature order
model = pickle.load(open("liver_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
FEATURE_ORDER = pickle.load(open("feature_order.pkl", "rb"))

# Streamlit UI
st.title("🩺 Liver Disease Prediction App")

st.subheader("👤 User Information")
user_id = str(uuid.uuid4())
name = st.text_input("Name")
phone = st.text_input("Phone Number")

st.subheader("🧪 Medical Inputs")
age = st.number_input("Age", 1, 100)
gender = st.radio("Gender", ["Male", "Female"])
total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0)
direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 5.0)
alkphos = st.number_input("Alkphos Alkaline Phosphotase", 0.0, 300.0)
sgpt = st.number_input("Sgpt Alamine Aminotransferase", 0.0, 100.0)
sgot = st.number_input("Sgot Aspartate Aminotransferase", 0.0, 100.0)
total_proteins = st.number_input("Total Proteins", 0.0, 10.0)
albumin = st.number_input("Albumin", 0.0, 6.0)
ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0)

# Predict Button
if st.button("🔍 Predict"):
    # Map gender input to numeric value (1 for Male, 0 for Female)
    gender_val = 1 if gender == "Male" else 0
    
    # Prepare input data as a DataFrame, making sure to match column names exactly as in the model training
    input_data = pd.DataFrame([[age, gender_val, total_bilirubin, direct_bilirubin, alkphos, sgpt, sgot, total_proteins, albumin, ag_ratio]],
                              columns=["Age of the patient", "Gender", "Total Bilirubin", "Direct Bilirubin", 
                                       "Alkphos Alkaline Phosphotase", "Sgpt Alamine Aminotransferase", 
                                       "Sgot Aspartate Aminotransferase", "Total Protiens", "ALB Albumin", 
                                       "A/G Ratio Albumin and Globulin Ratio"])

    # Debugging: Check the columns of input data and feature order
    

    # Reorder columns to match the feature order from training
    try:
        input_data = input_data[FEATURE_ORDER]
    except KeyError as e:
        st.error(f"KeyError: {e}. This indicates that some columns are missing in the input data. Please verify column names.")
        st.stop()

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict
    result = model.predict(input_scaled)[0]
    result_label = "Positive" if result == 1 else "Negative"

    # Show result
    st.success(f"Prediction: **{result_label}**")

    # Insert prediction data into the database
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO liver_user_predictions (
                user_id, name, phone, age, gender, total_bilirubin, direct_bilirubin,
                total_proteins, albumin, ag_ratio, result
            ) VALUES (
                :user_id, :name, :phone, :age, :gender, :total_bilirubin, :direct_bilirubin,
                :total_proteins, :albumin, :ag_ratio, :result
            )
        """), {
            "user_id": user_id,
            "name": name,
            "phone": phone,
            "age": age,
            "gender": gender_val,
            "total_bilirubin": total_bilirubin,
            "direct_bilirubin": direct_bilirubin,
            "total_proteins": total_proteins,
            "albumin": albumin,
            "ag_ratio": ag_ratio,
            "result": result_label
        })

    st.info("✅ Prediction saved to the database.")
