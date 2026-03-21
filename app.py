import streamlit as st
import pandas as pd
import numpy as np
import pickle
import uuid
from sqlalchemy import create_engine, text

# ==============================
# 🔹 DATABASE CONFIG (CHANGE HERE)
# ==============================
DATABASE_URL = "postgresql://liver_user:U0MomnFuiYa6WsYv4LQLbMhc9rnB2enY@dpg-d6v706lm5p6s73a63nk0-a.oregon-postgres.render.com/liver"

# Create engine with SSL (IMPORTANT for Render)
engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True
)

# ==============================
# 🔹 INIT DATABASE (SAFE)
# ==============================
def init_db():
    try:
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
    except Exception as e:
        st.error(f"❌ Database Init Error: {e}")

# Call once
init_db()

# ==============================
# 🔹 LOAD MODEL FILES
# ==============================
try:
    model = pickle.load(open("liver_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    FEATURE_ORDER = pickle.load(open("feature_order.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ==============================
# 🔹 STREAMLIT UI
# ==============================
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

# ==============================
# 🔹 PREDICTION LOGIC
# ==============================
if st.button("🔍 Predict"):

    gender_val = 1 if gender == "Male" else 0

    input_data = pd.DataFrame(
        [[age, gender_val, total_bilirubin, direct_bilirubin,
          alkphos, sgpt, sgot, total_proteins, albumin, ag_ratio]],
        columns=[
            "Age of the patient", "Gender", "Total Bilirubin", "Direct Bilirubin",
            "Alkphos Alkaline Phosphotase", "Sgpt Alamine Aminotransferase",
            "Sgot Aspartate Aminotransferase", "Total Protiens",
            "ALB Albumin", "A/G Ratio Albumin and Globulin Ratio"
        ]
    )

    # Ensure correct feature order
    try:
        input_data = input_data[FEATURE_ORDER]
    except KeyError as e:
        st.error(f"❌ Column mismatch: {e}")
        st.stop()

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    result = model.predict(input_scaled)[0]
    result_label = "Positive" if result == 1 else "Negative"

    st.success(f"Prediction: **{result_label}**")

    # ==============================
    # 🔹 SAVE TO DATABASE
    # ==============================
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO liver_user_predictions (
                    user_id, name, phone, age, gender,
                    total_bilirubin, direct_bilirubin,
                    total_proteins, albumin, ag_ratio, result
                ) VALUES (
                    :user_id, :name, :phone, :age, :gender,
                    :total_bilirubin, :direct_bilirubin,
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

        st.info("✅ Prediction saved to database")

    except Exception as e:
        st.error(f"❌ Failed to save data: {e}")
