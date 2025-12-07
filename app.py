
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Rice AI Final", layout="centered")

# --- 1. SETUP (THE FIX) ---
# Paste your Google Sheet Link exactly inside these quotes:
SHEET_URL = "https://docs.google.com/spreadsheets/d/10lXOiNCfJDnz5bvTtTydEcLfX4FSirfxvitum4udmNs/edit?gid=0#gid=0"

# Connect to the robot
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    try:
        # We tell the robot EXACTLY which sheet to look at using the URL
        return conn.read(spreadsheet=SHEET_URL, worksheet="Sheet1", ttl=0)
    except Exception as e:
        # If it fails, we print the error to help debug
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

def add_data(row):
    df = get_data()
    new_df = pd.DataFrame([row])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    # We tell the robot EXACTLY where to save
    conn.update(spreadsheet=SHEET_URL, worksheet="Sheet1", data=updated_df)

# --- 2. APP INTERFACE ---
st.title("🌾 Rice Quality AI")

tab1, tab2 = st.tabs(["📸 Add Data", "🤖 Predict"])

with tab1:
    st.header("Train the AI")
    
    # Inputs
    img = st.camera_input("Take a photo")
    use = st.selectbox("Best Use", ["Biryani", "Daily", "Porridge"])
    prot = st.slider("Protein", 1.0, 15.0, 7.0)
    hard = st.slider("Hardness", 1, 10, 5)
    moist = st.slider("Moisture", 1, 20, 12)
    sugg = st.text_input("Suggestion", "Good")
    
    if st.button("Save Data"):
        row = {
            "date": str(pd.Timestamp.now()), 
            "age": "New", 
            "use": use, 
            "protein": prot, 
            "hardness": hard, 
            "moisture": moist, 
            "suggestion": sugg
        }
        add_data(row)
        st.success("Saved!")

with tab2:
    st.header("Predict")
    if st.button("Run Prediction"):
        df = get_data()
        if len(df) < 2:
            st.warning("Please save at least 2 rows of data first!")
        else:
            X = df[['protein', 'hardness', 'moisture']]
            y = df['use'].astype(str)
            model = RandomForestClassifier()
            model.fit(X, y)
            res = model.predict([[prot, hard, moist]])[0]
            st.success(f"Prediction: {res}")
