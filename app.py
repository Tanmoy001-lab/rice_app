import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rice AI Ultimate", layout="centered")

# --- 1. SETUP (PASTE YOUR LINK HERE) ---
# ⚠️ PASTE YOUR GOOGLE SHEET LINK INSIDE THE QUOTES BELOW:
SHEET_URL = "https://docs.google.com/spreadsheets/d/10lXOiNCfJDnz5bvTtTydEcLfX4FSirfxvitum4udmNs/edit?gid=0#gid=0"

# Connect to the robot
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    try:
        # Read the sheet using the hardcoded URL
        return conn.read(spreadsheet=SHEET_URL, worksheet="Sheet1", ttl=0)
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

def add_data(row):
    df = get_data()
    new_df = pd.DataFrame([row])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    conn.update(spreadsheet=SHEET_URL, worksheet="Sheet1", data=updated_df)

def train_and_predict_all(input_features):
    df = get_data()
    if len(df) < 2:
        return "Not enough data"
    
    # Inputs: Protein, Hardness, Moisture
    X = df[['protein', 'hardness', 'moisture']].values
    
    # We train 3 separate brains to predict 3 different things
    y_use = df['use'].astype(str)
    y_age = df['age'].astype(str)
    y_sugg = df['suggestion'].astype(str)

    # Brain 1: Predict Use
    model_use = RandomForestClassifier()
    model_use.fit(X, y_use)
    pred_use = model_use.predict([input_features])[0]

    # Brain 2: Predict Age
    model_age = RandomForestClassifier()
    model_age.fit(X, y_age)
    pred_age = model_age.predict([input_features])[0]

    # Brain 3: Predict Suggestion
    model_sugg = RandomForestClassifier()
    model_sugg.fit(X, y_sugg)
    pred_sugg = model_sugg.predict([input_features])[0]

    return pred_use, pred_age, pred_sugg

# --- 2. APP INTERFACE ---
st.title("🌾 Rice Quality AI Pro")

tab1, tab2 = st.tabs(["📸 Train (Add Data)", "🔮 Predict Full Details"])

# --- TAB 1: ADD DATA ---
with tab1:
    st.header("Add to Knowledge Base")
    
    # 1. Image Option (Camera OR Upload)
    img_source = st.radio("Image Source:", ["Camera", "Upload File"], horizontal=True)
    
    if img_source == "Camera":
        img = st.camera_input("Take a photo")
    else:
        img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    # 2. Details
    col1, col2 = st.columns(2)
    with col1:
        # Added Age Input
        d_age = st.selectbox("Rice Age", ["New (<6 Months)", "Mid (6-12 Months)", "Old (>1 Year)"])
        d_use = st.selectbox("Best Use", ["Biryani", "Daily Rice", "Porridge", "Fried Rice", "Idli/Dosa"])
    
    with col2:
        # Sliders now go to 100
        d_prot = st.slider("Protein (%)", 0, 100, 10)
        d_hard = st.slider("Hardness (0-100)", 0, 100, 50)
        d_moist = st.slider("Moisture (%)", 0, 100, 20)

    d_sugg = st.text_input("Expert Suggestion", "Excellent quality for cooking")

    if st.button("Save to Database"):
        if img is None:
            st.warning("⚠️ Please provide an image first.")
        else:
            row = {
                "date": str(pd.Timestamp.now()), 
                "age": d_age, 
                "use": d_use, 
                "protein": d_prot, 
                "hardness": d_hard, 
                "moisture": d_moist, 
                "suggestion": d_sugg
            }
            add_data(row)
            st.success("✅ Saved! The AI is getting smarter.")

# --- TAB 2: PREDICT ---
with tab2:
    st.header("Predict Quality")
    st.write("Enter values to see what the AI thinks.")
    
    # Prediction Inputs (Also 0-100)
    p_prot = st.slider("Detected Protein", 0, 100, 10, key="p_p")
    p_hard = st.slider("Detected Hardness", 0, 100, 50, key="p_h")
    p_moist = st.slider("Detected Moisture", 0, 100, 20, key="p_m")

    if st.button("Analyze Results"):
        with st.spinner("Analyzing patterns..."):
            features = [p_prot, p_hard, p_moist]
            result = train_and_predict_all(features)
            
            if result == "Not enough data":
                st.error("⚠️ Not enough training data! Please go to the 'Train' tab and add at least 2 samples.")
            else:
                # Unpack the 3 results
                res_use, res_age, res_sugg = result
                
                st.balloons()
                
                # Show results in nice boxes
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Use", res_use)
                c2.metric("Predicted Age", res_age)
                c3.metric("Suggestion", "See below")
                
                st.info(f"💡 **AI Suggestion:** {res_sugg}")