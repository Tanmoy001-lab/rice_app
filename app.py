import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rice AI Secure", layout="centered")

# --- HIDE STREAMLIT BRANDING (Nuclear Option) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            [data-testid="stToolbar"] {visibility: hidden !important;}
            [data-testid="stDecoration"] {visibility: hidden !important;}
            [data-testid="stStatusWidget"] {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 🔐 SECURITY SYSTEM (LOGIN)
# ==========================================
# 👇 CHANGE YOUR PASSWORD HERE:
SECRET_PASSWORD = "rice1234" 

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Login Required")
    st.write("Private Access Only.")
    
    password_input = st.text_input("Enter Password", type="password")
    
    if st.button("Login"):
        if password_input == SECRET_PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Wrong Password!")
    st.stop() # 🛑 STOPS HERE if not logged in

# ==========================================
# ✅ MAIN APP STARTS HERE
# ==========================================

# ⚠️ YOUR GOOGLE SHEET LINK:
SHEET_URL = "https://docs.google.com/spreadsheets/d/10lXOiNCfJDnz5bvTtTydEcLfX4FSirfxvitum4udmNs/edit?gid=0#gid=0"

# Connect to the robot
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    try:
        return conn.read(spreadsheet=SHEET_URL, worksheet="Sheet1", ttl=0)
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

def add_data(row):
    df = get_data()
    new_df = pd.DataFrame([row])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    conn.update(spreadsheet=SHEET_URL, worksheet="Sheet1", data=updated_df)

# --- VISION SYSTEM ---
def predict_age_from_image(uploaded_image):
    try:
        img = Image.open(uploaded_image)
        img = img.resize((100, 100))
        img_array = np.array(img)
        avg_r = np.mean(img_array[:, :, 0])
        avg_b = np.mean(img_array[:, :, 2])
        if avg_b < (avg_r * 0.9): 
            return "Old (>1 Year)", "Yellowish tint detected."
        else:
            return "New (<6 Months)", "Bright white color detected."
    except:
        return "Unknown", "Could not analyze image."

# --- AI TRAINING SYSTEM ---
def train_and_predict_all(input_features):
    df = get_data()
    if len(df) < 2:
        return "Not enough data"
    
    X = df[['protein', 'hardness', 'moisture']].values
    y_use = df['use'].astype(str)
    y_age = df['age'].astype(str)
    y_sugg = df['suggestion'].astype(str)

    model_use = RandomForestClassifier().fit(X, y_use)
    pred_use = model_use.predict([input_features])[0]

    model_age = RandomForestClassifier().fit(X, y_age)
    pred_age = model_age.predict([input_features])[0]

    model_sugg = RandomForestClassifier().fit(X, y_sugg)
    pred_sugg = model_sugg.predict([input_features])[0]

    return pred_use, pred_age, pred_sugg

# --- APP INTERFACE ---
st.title("🌾 Rice Quality AI Pro")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

tab1, tab2 = st.tabs(["📸 Train (Add Data)", "🔮 Predict (Vision + AI)"])

# --- TAB 1: ADD DATA ---
with tab1:
    st.header("Add to Knowledge Base")
    img_source = st.radio("Image Source:", ["Camera", "Upload File"], horizontal=True, key="train_source")
    
    if img_source == "Camera":
        img = st.camera_input("Take a photo", key="train_cam")
    else:
        img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'], key="train_upload")

    c1, c2 = st.columns(2)
    with c1:
        d_age = st.selectbox("Rice Age", ["New (<6 Months)", "Mid (6-12 Months)", "Old (>1 Year)"])
        d_use = st.selectbox("Best Use", ["Biryani", "Daily Rice", "Porridge", "Fried Rice", "Idli/Dosa"])
    with c2:
        d_prot = st.slider("Protein (%)", 0, 100, 10)
        d_hard = st.slider("Hardness (0-100)", 0, 100, 50)
        d_moist = st.slider("Moisture (%)", 0, 100, 20)

    d_sugg = st.text_input("Expert Suggestion", "Excellent quality for cooking")

    if st.button("Save to Database"):
        if img is None:
            st.warning("⚠️ Please provide an image first.")
        else:
            add_data({
                "date": str(pd.Timestamp.now()), "age": d_age, "use": d_use, 
                "protein": d_prot, "hardness": d_hard, "moisture": d_moist, "suggestion": d_sugg
            })
            st.success("✅ Saved!")

# --- TAB 2: PREDICT ---
with tab2:
    st.header("Predict Quality")
    
    # 1. Vision Input (Upload or Camera)
    pred_source = st.radio("Image Source:", ["Camera", "Upload File"], horizontal=True, key="pred_source")
    
    p_img = None
    if pred_source == "Camera":
        p_img = st.camera_input("Scan Rice", key="pred_cam")
    else:
        p_img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'], key="pred_upload")
    
    detected_age_visual = "No Photo"
    if p_img:
        detected_age_visual, reason = predict_age_from_image(p_img)
        st.info(f"📸 **Vision Analysis:** I think this rice is **{detected_age_visual}**")
        st.caption(f"Reason: {reason}")
    
    st.write("---")
    p_prot = st.slider("Detected Protein", 0, 100, 10, key="p_p")
    p_hard = st.slider("Detected Hardness", 0, 100, 50, key="p_h")
    p_moist = st.slider("Detected Moisture", 0, 100, 20, key="p_m")

    if st.button("Analyze Full Results"):
        with st.spinner("Thinking..."):
            features = [p_prot, p_hard, p_moist]
            result = train_and_predict_all(features)
            
            if result == "Not enough data":
                st.error("⚠️ Not enough training data!")
            else:
                res_use, res_age, res_sugg = result
                st.balloons()
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Use", res_use)
                c2.metric("Visual Age", detected_age_visual if p_img else res_age)
                c3.metric("Suggestion", "See below")
                st.info(f"💡 **Suggestion:** {res_sugg}")