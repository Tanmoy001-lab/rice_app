import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rice AI Ultimate", layout="centered")

# --- HIDE STREAMLIT BRANDING (Nuclear Option) ---
hide_st_style = """
            <style>
            /* Hides the Main Menu (top right) */
            #MainMenu {visibility: hidden;}
            
            /* Hides the footer (Made with Streamlit) */
            footer {visibility: hidden;}
            
            /* Hides the header (The colored bar at the top) */
            header {visibility: hidden;}
            
            /* Hides the "Manage App" button (bottom right) */
            .stDeployButton {display:none;}
            
            /* Hides the Toolbar */
            [data-testid="stToolbar"] {visibility: hidden !important;}
            
            /* Hides the decoration */
            [data-testid="stDecoration"] {visibility: hidden !important;}
            
            /* Hides the status widget */
            [data-testid="stStatusWidget"] {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 1. SETUP ---
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

# --- 2. VISION SYSTEM (Detect Age from Photo) ---
def predict_age_from_image(uploaded_image):
    """
    Analyzes the 'Yellowing' of the rice to guess age.
    Old rice = Yellowish (Low Blue).
    New rice = White (High Blue).
    """
    try:
        img = Image.open(uploaded_image)
        img = img.resize((100, 100)) # Resize for speed
        img_array = np.array(img)
        
        # Get Average Red, Green, Blue values
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        # LOGIC: If Blue is significantly lower than Red, it's Yellow (Old)
        if avg_b < (avg_r * 0.9): 
            return "Old (>1 Year)", "Yellowish tint detected."
        else:
            return "New (<6 Months)", "Bright white color detected."
            
    except Exception as e:
        return "Unknown", "Could not analyze image."

# --- 3. AI TRAINING SYSTEM ---
def train_and_predict_all(input_features):
    df = get_data()
    if len(df) < 2:
        return "Not enough data"
    
    # Inputs: Protein, Hardness, Moisture
    X = df[['protein', 'hardness', 'moisture']].values
    
    # We train 3 separate brains
    y_use = df['use'].astype(str)
    y_age = df['age'].astype(str)
    y_sugg = df['suggestion'].astype(str)

    # Brain 1: Predict Use
    model_use = RandomForestClassifier()
    model_use.fit(X, y_use)
    pred_use = model_use.predict([input_features])[0]

    # Brain 2: Predict Age (Based on Stats)
    model_age = RandomForestClassifier()
    model_age.fit(X, y_age)
    pred_age = model_age.predict([input_features])[0]

    # Brain 3: Predict Suggestion
    model_sugg = RandomForestClassifier()
    model_sugg.fit(X, y_sugg)
    pred_sugg = model_sugg.predict([input_features])[0]

    return pred_use, pred_age, pred_sugg

# --- 4. APP INTERFACE ---
st.title("🌾 Rice Quality AI Pro")

tab1, tab2 = st.tabs(["📸 Train (Add Data)", "🔮 Predict (Vision + AI)"])

# --- TAB 1: ADD DATA ---
with tab1:
    st.header("Add to Knowledge Base")
    
    img_source = st.radio("Image Source:", ["Camera", "Upload File"], horizontal=True)
    
    if img_source == "Camera":
        img = st.camera_input("Take a photo")
    else:
        img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    col1, col2 = st.columns(2)
    with col1:
        d_age = st.selectbox("Rice Age", ["New (<6 Months)", "Mid (6-12 Months)", "Old (>1 Year)"])
        d_use = st.selectbox("Best Use", ["Biryani", "Daily Rice", "Porridge", "Fried Rice", "Idli/Dosa"])
    with col2:
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
    st.write("Take a photo to detect Age, then predict quality.")
    
    # 1. Vision Input
    p_img = st.camera_input("Scan Rice for Age Prediction", key="pred_cam")
    
    detected_age_visual = "No Photo"
    
    if p_img:
        # RUN VISION ANALYSIS IMMEDIATELY
        detected_age_visual, reason = predict_age_from_image(p_img)
        st.info(f"📸 **Vision Analysis:** I think this rice is **{detected_age_visual}**")
        st.caption(f"Reason: {reason}")
    
    # 2. Stats Input
    st.write("---")
    st.caption("Adjust detected stats:")
    p_prot = st.slider("Detected Protein", 0, 100, 10, key="p_p")
    p_hard = st.slider("Detected Hardness", 0, 100, 50, key="p_h")
    p_moist = st.slider("Detected Moisture", 0, 100, 20, key="p_m")

    if st.button("Analyze Full Results"):
        with st.spinner("Analyzing patterns..."):
            features = [p_prot, p_hard, p_moist]
            result = train_and_predict_all(features)
            
            if result == "Not enough data":
                st.error("⚠️ Not enough training data! Please go to the 'Train' tab and add at least 2 samples.")
            else:
                res_use, res_age, res_sugg = result
                
                st.balloons()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Use", res_use)
                # We show the Visual Age if available, otherwise the Statistical Age
                if p_img:
                    c2.metric("Visual Age", detected_age_visual)
                else:
                    c2.metric("Statistical Age", res_age)
                c3.metric("Suggestion", "See below")
                
                st.info(f"💡 **AI Suggestion:** {res_sugg}")