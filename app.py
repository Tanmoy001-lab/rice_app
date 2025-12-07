import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Rice AI Pro", layout="centered")

# --- 1. CONNECT TO GOOGLE SHEET ---
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    try:
        # Read the sheet. If it fails, return an empty table.
        return conn.read(worksheet="Sheet1", ttl=0)
    except:
        return pd.DataFrame()

def add_data(row):
    df = get_data()
    new_df = pd.DataFrame([row])
    # Combine old data with new data
    updated_df = pd.concat([df, new_df], ignore_index=True)
    # Save back to Google Sheets
    conn.update(worksheet="Sheet1", data=updated_df)

# --- 2. THE APP INTERFACE ---
st.title("🌾 Rice Quality AI")
st.write("Upload an image or take a photo to analyze rice quality.")

# Create the tabs
tab1, tab2 = st.tabs(["📸 Train (Add Data)", "🤖 Predict"])

# --- TAB 1: TRAIN ---
with tab1:
    st.header("Add to Knowledge Base")
    
    # 1. IMAGE INPUT (The missing part!)
    img_option = st.radio("Choose Image Source:", ["Camera", "Upload File"])
    
    image_data = None
    if img_option == "Camera":
        image_data = st.camera_input("Take a photo of the rice")
    else:
        image_data = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    # 2. DATA INPUTS
    col1, col2 = st.columns(2)
    with col1:
        use = st.selectbox("Best Use", ["Biryani", "Daily Rice", "Porridge", "Fried Rice"])
        age = st.selectbox("Age", ["New (<6mo)", "Old (>1yr)"])
    with col2:
        prot = st.slider("Protein (%)", 1.0, 15.0, 7.0)
        hard = st.slider("Hardness (1-10)", 1, 10, 5)
        moist = st.slider("Moisture (%)", 1, 20, 12)
        
    sugg = st.text_input("Expert Suggestion", "Good for cooking")

    # 3. SAVE BUTTON
    if st.button("Save to Database"):
        if image_data is None:
            st.error("⚠️ Please take a photo or upload an image first!")
        else:
            # Create the data row
            new_row = {
                "date": str(pd.Timestamp.now()),
                "age": age,
                "use": use,
                "protein": prot,
                "hardness": hard,
                "moisture": moist,
                "suggestion": sugg
            }
            # Save to Google Sheets
            add_data(new_row)
            st.success("✅ Saved! The AI has learned from this sample.")
            st.info("(Note: We saved the data. The actual image is not stored in Sheets to save space.)")

# --- TAB 2: PREDICT ---
with tab2:
    st.header("Predict Quality")
    
    # Image Input for Prediction
    pred_img = st.camera_input("Take a photo to analyze")
    
    if pred_img:
        st.write("Analysing image...")
        # (Here we simulate the AI extracting features from the image)
        
        st.subheader("Detected Values:")
        # We let the user adjust these if the 'image detection' isn't perfect
        p_prot = st.slider("Detected Protein", 1.0, 15.0, 7.0, key="p_pred")
        p_hard = st.slider("Detected Hardness", 1, 10, 5, key="h_pred")
        p_moist = st.slider("Detected Moisture", 1, 20, 12, key="m_pred")
        
        if st.button("Run Prediction"):
            df = get_data()
            if len(df) < 3:
                st.warning("⚠️ database is empty! Please go to 'Train' and add 3-5 samples first.")
            else:
                # TRAIN THE MODEL INSTANTLY
                X = df[['protein', 'hardness', 'moisture']]
                y = df['use']
                
                model = RandomForestClassifier()
                model.fit(X, y)
                
                # PREDICT
                prediction = model.predict([[p_prot, p_hard, p_moist]])[0]
                
                st.balloons()
                st.success(f"🍚 Best Use: **{prediction}**")