import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# --- CONFIGURATION ---
DATA_FILE = 'rice_database.csv'
IMG_FOLDER = 'uploaded_images'

# Ensure folders exist
if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

# Page setup
st.set_page_config(page_title="Rice Quality AI", layout="centered")

# --- HELPER FUNCTIONS ---
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=['filename', 'date', 'age_range', 'best_use', 'protein', 'hardness', 'moisture', 'suggestion'])

def process_image_for_model(img_path):
    """
    Converts an image to a flat array of numbers for the AI to read.
    Resizes to 64x64 to keep it fast.
    """
    try:
        img = Image.open(img_path).convert('L') # Convert to grayscale
        img = img.resize((64, 64))
        return np.array(img).flatten()
    except Exception as e:
        return None

def train_and_predict(target_image):
    """
    1. Loads all saved data.
    2. Trains a Random Forest model on the fly.
    3. Predicts the stats for the target_image.
    """
    df = load_data()
    
    if len(df) < 5:
        return "Not enough data to train! Please add at least 5 samples in the 'Train' tab."

    # Prepare Training Data
    X = [] # Images
    y_protein = [] # Labels
    y_hardness = []
    
    for index, row in df.iterrows():
        img_p = os.path.join(IMG_FOLDER, row['filename'])
        if os.path.exists(img_p):
            features = process_image_for_model(img_p)
            if features is not None:
                X.append(features)
                y_protein.append(row['protein'])
                y_hardness.append(row['hardness'])
    
    if not X:
        return "Error loading training images."

    # Train Models (Simple Random Forest)
    # Note: In a real large app, you wouldn't retrain on every click, 
    # you would save the model file. For this size, this is fine.
    model_protein = RandomForestClassifier(n_estimators=10)
    model_protein.fit(X, y_protein)
    
    model_hardness = RandomForestClassifier(n_estimators=10)
    model_hardness.fit(X, y_hardness)

    # Predict
    target_features = process_image_for_model(target_image).reshape(1, -1)
    pred_protein = model_protein.predict(target_features)[0]
    pred_hardness = model_hardness.predict(target_features)[0]
    
    return {
        "protein": pred_protein,
        "hardness": pred_hardness,
        "suggestion": "Based on hardness, suitable for general cooking." if pred_hardness > 5 else "Soft rice, good for porridge."
    }

# --- MAIN APP INTERFACE ---
st.title("🌾 Rice Grain Analyzer AI")
st.write("Upload or take a photo of rice grains to Train or Predict.")

# Create Tabs
tab1, tab2 = st.tabs(["🏋️ Train Database", "🔮 Predict Quality"])

# --- TAB 1: TRAIN ---
with tab1:
    st.header("Add Data to Knowledge Base")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        d_date = st.date_input("Date", datetime.now())
        d_age = st.selectbox("Age Range", ["New Crop", "6 Months", "1 Year", "Old"])
        d_use = st.selectbox("Best Use", ["Biryani", "Daily Rice", "Idli/Dosa", "Porridge"])
    with col2:
        d_protein = st.slider("Protein Level (%)", 0.0, 15.0, 7.0)
        d_hardness = st.slider("Hardness (1-10)", 1, 10, 5)
        d_moisture = st.slider("Moisture (%)", 0, 20, 12)
    
    d_sugg = st.text_area("Expert Suggestion", "Good for daily consumption.")

    # Image Input (Camera or Upload)
    img_source = st.radio("Image Source", ["Camera", "Upload File"])
    image_file = None
    
    if img_source == "Camera":
        image_file = st.camera_input("Take a picture of the rice")
    else:
        image_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    if st.button("Save to Database"):
        if image_file is not None:
            # Save Image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rice_{timestamp}.jpg"
            filepath = os.path.join(IMG_FOLDER, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_file.getbuffer())
            
            # Save Data
            new_data = {
                'filename': filename,
                'date': d_date,
                'age_range': d_age,
                'best_use': d_use,
                'protein': d_protein,
                'hardness': d_hardness,
                'moisture': d_moisture,
                'suggestion': d_sugg
            }
            
            df = load_data()
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            
            st.success("✅ Data Saved Successfully!")
        else:
            st.error("Please provide an image.")

    # Show current data
    if st.checkbox("Show Database"):
        st.dataframe(load_data())

# --- TAB 2: PREDICT ---
with tab2:
    st.header("Analyze Rice Sample")
    
    pred_source = st.radio("Select Source", ["Camera", "Upload File"], key="pred_radio")
    pred_file = None
    
    if pred_source == "Camera":
        pred_file = st.camera_input("Take a picture for prediction", key="pred_cam")
    else:
        pred_file = st.file_uploader("Upload for prediction", type=['jpg', 'png', 'jpeg'], key="pred_upload")

    if pred_file is not None:
        st.image(pred_file, caption="Analyzing...", width=300)
        
        if st.button("Predict Results"):
            with st.spinner("AI is processing..."):
                # Save temp file for processing
                temp_path = "temp_predict.jpg"
                with open(temp_path, "wb") as f:
                    f.write(pred_file.getbuffer())
                
                # Run Logic
                result = train_and_predict(temp_path)
                
                if isinstance(result, str):
                    st.warning(result)
                else:
                    st.balloons()
                    st.subheader("🎯 Analysis Results")
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted Protein", f"{result['protein']}%")
                    c2.metric("Predicted Hardness", f"{result['hardness']}/10")
                    st.info(f"💡 Suggestion: {result['suggestion']}")