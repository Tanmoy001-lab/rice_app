import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Rice AI Ultimate", layout="centered")

# --- DATABASE CONNECTION ---
# Connects to your Google Sheet using the secrets you set up
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    """Reads the latest data from Google Sheets"""
    try:
        df = conn.read(worksheet="Sheet1", ttl=0)
        # Handle cases where data might be empty or missing columns
        if df.empty:
            return pd.DataFrame(columns=['date', 'age', 'use', 'protein', 'hardness', 'moisture', 'suggestion'])
        return df
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return pd.DataFrame()

def add_data(row_dict):
    """Adds a new row to the database"""
    df = get_data()
    new_df = pd.DataFrame([row_dict])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    conn.update(worksheet="Sheet1", data=updated_df)

# --- THE AI ENGINE ---
def train_and_predict_all(input_features):
    """
    Trains multiple models to predict ALL topics at once.
    """
    df = get_data()
    
    # 1. Check if we have enough data
    if len(df) < 3:
        return "⚠️ Not enough data! Please add at least 3 rows in the 'Train' tab."
    
    # 2. Prepare the Inputs (X) and Targets (y)
    # Ensure column names match your Google Sheet exactly
    X = df[['protein', 'hardness', 'moisture']].values
    
    # Target 1: Best Use
    y_use = df['use'].astype(str).values
    
    # Target 2: Suggestion
    y_sugg = df['suggestion'].astype(str).values
    
    # Target 3: Age
    y_age = df['age'].astype(str).values

    # 3. Create and Train the Models (The "Brains")
    # We use Try/Except blocks to prevent the app from crashing if one column is empty
    try:
        model_use = RandomForestClassifier(n_estimators=10)
        model_use.fit(X, y_use)
        pred_use = model_use.predict([input_features])[0]
    except:
        pred_use = "Unknown (Not enough info)"

    try:
        model_sugg = RandomForestClassifier(n_estimators=10)
        model_sugg.fit(X, y_sugg)
        pred_sugg = model_sugg.predict([input_features])[0]
    except:
        pred_sugg = "Unknown"

    try:
        model_age = RandomForestClassifier(n_estimators=10)
        model_age.fit(X, y_age)
        pred_age = model_age.predict([input_features])[0]
    except:
        pred_age = "Unknown"

    # Return all results as a dictionary
    return {
        "use": pred_use,
        "suggestion": pred_sugg,
        "age": pred_age
    }

# --- WEBSITE INTERFACE ---
st.title("🌾 Rice AI: Complete Analysis")
st.markdown("Predicts **Best Use**, **Age**, and **Expert Suggestions** based on grain quality.")

tab1, tab2 = st.tabs(["📝 Train (Add Data)", "🔮 Predict (All Topics)"])

# --- TAB 1: ADD DATA ---
with tab1:
    st.subheader("Add Knowledge to Database")
    
    c1, c2 = st.columns(2)
    with c1:
        d_age = st.selectbox("Rice Age", ["New Crop", "6 Months", "1 Year", "Old (>2 Years)"])
        d_use = st.selectbox("Best Use", ["Biryani", "Daily Rice", "Idli/Dosa", "Porridge", "Fried Rice"])
    with c2:
        d_pro = st.slider("Protein %", 1.0, 15.0, 7.0)
        d_hard = st.slider("Hardness", 1, 10, 5)
        d_moist = st.slider("Moisture %", 1, 20, 12)
        
    d_sugg = st.text_input("Suggestion (e.g., 'Cook with extra water')", "Standard cooking required")
    
    if st.button("Save Entry to Cloud"):
        new_row = {
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "age": d_age,
            "use": d_use,
            "protein": d_pro,
            "hardness": d_hard,
            "moisture": d_moist,
            "suggestion": d_sugg
        }
        with st.spinner("Saving to Google Sheets..."):
            add_data(new_row)
            st.success("✅ Saved! The AI has learned from this.")

    with st.expander("View Current Database"):
        st.dataframe(get_data())

# --- TAB 2: PREDICT ---
with tab2:
    st.subheader("Predict All Topics")
    
    st.write("Set the detected parameters:")
    
    # Input Sliders
    p1 = st.slider("Detected Protein", 1.0, 15.0, 7.5, key="p1")
    p2 = st.slider("Detected Hardness", 1, 10, 6, key="p2")
    p3 = st.slider("Detected Moisture", 1, 20, 11, key="p3")
    
    st.caption("Tip: You can change these inputs to see how the prediction changes.")
    
    if st.button("Analyze Results"):
        with st.spinner("Running multiple AI models..."):
            
            # Run the prediction function
            results = train_and_predict_all([p1, p2, p3])
            
            # Check if it returned an error string or the result dictionary
            if isinstance(results, str):
                st.error(results)
            else:
                # Display Results nicely
                st.balloons()
                
                # Create 3 nice boxes for the output
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Recommended Use", results['use'])
                
                with col_b:
                    st.metric("Estimated Age", results['age'])
                
                with col_c:
                    st.info(f"💡 **Suggestion:**\n\n{results['suggestion']}")