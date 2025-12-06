import streamlit as st
import pandas as pd
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Rice Excel Manager")

# --- HELPER FUNCTIONS ---
def convert_df_to_excel(df):
    # Converts the internal data back to an Excel file for downloading
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- MAIN APP ---
st.title("🌾 Rice AI (Excel Version)")
st.caption("Upload your Excel file -> Add Data -> Download Updated File")

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("📂 Step 1: Upload your 'rice_data.xlsx'", type=["xlsx"])

if uploaded_file:
    # Load the data into memory
    df = pd.read_excel(uploaded_file)
    
    # Create Tabs
    tab1, tab2 = st.tabs(["📝 Add Data", "🤖 Predict"])

    # --- TAB 1: ADD DATA ---
    with tab1:
        st.header("Add New Sample")
        col1, col2 = st.columns(2)
        with col1:
            new_use = st.selectbox("Best Use", ["Biryani", "Daily", "Porridge"])
            new_sugg = st.text_input("Suggestion", "Good quality")
        with col2:
            new_prot = st.slider("Protein", 0.0, 15.0, 7.0)
            new_hard = st.slider("Hardness", 1, 10, 5)
            new_moist = st.slider("Moisture", 1, 20, 12)

        if st.button("Add to List"):
            # Create a new row
            new_row = {
                "Date": pd.Timestamp.now(), 
                "Use": new_use, 
                "Protein": new_prot, 
                "Hardness": new_hard, 
                "Moisture": new_moist, 
                "Suggestion": new_sugg
            }
            # Add to the existing data
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Added! Now download the file below to save it permanently.")

        # Show Data
        st.dataframe(df.tail(5))

        # DOWNLOAD BUTTON (Crucial for saving)
        st.download_button(
            label="💾 Download Updated Excel File",
            data=convert_df_to_excel(df),
            file_name="rice_data_updated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # --- TAB 2: PREDICT ---
    with tab2:
        st.header("Predict Quality")
        
        if len(df) < 5:
            st.warning("⚠️ Please add at least 5 rows of data in your Excel file to train the AI.")
        else:
            # Train Model on the uploaded Excel data
            X = df[['Protein', 'Hardness', 'Moisture']]
            y = df['Use']
            
            clf = RandomForestClassifier()
            clf.fit(X, y)
            
            st.write("Enter detected values:")
            p_prot = st.number_input("Protein", 0.0, 15.0, 7.0)
            p_hard = st.number_input("Hardness", 1, 10, 5)
            p_moist = st.number_input("Moisture", 1, 20, 12)
            
            if st.button("Predict"):
                prediction = clf.predict([[p_prot, p_hard, p_moist]])[0]
                st.balloons()
                st.success(f"Best Use: {prediction}")

else:
    st.info("Please upload an Excel file to start.")