import streamlit as st
import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Drive Connection Doctor", layout="centered")

st.title("🚑 Drive Connection Doctor")

# --- CONFIGURATION ---
TARGET_FOLDER_ID = "1h0UDCfnGCgdTD3AF7nLjvsBJpDvidqWr" 
KEY_FILE = "drive-key.json"

# 1. Check Key File
if not os.path.exists(KEY_FILE):
    st.error("❌ CRITICAL: `drive-key.json` was not found in this folder.")
    st.stop()

# 2. Authenticate
try:
    gauth = GoogleAuth()
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        KEY_FILE, ["https://www.googleapis.com/auth/drive"]
    )
    drive = GoogleDrive(gauth)
    
    # GET THE EMAIL
    robot_email = gauth.credentials.service_account_email
    
    st.success("✅ Authentication Successful")
    st.info(f"🤖 **I AM LOGGED IN AS:** `{robot_email}`")
    st.warning(f"📂 **I AM LOOKING FOR FOLDER ID:** `{TARGET_FOLDER_ID}`")

except Exception as e:
    st.error(f"Authentication Crashed: {e}")
    st.stop()

# 3. Test Connection
if st.button("🔴 TEST CONNECTION NOW"):
    try:
        # Try to see the folder
        folder = drive.CreateFile({'id': TARGET_FOLDER_ID})
        folder.FetchMetadata()
        
        st.balloons()
        st.success(f"✅ **SUCCESS!** I found the folder named: **'{folder['title']}'**")
        st.write("The connection works. You can now use the main app.")

    except Exception as e:
        st.error("❌ **ACCESS DENIED**")
        st.write(f"The Robot (`{robot_email}`) **CANNOT SEE** the folder (`{TARGET_FOLDER_ID}`).")
        
        st.markdown("### 🛠 HOW TO FIX IT:")
        st.markdown(f"1. Copy this email: `{robot_email}`")
        st.markdown(f"2. Go to your Google Drive folder.")
        st.markdown("3. Click **Share**.")
        st.markdown("4. Paste the email and set it to **Editor**.")
        st.markdown("5. Click **Send** and try this button again.")
        
        st.code(str(e))