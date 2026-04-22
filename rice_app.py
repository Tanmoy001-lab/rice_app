import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import requests
import tensorflow as tf
import cv2
from google.auth.transport import requests as google_requests

from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model

import json
from datetime import datetime

# from pydrive2.auth import GoogleAuth  # Removed in favor of Service Account
# from pydrive2.drive import GoogleDrive # Removed in favor of Service Account

import gspread
import gspread
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# # --------------------------------------------------
# # CONFIG
# # --------------------------------------------------

# st.set_page_config(page_title="Rice Quality AI Pro", layout="centered")

# SHEET_ID        = "10lXOiNCfJDnz5bvTtTydEcLfX4FSirfxvitum4udmNs"
# TARGET_FOLDER_ID = "1h0UDCfnGCgdTD3AF7nLjvsBJpDvidqWr"

# AGE_LABELS = ["0-3 months", "3-6 months", "6-12 months", "1-2 years", "2+ years"]
# USE_LABELS = ["Biryani", "Daily Rice", "Fried Rice"]

# AGE_TIPS = {
#     "0-3 months":  "🍚 Very fresh rice — best for soft dishes like porridge or sticky rice.",
#     "3-6 months":  "🍚 Balanced rice — suitable for everyday cooking.",
#     "6-12 months": "🍚 Ideal for biryani — aroma and texture are well developed.",
#     "1-2 years":   "🍚 Well-aged rice — rich flavour, great for premium dishes.",
#     "2+ years":    "🍚 Highly aged rice — strong texture, use carefully.",
# }
# --------------------------------------------------
# CONFIG (TEMPORARY PAPAYA SETUP)
# --------------------------------------------------

st.set_page_config(page_title="Papaya Quality AI Test", layout="centered")

SHEET_ID        = "10lXOiNCfJDnz5bvTtTydEcLfX4FSirfxvitum4udmNs"
TARGET_FOLDER_ID = "1h0UDCfnGCgdTD3AF7nLjvsBJpDvidqWr"

# Temporary Papaya Labels
AGE_LABELS = ["Day 0", "Day 5", "Day 10", "Day 15", "Day 20", "Day 25", "Day 30"]
USE_LABELS = ["Raw", "Semi-Ripe", "Ripe", "Overripe"] # Kept to maintain the 2-output NN architecture

AGE_TIPS = {
    "Day 0":  "🥭 Day 0: Completely raw and hard.",
    "Day 5":  "🥭 Day 5: Starting to show slight color changes.",
    "Day 10": "🥭 Day 10: Semi-ripe, getting softer.",
    "Day 15": "🥭 Day 15: Almost ripe.",
    "Day 20": "🥭 Day 20: Perfectly ripe and ready to eat.",
    "Day 25": "🥭 Day 25: Getting overripe and very soft.",
    "Day 30": "🥭 Day 30: Highly overripe or spoiling.",
}
# --------------------------------------------------
# LOGIN CONFIG
# --------------------------------------------------

# from auth_config import ADMIN_EMAILS
ADMIN_EMAILS = [
    "admin@gmail.com",
    "mithusahoo943@gmail.com",
]

CLIENT_SECRETS_FILE = "client_secrets.json"
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# --------------------------------------------------
# SESSION STATE DEFAULTS
# --------------------------------------------------

for key, default in {
    "logged_in":        False,
    "role":             None,
    "selected_role":    None,
    "train_img_bytes":  None,
    "pred_img_bytes":   None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = "rice_model.keras"

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH, compile=False)
    return None
model = load_my_model()

# --------------------------------------------------
# GOOGLE SHEET
# --------------------------------------------------

def get_sheet():
    creds = get_gcp_creds()
    if not creds:
        st.error("Missing GCP Credentials")
        return None
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID).sheet1


def get_data():
    try:
        return pd.DataFrame(get_sheet().get_all_records())
    except Exception as e:
        st.error(f"Sheet error: {e}")
        return pd.DataFrame()


def add_data(row):
    get_sheet().append_row([
        row["date"],      row["image"],    row["age"],
        row["use"],       row["protein"],  row["hardness"],
        row["moisture"],  row["suggestion"],
    ])

# --------------------------------------------------
# GOOGLE DRIVE
# --------------------------------------------------

# --------------------------------------------------
# GOOGLE AUTH & DRIVE
# --------------------------------------------------

def get_gcp_creds():
    """Returns Credentials from st.secrets (user token) or Service Account."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    
    # ------------------------------------------------------------------
    # 1. PREFERRED: User Token from Secrets (Bypasses Storage Quota)
    # ------------------------------------------------------------------
    # The user pastes their local token into secrets to upload as themselves.
    if "drive_token" in st.secrets:
        try:
            token_info = st.secrets["drive_token"]
            # Use the scopes from the token itself (likely just 'drive')
            # 'drive' scope is sufficient for Sheets too.
            # Explictly requesting 'spreadsheets' when the token doesn't have it causes "invalid_scope".
            return UserCredentials.from_authorized_user_info(token_info, scopes=None)
        except Exception as e:
            st.error(f"Failed to load User Token from secrets: {e}")

    # ------------------------------------------------------------------
    # 2. FALLBACK: Service Account (Has 0GB storage on personal Gmail)
    # ------------------------------------------------------------------
    if os.path.exists("drive-key.json"):
        return ServiceAccountCredentials.from_service_account_file("drive-key.json", scopes=scopes)
    
    if "gcp_service_account" in st.secrets:
        return ServiceAccountCredentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    
    return None

def get_drive_service():
    creds = get_gcp_creds()
    if not creds:
        st.error("Missing GCP Credentials. Please set [drive_token] or [gcp_service_account] in Secrets.")
        return None
    return build("drive", "v3", credentials=creds)


class _FakeFile:
    """Wraps raw bytes so upload_to_drive() can call .getbuffer()."""
    def __init__(self, data):
        self._data = data
    def getbuffer(self):
        return self._data


import tempfile

def upload_to_drive(uploaded_file):
    drive_service = get_drive_service()
    if not drive_service:
        return None

    # Use a temporary file in the system temp directory so it doesn't clutter the project folder
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    # Explicitly close the file handle so Windows can access/delete it later
    tmp.close()

    file_id = None
    try:
        file_metadata = {
            'name': f"{uuid.uuid4()}.jpg",
            'parents': [TARGET_FOLDER_ID]
        }
        
        # Use simple upload (media_body)
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(tmp_path, mimetype='image/jpeg')
        
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = file.get('id')
        
        # Make it public (reader) - Optional, depends if you want the link to be accessible
        # drive_service.permissions().create(
        #     fileId=file_id,
        #     body={'role': 'reader', 'type': 'anyone'}
        # ).execute()

    except Exception as e:
        st.error(f"Drive Upload Error: {e}")
        return None
        
    finally:
        # Ensure the temp file is deleted
        if os.path.exists(tmp_path):
            # Retry deletion a few times for Windows file lock issues
            for _ in range(5):
                try:
                    os.remove(tmp_path)
                    break
                except Exception:
                    time.sleep(0.5)

    if file_id:
        return f"https://drive.google.com/uc?id={file_id}"
    return None

# --------------------------------------------------
# TRAIN MODEL  (MobileNetV2 transfer learning, multi-output)
# --------------------------------------------------

def train_model_from_sheet():
    global model

    df = get_data()
    if df.empty:
        st.error("No training data found.")
        return

    age_map = {k: i for i, k in enumerate(AGE_LABELS)}
    use_map = {k: i for i, k in enumerate(USE_LABELS)}

    images, age_labels, use_labels = [], [], []
    progress = st.progress(0, text="Loading images…")

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            if row.get("age") not in age_map or row.get("use") not in use_map:
                continue
            r   = requests.get(row["image"], timeout=10)
            img = Image.open(BytesIO(r.content)).convert("RGB").resize((128, 128))
            images.append(np.array(img) / 255.0)
            age_labels.append(age_map[row["age"]])
            use_labels.append(use_map[row["use"]])
        except:
            continue
        progress.progress((i + 1) / len(df), text=f"Loaded {i+1}/{len(df)} images")

    progress.empty()

    if len(images) < 15:
        st.error(f"Need at least 15 valid images — only {len(images)} loaded.")
        return

    X     = np.array(images)
    y_age = tf.keras.utils.to_categorical(age_labels, len(AGE_LABELS))
    y_use = tf.keras.utils.to_categorical(use_labels, len(USE_LABELS))

    # MobileNetV2 backbone
    inp     = tf.keras.layers.Input(shape=(128, 128, 3))
    
    # Use input_tensor to ensure correct graph connectivity for Grad-CAM
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inp,
        include_top=False, 
        weights="imagenet"
        
    )
    base_model.trainable = False

    x = base_model.output
    # x = base_model(inp, training=False) # Removed to avoid nesting
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dense(128, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    age_out = tf.keras.layers.Dense(len(AGE_LABELS), activation="softmax", name="age")(x)
    use_out = tf.keras.layers.Dense(len(USE_LABELS),  activation="softmax", name="use")(x)

    model_local = tf.keras.Model(inp, [age_out, use_out])
    model_local.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "accuracy"])

    st.info("Training started — please wait…")
    history = model_local.fit(X, [y_age, y_use], epochs=8, validation_split=0.1)
    model_local.save(MODEL_PATH)

    st.cache_resource.clear()
    model = load_my_model()

    age_acc = history.history.get("age_accuracy", [0])[-1]
    use_acc = history.history.get("use_accuracy", [0])[-1]

    st.success("✅ Model trained and saved!")
    col1, col2 = st.columns(2)
    col1.metric("Age accuracy", f"{round(age_acc * 100, 1)}%")
    col2.metric("Use accuracy", f"{round(use_acc * 100, 1)}%")
    st.rerun()

# --------------------------------------------------
# AUTO-RETRAIN CHECK
# --------------------------------------------------

def auto_retrain_check():
    df = get_data()
    if len(df) > 0 and len(df) % 20 == 0:
        st.info("🔄 Auto-retrain triggered (every 20 records)…")
        train_model_from_sheet()

# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def generate_gradcam(img_bytes, model_ref):

    try:
        # ---------- IMAGE ----------
        image = Image.open(BytesIO(img_bytes)).convert("RGB").resize((128,128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype("float32")

        # ---------- LAST CONV LAYER ----------
        last_conv_layer = model_ref.get_layer("Conv_1")

        # ---------- GRAD MODEL ----------
        grad_model = tf.keras.models.Model(
            inputs=model_ref.inputs,
            outputs=[last_conv_layer.output, model_ref.output[0]]
        )

        with tf.GradientTape() as tape:
            # Pass input as list and use training=False
            conv_outputs, predictions = grad_model([img_array], training=False)

            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap,0) / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # ---------- OVERLAY ----------
        heatmap = cv2.resize(heatmap,(128,128))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = np.array(image)
        superimposed = cv2.addWeighted(original,0.6,heatmap,0.4,0)

        return superimposed

    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        # st.error(str(e)) # Uncomment to see full error
        return None


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

def predict_image(img_bytes):
    if model is None:
        st.error("⚠️ No trained model found. Ask the admin to train the model first.")
        return None, None

    image        = Image.open(BytesIO(img_bytes)).convert("RGB").resize((128, 128))
    arr          = np.array(image) / 255.0
    arr          = arr.reshape(1, 128, 128, 3)

    preds = model.predict(arr, verbose=0)

    if isinstance(preds, list) and len(preds) == 2:
        pred_age, pred_use = preds
        age  = AGE_LABELS[np.argmax(pred_age)]
        use  = USE_LABELS[np.argmax(pred_use)]
        conf = float(np.max(pred_age)) * 100
        st.metric("Confidence", f"{round(conf, 2)}%")
        return age, use
    else:
        st.error(f"⚠️ Model architecture mismatch. Expected 2 outputs (age, use), but got: {type(preds)}")
        if not isinstance(preds, list):
             st.write(f"Output shape: {preds.shape}")
        st.warning("Please ask the Admin to retrain the model.")
        return None, None

# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

def logout():
    for key in ["logged_in", "role", "selected_role", "train_img_bytes", "pred_img_bytes", "user_email"]:
        st.session_state[key] = None if key in ("role", "selected_role", "user_email") else False if key == "logged_in" else None
    
    if os.path.exists("app_token.json"):
        os.remove("app_token.json")
        
    st.rerun()

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

# Login page removed in favor of Sidebar Login

# --------------------------------------------------
# ADMIN PANEL
# --------------------------------------------------

def admin_panel():
    hd1, hd2 = st.columns([5, 1])
    with hd1:
        st.title("👨‍💼 Admin Panel")
    with hd2:
        st.write("")
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    status = "✅ Model exists" if os.path.exists(MODEL_PATH) else "⚠️ No model yet"
    st.caption(f"Model status: {status}")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📤 Add Training Data", "📊 View Database", "🧠 Train Model", "🔮 Prediction"])

    # ── TAB 1: ADD DATA ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("Upload & Label Training Data")

        uploaded = st.file_uploader("Upload Rice Image", type=["jpg","png","jpeg"], key="admin_uploader")
        if uploaded:
            st.session_state.train_img_bytes = uploaded.getvalue()

        if st.session_state.train_img_bytes:
            st.image(st.session_state.train_img_bytes, caption="Preview", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            d_age  = st.selectbox("Rice Age",    AGE_LABELS)
            d_use  = st.selectbox("Best Use",    USE_LABELS)
            d_prot = st.slider("Protein %",      0, 100, 10, key="tp")
        with c2:
            d_hard  = st.slider("Hardness",      0, 100, 50, key="th")
            d_moist = st.slider("Moisture %",    0, 100, 20, key="tm")
            d_sugg  = st.text_input("Suggestion","Good quality")

        if st.button("💾 Save to Database", type="primary"):
            if st.session_state.train_img_bytes:
                with st.spinner("Uploading to Google Drive…"):
                    url = upload_to_drive(_FakeFile(st.session_state.train_img_bytes))
                add_data({
                    "date":       str(pd.Timestamp.now()),
                    "image":      url,
                    "age":        d_age,
                    "use":        d_use,
                    "protein":    d_prot,
                    "hardness":   d_hard,
                    "moisture":   d_moist,
                    "suggestion": d_sugg,
                })
                st.success("✅ Saved to Google Sheet!")
                st.session_state.train_img_bytes = None
                auto_retrain_check()
            else:
                st.warning("⚠️ Please upload an image first.")

    # ── TAB 2: VIEW DATABASE ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Training Database")

        if st.button("🔄 Refresh"):
            pass  # triggers rerun naturally

        df = get_data()
        if df.empty:
            st.info("No records in the database yet.")
        else:
            st.success(f"Total records: {len(df)}")
            st.dataframe(df, use_container_width=True)

            if "age" in df.columns:
                st.markdown("#### Age Distribution")
                st.bar_chart(df["age"].value_counts())

            if "use" in df.columns:
                st.markdown("#### Best Use Distribution")
                st.bar_chart(df["use"].value_counts())

    # ── TAB 3: TRAIN MODEL ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Train AI Model")

        df_check     = get_data()
        record_count = len(df_check) if not df_check.empty else 0

        st.info(f"Database has **{record_count}** records. (Minimum 15 required)")

        if record_count < 15:
            st.warning("Add more training images before training.")
        else:
            st.success(f"Ready to train with {record_count} records.")

        if st.button("🚀 Start Training", type="primary", disabled=(record_count < 15)):
            train_model_from_sheet()

    # ── TAB 4: PREDICTION ─────────────────────────────────────────────────────
    with tab4:
        st.subheader("🔮 Run Prediction")
        render_prediction_ui(key_prefix="admin_pred")

# --------------------------------------------------
# SHARED PREDICTION UI
# --------------------------------------------------

def render_prediction_ui(key_prefix="user"):
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ No trained model available yet. Please contact the admin.")
        return

    st.markdown("Upload a rice image to get an AI-powered quality prediction.")
    st.markdown("---")

    uploaded = st.file_uploader("📸 Upload Rice Image", type=["jpg","png","jpeg"], key=f"{key_prefix}_uploader")
    if uploaded:
        st.session_state.pred_img_bytes = uploaded.getvalue()

    if st.session_state.pred_img_bytes:
        st.image(st.session_state.pred_img_bytes, caption="Uploaded Image", use_container_width=True)

        st.markdown("---")

        if st.button("🔮 Predict Rice Quality", type="primary", key=f"{key_prefix}_predict"):
            with st.spinner("Analysing image…"):
                age, use = predict_image(st.session_state.pred_img_bytes)

            if age:
                st.markdown("### 🌾 Prediction Results")
                r1, r2 = st.columns(2)
                r1.metric("Estimated Age", age)
                r2.metric("Best Use",      use)
                st.info(AGE_TIPS.get(age, ""))

                st.markdown("---")
                st.subheader("🔥 Grad-CAM — Where the model focused")
                with st.spinner("Generating heatmap…"):
                    heat = generate_gradcam(st.session_state.pred_img_bytes, model)

                if heat is not None:
                    g1, g2 = st.columns(2)
                    g1.image(st.session_state.pred_img_bytes, caption="Original",          use_container_width=True)
                    g2.image(heat,                            caption="Grad-CAM Heatmap",   use_container_width=True)
                else:
                    st.info("Grad-CAM visualisation not available for this model.")
    else:
        st.info("👆 Upload a rice image above to begin.")

# --------------------------------------------------
# USER PANEL
# --------------------------------------------------

# User panel removed

# --------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------

# --------------------------------------------------
# FIREBASE & ADMIN SETUP
# --------------------------------------------------

def load_firebase_config():
    try:
        with open(".firebase_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to Streamlit secrets when deployed to the Cloud
        if "firebase_config" in st.secrets:
            # Streamlit secrets are dict-like, just return it directly
            return st.secrets["firebase_config"]
        return {"allowed_admins": []}

firebase_config = load_firebase_config()

def log_user_to_firestore(user_email):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        db = firestore.client()
        db.collection("users").add({
            "email": user_email,
            "timestamp": datetime.now()
        })
    except Exception as e:
        print(f"Firestore logging failed: {e}")

# --------------------------------------------------
# SIDEBAR LOGIN (Admin Access)
# --------------------------------------------------
import streamlit.components.v1 as components
import os

try:
    _firebase_login = components.declare_component("firebase_login", path=os.path.join(os.path.dirname(__file__), "firebase_login"))
except:
    _firebase_login = None

def render_sidebar_login():
    with st.sidebar:
        st.header("🔐 Admin Access")
        
        if st.session_state.get("is_admin") or st.session_state.get("logged_in"):
            st.success("Logged in as Admin")
            if st.button("Logout"):
                logout()
            return

        if "user" not in st.session_state:
            st.markdown("Please sign in securely with Google to train the model:")
            if _firebase_login:
                user_email = _firebase_login(firebase_config=firebase_config, key="google_login")
                
                if user_email:
                    if user_email in firebase_config.get("allowed_admins", []):
                        st.session_state["is_admin"] = True
                        st.session_state.logged_in = True
                        st.session_state.role = "admin"
                        st.session_state["user"] = user_email
                        st.success("Access Granted")
                        log_user_to_firestore(user_email)
                        st.rerun()
                    else:
                        st.error("Access Denied: You are not authorized.")
            else:
                st.error("Custom Firebase component failed to load locally.")

# --------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------

# Always show the title
# st.title("🌾 Rice Quality AI Pro") # Title is already in panels, avoid Double Title

# Render Sidebar for Admin Login
render_sidebar_login()

# Main Content Logic
if st.session_state.get("is_admin") or (st.session_state.get("logged_in") and st.session_state.get("role") == "admin"):
    admin_panel()
else:
    # Public View (Prediction Only)
    st.title("🌾 Rice Quality AI Pro")
    st.info("👋 Welcome! This tool predicts rice quality using AI.")
    st.warning("Only admin can train model")
    render_prediction_ui(key_prefix="public")
