from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import os

# --- CONFIGURATION ---
# PASTE YOUR FOLDER ID HERE
FOLDER_ID = "1h0UDCfnGCgdTD3AF7nLjvsBJpDvidqWr" 
KEY_FILE = "drive-key.json"

def test_connection():
    # 1. Check Key File
    if not os.path.exists(KEY_FILE):
        print("❌ Error: drive-key.json not found!")
        return

    # 2. Authenticate
    print("Populating credentials...")
    gauth = GoogleAuth()
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        KEY_FILE, ["https://www.googleapis.com/auth/drive"]
    )
    drive = GoogleDrive(gauth)
    
    # 3. Print Robot Email
    print(f"🤖 Robot Email: {gauth.credentials.service_account_email}")
    print("⚠️ PLEASE CHECK: Did you share the folder with THIS exact email?")

    # 4. Try to Upload
    print(f"🚀 Attempting upload to Folder ID: {FOLDER_ID}")
    
    try:
        file1 = drive.CreateFile({
            'title': 'TEST_CONNECTION.txt',
            'parents': [{'id': FOLDER_ID}]  # This forces it to go to YOUR folder
        })
        file1.SetContentString('If you see this, the connection works!')
        file1.Upload()
        print("✅ SUCCESS! The file was uploaded.")
        print(f"📄 View it here: {file1['alternateLink']}")
    except Exception as e:
        print("\n❌ UPLOAD FAILED")
        print("Error details:", e)
        print("\nPossible causes:")
        print("1. The Folder ID is wrong.")
        print("2. You shared the folder with the wrong email.")
        print("3. You set the permission to 'Viewer' instead of 'Editor'.")

if __name__ == "__main__":
    test_connection()