import zipfile
import os

def extract_zip():
    try:
        # Create the target directory if it doesn't exist
        if not os.path.exists('archive (3)'):
            os.makedirs('archive (3)')
        
        # Extract the zip file
        with zipfile.ZipFile('music.zip', 'r') as zip_ref:
            zip_ref.extractall('archive (3)')
        print("Successfully extracted music.zip to 'archive (3)' directory")
    except Exception as e:
        print(f"Error extracting zip file: {str(e)}")

if __name__ == "__main__":
    extract_zip() 