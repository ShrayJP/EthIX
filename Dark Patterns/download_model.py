
print("Initializing Dark Pattern Detection System...")
print("This will download the AI models if they are not already cached.")
print("This may take 5-10 minutes depending on your internet connection.")

from static_detection_module import StaticDetectionModule

# This line triggers the download
detector = StaticDetectionModule()

print("\nSUCCESS: Models downloaded successfully!")
print("You can now run 'python server/app.py' without waiting.")
