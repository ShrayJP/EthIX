# How to Release Your Dark Pattern Detector to Third Parties

Currently, your setup is in **Developer Mode** (Client + Local Server).
To release this to a user who **does not have the code** and **cannot install Python**, you have two main options:

---

## Option 1: The "Professional" Way (Cloud Hosting)
**Best for:** Real-world users, Chrome Web Store distribution, pure ease of use.
**How it works:** You host the heavy AI brain on a cloud server. The user only installs the lightweight Chrome Extension.

### Steps:
1.  **Deploy the Backend (`server/app.py`)**:
    -   Upload your Python code to a cloud provider like **AWS EC2**, **Google Cloud Run**, or **Render**.
    -   Install dependencies (`pip install -r requirements.txt`).
    -   Run the server so it's accessible via a public URL (e.g., `https://api.mydarkpatterndetector.com`).

2.  **Update the Extension**:
    -   Go to `extension/content.js`.
    -   Change the API URL from `http://localhost:5000/analyze` to your new cloud URL (e.g., `https://api.mydarkpatterndetector.com/analyze`).
    -   Update `extension/manifest.json` to allow permissions for the new domain.

3.  **Distribute the Extension**:
    -   **Zip File**: Zip the `extension` folder and email it. The user loads it via "Load Unpacked".
    -   **Web Store**: Publish it to the Chrome Web Store (~$5 fee). The user installs it with one click.

**Result:** The user installs the extension and it "just works" without installing Python or downloading models.

---

## Option 2: The "Offline/Hackathon" Way (Packaged Executable)
**Best for:** Privacy (data stays local), free hosting, demos where internet is flaky.
**How it works:** You bundle the Python server into a single `.exe` file. The user runs the `.exe` and installs the extension.

### Steps:
1.  **Bundle the Python Server**:
    -   Use a tool like **PyInstaller**:
        ```bash
        pip install pyinstaller
        pyinstaller --onefile --add-data "static_detection_module.py;." server/app.py
        ```
    -   This creates a standalone `dist/app.exe` (Windows) or binary (Mac/Linux).
    -   *Note*: The first run might still need to download the model, or you can bundle the model files (increases size to ~2GB).

2.  **Update the Extension**:
    -   Keep it pointing to `localhost:5000`.

3.  **Distribute**:
    -   Send the user a zip file containing:
        -   `DarkPatternServer.exe`
        -   `extension/` folder (or a `.crx` file)
        -   `README.txt` instructions: "1. Run the .exe. 2. Install the extension."

**Result:** The user double-clicks the `.exe`, installs the extension, and it works locally.

---

## Recommendation for Judges/Demo
If you are presenting to judges:
1.  **Stick to Localhost** for the live demo (it's faster and safer).
2.  **Mention Option 1 (Cloud)** as your "Go-to-Market Strategy". "We plan to host the backend on AWS so users just need the plugin."
