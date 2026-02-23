# Dark Pattern Detection System - Usage Instructions

This guide provides step-by-step instructions on how to install, configure, run, and test the Context-Aware Dark Pattern Detection system.

## 1. Prerequisites & Installation

### System Requirements
- Python 3.8+
- Internet connection (for downloading AI models)
- Chrome Browser (for web scraping)

### Installation Steps
1.  **Clone or Download** the repository to your local machine.
2.  **navigate** to the project directory:
    ```bash
    cd "c:/Dark Patterns"
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note**: This will install PyTorch, Transformers, OpenCV, Selenium, and other required libraries.

## 2. Configuration

The system is fully configurable via `config.py`. You can adjust:
-   **AI Model**: Change the zero-shot classifier model (default: `facebook/bart-large-mnli`).
-   **Keywords**: Add or remove keywords for simple matching.
-   **Thresholds**: Adjust confidence scores for flagging dark patterns.
-   **DOM Settings**: Customize which HTML elements to analyze.

To modify settings, simply edit `config.py` in your text editor.

## 3. Running Demos

We provide a `demo.py` script to showcase the detection capabilities without needing a browser.

**Command:**
```bash
python demo.py
```

**What it does:**
-   Runs text analysis on sample phrases.
-   Runs visual analysis on synthetic images.
-   Demonstrates combined detection.

## 4. Benchmarking Performance

To measure the speed and efficiency of the AI model, run the benchmark script.

**Command:**
```bash
python benchmark_performance.py
```

**What it measures:**
-   **Latency**: How long it takes to analyze a single text element.
-   **Throughput**: How many elements can be processed per second.
-   **Memory Usage**: RAM consumption of the loaded model.

## 5. Testing the System

We have a comprehensive test suite to verify different aspects of the system.

### A. Context-Aware Detection (The "Smart" AI)
Verifies that the system understands nuance (e.g., distinguishing "Hurry up" from "No need to hurry").

**Command:**
```bash
python test_context_aware.py
```

### B. Regression Testing (Core Logic)
Ensures that the basic detection logic (keyword matching, visual analysis) is working correctly.

**Command:**
```bash
python test_static_detection.py
```

### C. Configuration Integration
Verifies that your changes in `config.py` are actually being picked up by the system.

**Command:**
```bash
python test_config_integration.py
```

## 6. Running the Full Pipeline

To analyze a real website, use the `detection_pipeline.py` script.

**Usage (Python):**
```python
from detection_pipeline import DarkPatternDetectionPipeline

# Initialize pipeline (headless=True runs Chrome in background)
pipeline = DarkPatternDetectionPipeline(headless=True)

# Analyze a URL
results = pipeline.analyze_webpage("https://example.com")

# Print summary
print(results['summary'])

# Close pipeline
pipeline.close()
```

> **Note**: This requires Google Chrome to be installed. `webdriver-manager` (installed via requirements.txt) will automatically handle the ChromeDriver setup.

## Troubleshooting

-   **Model Download Issues**: The first time you run `demo.py` or `test_context_aware.py`, it will download a large AI model (~1.6GB). Ensure you have a stable internet connection.
-   **Memory Errors**: If you run out of memory, try switching to a smaller model in `config.py` (e.g., `valhalla/distilbart-mnli-12-3`).
-   **Browser Issues**: If `detection_pipeline.py` fails, ensure your Chrome browser is up to date.

## 7. Showcase Demo (For Judges)

This section provides a scripted walkthrough to demonstrate the core value of the project effectively in a short time.

### **Preparation (Before Presentation)**
1.  **Warm Up the Model**: The AI model takes a few seconds to load the first time. Run `python demo.py` once before your presentation so the model is cached in memory (if running back-to-back) or just to ensure it's downloaded.
2.  **Clear Output**: Type `cls` (Windows) or `clear` (Mac/Linux) in your terminal so it looks clean.

### **Step 1: The "Naive" Approach (Visual Detection)**
*   **Goal**: Show that you can detect basic visual tricks.
*   **Action**: Run `demo.py` specifically for visual elements.
    ```bash
    python demo.py
    ```
*   **Narrative**: "First, our system analyzes the visual hierarchy. Here you can see it detects a 'Low Contrast' element designed to be hidden, and a 'Bright Red' button designed to force user action."

### **Step 2: The "Context-Aware" AI (The "Wow" Factor)**
*   **Goal**: Show that your system is smarter than simple keyword matching.
*   **Action**: Run the context-aware test script.
    ```bash
    python test_context_aware.py
    ```
*   **Narrative**: 
    > "Most detectors just look for words like 'Urgency'. Ours understands **context**."
    > "Watch this:
    > 1. Input: *'You have 300 seconds before this opportunity vanishes.'* -> **Detected Urgency** (even without keywords).
    > 2. Input: *'Take your time, detection is running.'* -> **Ignored** (Correctly understood as non-urgent)."

### **Step 3: Real-Time Performance (Feasibility)**
*   **Goal**: Prove it's fast enough to actually work in a browser.
*   **Action**: Run the benchmark.
    ```bash
    python benchmark_performance.py
    ```

## 8. Running as Chrome Extension

For real-time analysis in your browser, follow these steps:

### A. Start the Backend Server
The extension needs a local server to run the AI model.
1.  Open a terminal.
2.  Run:
    ```bash
    python server/app.py
    ```
3.  Wait until you see "Static Detection Module Ready!".

### B. Install the Extension
1.  Open Chrome and go to `chrome://extensions`.
2.  Enable **Developer mode** (top right toggle).
3.  Click **Load unpacked**.
4.  Select the `extension` folder inside `c:/Dark Patterns`.

### C. Use It
1.  Go to any website (e.g., an e-commerce site).
2.  Click the **Dark Pattern Detector** icon in your toolbar.
3.  Click **Analyze Page**.
4.  Wait a few seconds. Detected patterns will be highlighted with **Red Borders**.

