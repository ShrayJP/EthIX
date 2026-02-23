
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import logging

# Add parent directory to path to import static_detection_module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from static_detection_module import StaticDetectionModule

app = Flask(__name__)
# Enable CORS so the extension (from any origin) can call this API
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Detector (Global variable to keep it in memory)
logger.info("Initializing Static Detection Module...")
detector = StaticDetectionModule()
logger.info("Static Detection Module Ready!")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "model": detector.config["model"]["text_model"]})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze text or DOM elements from the extension
    Expected JSON:
    {
        "elements": [
            {"text": "...", "type": "button", "id": "...", "xpath": "..."},
            ...
        ],
        "url": "..."
    }
    """
    try:
        data = request.json
        if not data or 'elements' not in data:
            return jsonify({"error": "No elements provided"}), 400
        
        elements = data['elements']
        logger.info(f"Received {len(elements)} elements for analysis from {data.get('url', 'unknown')}")
        
        results = []
        
        # Analyze each element
        # In a real scenario, we might batch this for performance
        # Analyze each element in parallel for better performance
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_element(el):
            text = el.get('text', '').strip()
            word_count = len(text.split())
            
            # Simple Filter: Ignore text with less than 4 words 
            # (unless it contains high-risk keywords like "hurry", "only")
            # This prevents navigation items like "Flights" (1 word) or "Sign In" (2 words) from being analyzed by the heavy model.
            if word_count < 4:
                low_text = text.lower()
                high_risk = ["hurry", "only", "left", "limited", "now", "offer"]
                if not any(kw in low_text for kw in high_risk):
                    return None
            
            # Use detector
            analysis = detector.text_detector.analyze_text(text, el.get('type', 'unknown'))
            
            if analysis['detected_patterns']:
                return {
                    "id": el.get('id'),
                    "xpath": el.get('xpath'),
                    "patterns": analysis['detected_patterns'],
                    "confidence": analysis['confidence_scores'],
                    "text": text
                }
            return None

        # Use ThreadPoolExecutor to parallelize
        # Note: Python GIL limits true parallelism for CPU-bound tasks if not released by C-extensions (like torch/numpy often do)
        # But this still helps with I/O or if torch releases GIL
        total_elements = len(elements)
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_el = {executor.submit(process_element, el): el for el in elements}
            
            for i, future in enumerate(as_completed(future_to_el)):
                processed_count += 1
                
                # Log progress every 20 elements
                if processed_count % 20 == 0 or processed_count == total_elements:
                    percentage = (processed_count / total_elements) * 100
                    logger.info(f"Progress: {processed_count}/{total_elements} elements analyzed ({percentage:.1f}%)")
                
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as exc:
                    logger.error(f"Element processing generated an exception: {exc}")

        response = {
            "count": len(results),
            "results": results
        }
        
        # Save results to JSON file
        result_dir = "analysis_output"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        import time
        timestamp = int(time.time())
        filename = f"{result_dir}/analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            import json
            json.dump(response, f, indent=2)
            
        logger.info(f"Analysis Complete! Results saved to {filename}. Found {len(results)} potential dark patterns")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on localhost:5000
    app.run(host='0.0.0.0', port=5000)
