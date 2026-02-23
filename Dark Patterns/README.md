# ETHIX - Static Dark Pattern Detection Module

An AI-powered system for detecting manipulative UI design patterns (dark patterns) in webpages using NLP and computer vision.

## Overview

The Static Detection Module is a core component of the ETHIX project that analyzes webpages to identify deceptive user interface practices. It combines:

- **Text Analysis**: NLP-based detection of manipulative language patterns
- **Visual Analysis**: Computer vision techniques to identify deceptive visual elements
- **DOM Analysis**: Automated extraction of UI elements from webpages

## Architecture

### Key Components

1. **DOMAnalyzer** (`dom_analyzer.py`)
   - Extracts UI elements from webpages using Selenium
   - Captures screenshots and element bounding boxes
   - Retrieves text content and CSS properties

2. **TextPatternDetector** (`static_detection_module.py`)
   - Uses transformer models (DistilBERT) for text analysis
   - Keyword-based pattern matching
   - Detects 10 types of textual dark patterns

3. **VisualPatternDetector** (`static_detection_module.py`)
   - Lightweight CNN-based visual analysis (replacing expensive Faster R-CNN)
   - Analyzes color contrast, attention-grabbing elements, element sizes
   - Detects 6 types of visual dark patterns

4. **DarkPatternDetectionPipeline** (`detection_pipeline.py`)
   - Integrates all components
   - Provides end-to-end webpage analysis
   - Generates risk scores and summaries

## Dark Pattern Types Detected

### Textual Patterns
- **Urgency**: "Only 2 left!", "Expires in 5 minutes"
- **Scarcity**: "Limited stock", "Almost gone"
- **Social Proof**: "10,000 people bought this"
- **Misdirection**: Hidden costs, confusing language
- **Forced Action**: "You must...", "Required"
- **Confirmshaming**: "No thanks, I don't want to save money"
- **Disguised Ads**: Misleading sponsored content
- **Trick Questions**: Pre-selected checkboxes with confusing text
- **Sneak Into Basket**: Items added without consent
- **Hidden Costs**: Fees revealed late in process

### Visual Patterns
- **Hidden Elements**: Low contrast text, tiny fonts
- **Attention Grabbing**: Bright colors on "accept" buttons
- **Visual Hierarchy**: Important options made less visible
- **False Urgency**: Countdown timers, progress bars
- **Disguised Close**: Hard-to-find close buttons
- **Fake Notifications**: Misleading notification badges

## File Structure

Here is a quick guide to the key files in this project:

-   **`config.py`**: The central control. Modify this to change AI models, keywords, and thresholds.
-   **`static_detection_module.py`**: The core "Brain". Contains the `TextPatternDetector` (AI model) and `VisualPatternDetector` (Computer Vision).
-   **`dom_analyzer.py`**: The "Eyes". Uses Selenium to browse websites and extract text/images.
-   **`detection_pipeline.py`**: The "Supervisor". Connects the "Brain" and "Eyes" to analyze a full URL.
-   **`demo.py`**: A quick demo script to show off capabilities without a browser.
-   **`benchmark_performance.py`**: Measures how fast the AI runs (latency/throughput).
-   **`test_context_aware.py`**: Verifies the "Smart" AI detection (e.g., understanding nuance).
-   **`test_static_detection.py`**: Verifies the basic logic works correctly.

## Installation

### Prerequisites
- Python 3.8+
- Chrome browser (for Selenium)
- CUDA-capable GPU (optional, for faster processing)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ethix-static-detection

# Install dependencies
pip install -r requirements.txt

# Download Chrome WebDriver (automatic with webdriver-manager)
```

## Usage

### Basic Usage

```python
from detection_pipeline import DarkPatternDetectionPipeline

# Initialize pipeline
pipeline = DarkPatternDetectionPipeline(headless=True)

try:
    # Analyze a webpage
    results = pipeline.analyze_webpage("https://example.com")
    
    # Access summary
    summary = results['summary']
    print(f"Risk Level: {summary['overall_risk']}")
    print(f"Risk Score: {summary['risk_score']}")
    print(f"Patterns Detected: {summary['unique_patterns_detected']}")
    
finally:
    pipeline.close()
```

### Analyzing Individual Elements

```python
from static_detection_module import StaticDetectionModule
import numpy as np

detector = StaticDetectionModule()

# Analyze text only
text_result = detector.analyze_element(
    text="Hurry! Only 2 left in stock!",
    element_type="button"
)

# Analyze with visual data
image = np.array(...)  # Your screenshot
result = detector.analyze_element(
    text="Click here now!",
    image=image,
    element_bbox=(100, 100, 200, 150),
    element_type="button"
)

print(f"Detected patterns: {result['combined_patterns']}")
print(f"Confidence: {result['overall_confidence']}")
```

### Extracting Elements from a Page

```python
from dom_analyzer import DOMAnalyzer

analyzer = DOMAnalyzer(headless=True)

try:
    # Load and analyze page
    page_data = analyzer.analyze_page("https://example.com")
    
    # Access elements
    for element in page_data['elements']:
        print(f"Type: {element['type']}")
        print(f"Text: {element['text']}")
        print(f"BBox: {element['bbox']}")
        
finally:
    analyzer.close_driver()
```

## Output Format

The pipeline produces comprehensive JSON results:

```json
{
  "url": "https://example.com",
  "title": "Example Page",
  "summary": {
    "overall_risk": "MEDIUM",
    "risk_score": 0.52,
    "total_elements_analyzed": 87,
    "high_risk_elements_count": 12,
    "unique_patterns_detected": ["urgency", "scarcity", "confirmshaming"],
    "pattern_counts": {
      "urgency": 5,
      "scarcity": 3,
      "confirmshaming": 2
    },
    "top_concerns": [
      {
        "element_type": "button",
        "patterns": ["urgency", "scarcity"],
        "confidence": 0.85,
        "text_sample": "Only 2 left! Order now..."
      }
    ]
  }
}
```

## Performance Considerations

### Architectural Improvements Over Faster R-CNN

This implementation uses lightweight alternatives to Faster R-CNN:

1. **Color/Contrast Analysis**: Fast OpenCV operations instead of heavy CNN inference
2. **Element-Level Processing**: Analyzes extracted DOM elements rather than full-page object detection
3. **Efficient Text Analysis**: Uses DistilBERT (66M parameters) instead of larger BERT models

### Performance Metrics

- **Page Analysis Time**: ~3-5 seconds per page (on CPU)
- **Memory Usage**: ~500MB-1GB (depending on page complexity)
- **GPU Acceleration**: Optional, provides 2-3x speedup for text embeddings

## Extending the Module

### Adding New Dark Pattern Types

To add a new pattern type, update the keyword dictionaries:

```python
# In TextPatternDetector.__init__()
self.pattern_keywords = {
    # ... existing patterns ...
    "new_pattern": ["keyword1", "keyword2", "phrase"],
}
```

### Custom Visual Analysis

Extend the `VisualPatternDetector` class:

```python
def detect_custom_pattern(self, image, element_bbox):
    """Your custom visual analysis logic"""
    # Implement detection
    return {
        "is_detected": True,
        "confidence": 0.8
    }
```

## Integration with Other Modules

This Static Detection Module is designed to work with:

- **Dynamic Detection Module**: For behavioral pattern detection
- **Explainability Module**: For generating user-friendly explanations
- **Browser Extension**: For real-time detection

## Future Enhancements

- [ ] Fine-tuned BERT model on dark pattern dataset
- [ ] Machine learning classifier for visual patterns
- [ ] Multi-language support
- [ ] Real-time streaming detection
- [ ] Pattern severity scoring
- [ ] A/B test detection

## Troubleshooting

### Common Issues

1. **Selenium WebDriver Issues**
   ```bash
   # Update Chrome and reinstall webdriver
   pip install --upgrade webdriver-manager
   ```

2. **Memory Issues with Large Pages**
   ```python
   # Reduce element extraction
   analyzer.target_elements = ["button", "a", "input"]  # Fewer tags
   ```

3. **Slow Processing**
   ```python
   # Use GPU if available
   # Enable in StaticDetectionModule.__init__()
   self.device = torch.device("cuda")  # Force GPU
   ```

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guide
- Add tests for new features
- Update documentation

## License

[Your License Here]

## Citation

If you use this module in your research, please cite:

```bibtex
@project{ethix2025,
  title={ETHIX - An Ethical Interface Experience},
  author={Abhighna K P, Devanarayan Nambiar, Shray Jibesh Panicker, Sri Vishnu Suresh},
  year={2025},
  institution={Model Engineering College}
}
```

## Contact

For questions or support:
- Email: [your-email]
- GitHub Issues: [repository-url/issues]

## Acknowledgments

- Model Engineering College, Department of Computer Engineering
- Guide: Ms. Kiran Mary Matthew
- CSD415 Project Phase I (October 2025)
