# ETHIX Static Detection Module - Quick Start Guide

This guide will help you get started with the Static Dark Pattern Detection Module in just a few minutes.

## Installation (5 minutes)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for ML models)
- Transformers (for NLP)
- OpenCV (for image processing)
- Selenium (for web scraping)
- Other utilities

### Step 2: Install Chrome WebDriver

The webdriver will be installed automatically when you first run the code, thanks to `webdriver-manager`.

### Step 3: Verify Installation

```bash
python demo.py
```

This should run all demonstration examples and verify that everything is working correctly.

## Basic Usage (3 minutes)

### Example 1: Analyze a Single Text Element

```python
from static_detection_module import StaticDetectionModule

# Initialize detector
detector = StaticDetectionModule()

# Analyze text
result = detector.analyze_element(
    text="Hurry! Only 2 left in stock! Order now!",
    element_type="button"
)

# View results
print(f"Patterns detected: {result['combined_patterns']}")
print(f"Confidence: {result['overall_confidence']:.2f}")
```

Expected output:
```
Patterns detected: ['urgency', 'scarcity']
Confidence: 0.60
```

### Example 2: Analyze a Webpage

```python
from detection_pipeline import DarkPatternDetectionPipeline

# Initialize pipeline
pipeline = DarkPatternDetectionPipeline(headless=True)

try:
    # Analyze any webpage
    results = pipeline.analyze_webpage("https://example.com")
    
    # Print summary
    summary = results['summary']
    print(f"Risk Level: {summary['overall_risk']}")
    print(f"Patterns Found: {summary['unique_patterns_detected']}")
    
finally:
    pipeline.close()
```

### Example 3: Batch Analysis of Multiple Pages

```python
from detection_pipeline import DarkPatternDetectionPipeline

urls = [
    "https://example.com",
    "https://another-site.com",
    # Add more URLs...
]

pipeline = DarkPatternDetectionPipeline(headless=True)

try:
    for url in urls:
        print(f"\nAnalyzing: {url}")
        results = pipeline.analyze_webpage(url)
        
        summary = results['summary']
        print(f"Risk: {summary['overall_risk']} ({summary['risk_score']:.2f})")
        print(f"Patterns: {len(summary['unique_patterns_detected'])}")

finally:
    pipeline.close()
```

## Understanding the Output

### Result Structure

```json
{
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
    "top_concerns": [...]
  }
}
```

### Risk Levels

- **LOW** (< 0.3): Few or no dark patterns detected
- **MEDIUM** (0.3-0.6): Some manipulative elements present
- **HIGH** (0.6-0.8): Multiple concerning patterns
- **CRITICAL** (> 0.8): Extensive use of dark patterns

### Dark Pattern Types

The system detects these categories:

**Text-based:**
1. Urgency ("Hurry!", "Limited time!")
2. Scarcity ("Only 2 left!")
3. Social Proof ("10,000 bought this")
4. Confirmshaming ("No thanks, I don't want savings")
5. Forced Action ("You must provide...")
6. Misdirection (Hidden costs, fine print)

**Visual-based:**
1. Hidden Elements (Low contrast)
2. Attention Grabbing (Bright colors)
3. Tiny Elements (Hard to click)
4. Visual Hierarchy (Important options minimized)

## Customization

### Adjust Detection Sensitivity

Edit `config.py`:

```python
DETECTION_THRESHOLDS = {
    "min_confidence": 0.3,  # Lower = more sensitive
    "visual": {
        "min_contrast": 0.10,  # Lower = stricter
        "attention_threshold": 0.7,  # Higher = less strict
    }
}
```

### Add Custom Dark Pattern Keywords

Edit `config.py`:

```python
PATTERN_KEYWORDS = {
    "urgency": [
        "hurry", "now", "quick",
        # Add your keywords here
        "your_custom_keyword"
    ],
}
```

### Change Output Location

Edit `config.py`:

```python
OUTPUT_CONFIG = {
    "output_dir": "/your/custom/path",
    "save_screenshots": True,
}
```

## Troubleshooting

### Issue: "WebDriver not found"

**Solution:**
```bash
pip install --upgrade webdriver-manager
```

### Issue: "CUDA out of memory"

**Solution:** Force CPU usage in `config.py`:
```python
MODEL_CONFIG = {
    "device": "cpu",
}
```

### Issue: "Slow analysis"

**Solutions:**
1. Use GPU if available (auto-detected)
2. Reduce elements analyzed:
```python
ANALYSIS_CONFIG = {
    "max_elements_per_page": 100,  # Default: 200
}
```

### Issue: "Too many false positives"

**Solution:** Increase confidence threshold:
```python
DETECTION_THRESHOLDS = {
    "min_confidence": 0.7,  # Default: 0.5
}
```

## Next Steps

1. **Run Tests**: `python test_static_detection.py`
2. **Try Demos**: `python demo.py`
3. **Read Full Documentation**: See `README.md`
4. **Integrate with Browser Extension**: See integration guide
5. **Add Custom Patterns**: Edit `config.py`

## Common Use Cases

### Use Case 1: Website Audit

```python
# Audit your own website
pipeline = DarkPatternDetectionPipeline()

try:
    results = pipeline.analyze_webpage("https://your-website.com")
    
    # Generate report
    if results['summary']['overall_risk'] != "LOW":
        print("⚠️ Dark patterns detected! Review these elements:")
        for concern in results['summary']['top_concerns']:
            print(f"  - {concern}")

finally:
    pipeline.close()
```

### Use Case 2: Competitive Analysis

```python
# Compare competitors
competitors = ["site1.com", "site2.com", "site3.com"]

for site in competitors:
    results = pipeline.analyze_webpage(f"https://{site}")
    print(f"{site}: Risk={results['summary']['overall_risk']}")
```

### Use Case 3: Research Dataset Creation

```python
# Collect data for research
import pandas as pd

data = []
for url in your_url_list:
    results = pipeline.analyze_webpage(url)
    data.append({
        'url': url,
        'risk_score': results['summary']['risk_score'],
        'patterns': results['summary']['unique_patterns_detected'],
    })

df = pd.DataFrame(data)
df.to_csv('dark_patterns_dataset.csv')
```

## Tips for Best Results

1. **Run on representative pages** - Test homepages, checkout flows, signup forms
2. **Combine with manual review** - The system is a tool to assist, not replace human judgment
3. **Adjust thresholds** - Different industries may require different sensitivity levels
4. **Regular updates** - Dark pattern techniques evolve, keep your keywords updated
5. **Check multiple times** - Some patterns may only appear under certain conditions

## Support

- GitHub Issues: [repository-url/issues]
- Email: [your-email]
- Documentation: See `README.md`

---

**Ready to start detecting dark patterns!** 🚀

Run `python demo.py` to see the system in action.
