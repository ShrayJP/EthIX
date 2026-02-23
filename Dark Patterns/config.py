"""
Configuration file for ETHIX Static Detection Module
Adjust these parameters to customize detection behavior
"""

# Model Configuration
MODEL_CONFIG = {
    # Text analysis model
    "text_model": "distilbert-base-uncased",  # Options: distilbert, bert-base, roberta-base
    
    # Zero-Shot Classification Model (for context-aware detection)
    "zero_shot_model": "facebook/bart-large-mnli", 
    
    # Device selection
    "device": "auto",  # Options: "auto", "cuda", "cpu"
    
    # Model precision
    "use_fp16": False,  # Use half-precision for faster inference (requires GPU)
}

# Detection Thresholds
DETECTION_THRESHOLDS = {
    # Minimum confidence score to flag an element (keyword based)
    "min_confidence": 0.5,
    
    # Minimum probability for ML model detection
    "min_model_confidence": 0.4,
    
    # Page-level risk score thresholds
    "risk_levels": {
        "low": 0.3,      # Below this is low risk
        "medium": 0.6,   # Between low and this is medium
        "high": 0.8,     # Between medium and this is high
        # Above high is critical
    },
    
    # Visual analysis thresholds
    "visual": {
        "min_contrast": 0.15,           # Below this is "hidden element"
        "attention_threshold": 0.6,      # Above this is "attention grabbing"
        "tiny_element_ratio": 0.001,     # Below this is "tiny element"
    }
}

# Pattern Keywords
# Add or modify keywords for different dark pattern types
PATTERN_KEYWORDS = {
    "urgency": [
        "hurry", "now", "today only", "expires", "deadline", "limited time",
        "act fast", "don't miss", "last chance", "ending soon", "quick",
        "immediate", "instant", "rush", "urgent"
    ],
    
    "scarcity": [
        "only", "left", "limited", "few remaining", "almost gone",
        "selling fast", "low stock", "rare", "exclusive", "while supplies last",
        "running out", "stock limited"
    ],
    
    "social_proof": [
        "people bought", "customers love", "best seller", "trending",
        "most popular", "recommended", "verified purchase", "rated",
        "reviews", "testimonial", "join thousands", "everyone is", "people who bought"
    ],
    
    "misdirection": [
        "hidden", "fee", "additional charge", "terms apply",
        "restrictions", "see details", "fine print", "conditions apply",
        "subject to", "may vary", "disclaimer"
    ],
    
    "forced_action": [
        "must", "required", "mandatory", "you have to", "need to",
        "obligation", "compulsory", "necessary", "essential"
    ],
    
    "confirmshaming": [
        "no thanks", "i don't want", "skip", "maybe later",
        "i'll pass", "no deals", "i prefer", "rather not",
        "don't care about", "not interested in saving"
    ],
    
    "disguised_ads": [
        "sponsored", "promoted", "advertisement", "partner content",
        "featured", "recommended for you"
    ],
    
    "trick_questions": [
        "opt out", "unsubscribe", "newsletter", "updates",
        "marketing emails", "promotional"
    ],
}

# DOM Extraction Settings
DOM_CONFIG = {
    # Elements to extract and analyze
    "target_elements": [
        "button", "a", "input", "select", "textarea",
        "label", "span", "div", "p", "h1", "h2", "h3"
    ],
    
    # Text attributes to check
    "text_attributes": [
        "innerText", "textContent", "placeholder", "value",
        "alt", "title", "aria-label"
    ],
    
    # Page load wait time (seconds)
    "page_load_timeout": 10,
    
    # Screenshot resolution
    "screenshot_width": 1920,
    "screenshot_height": 1080,
}

# Analysis Settings
ANALYSIS_CONFIG = {
    # Maximum elements to analyze per page (for performance)
    "max_elements_per_page": 200,
    
    # Minimum text length to analyze
    "min_text_length": 3,
    
    # Minimum element size (pixels) to analyze
    "min_element_area": 50,
    
    # Enable/disable specific analyses
    "enable_text_analysis": True,
    "enable_visual_analysis": True,
    
    # Logging level
    "log_level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR
}

# Output Settings
OUTPUT_CONFIG = {
    # Save results to file
    "save_results": True,
    
    # Output directory
    "output_dir": "/mnt/user-data/outputs",
    
    # Save annotated screenshots
    "save_screenshots": True,
    
    # Include embeddings in output (increases file size)
    "include_embeddings": False,
    
    # Pretty-print JSON
    "pretty_json": True,
}

# Performance Settings
PERFORMANCE_CONFIG = {
    # Batch size for processing multiple elements
    "batch_size": 16,
    
    # Enable multiprocessing
    "use_multiprocessing": False,
    
    # Number of worker processes
    "num_workers": 4,
    
    # Cache model outputs
    "enable_cache": True,
}

# Browser Settings (for Selenium)
BROWSER_CONFIG = {
    # Run in headless mode
    "headless": True,
    
    # Browser type
    "browser": "chrome",  # Options: chrome, firefox
    
    # Additional Chrome options
    "chrome_options": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
    ],
    
    # User agent (optional, None for default)
    "user_agent": None,
}


def get_config():
    """
    Get complete configuration dictionary
    
    Returns:
        Dictionary with all configuration settings
    """
    return {
        "model": MODEL_CONFIG,
        "thresholds": DETECTION_THRESHOLDS,
        "keywords": PATTERN_KEYWORDS,
        "dom": DOM_CONFIG,
        "analysis": ANALYSIS_CONFIG,
        "output": OUTPUT_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "browser": BROWSER_CONFIG,
    }


def update_config(updates: dict):
    """
    Update configuration settings
    
    Args:
        updates: Dictionary of config updates
    """
    config = get_config()
    
    for key, value in updates.items():
        if key in config:
            if isinstance(value, dict) and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    
    return config


if __name__ == "__main__":
    # Print current configuration
    import json
    config = get_config()
    print(json.dumps(config, indent=2))
