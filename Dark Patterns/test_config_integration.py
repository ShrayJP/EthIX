import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import config

# We need to patch BEFORE importing the modules that use config
# But since they import config at module level (or in __init__), we can patch config.get_config
from static_detection_module import TextPatternDetector, VisualPatternDetector
from dom_analyzer import DOMAnalyzer

class TestConfigIntegration(unittest.TestCase):
    
    def setUp(self):
        self.original_get_config = config.get_config
        
    def tearDown(self):
        config.get_config = self.original_get_config

    def test_text_detector_uses_config(self):
        """Test that TextPatternDetector uses keywords from config"""
        # Create a mock config with a unique keyword
        mock_keywords = {
            "test_category": ["test_keyword_12345"]
        }
        
        # Create a full mock config structure
        full_config = {
            "keywords": mock_keywords,
            "thresholds": {"visual": {}},
            "dom": {}
        }
        
        with patch('config.get_config', return_value=full_config):
            # Initialize detector
            detector = TextPatternDetector()
            
            # Verify patterns were loaded
            self.assertIn("test_category", detector.pattern_keywords)
            self.assertIn("test_keyword_12345", detector.pattern_keywords["test_category"])
            
            # Verify detection works with new keyword
            result = detector.detect_keyword_patterns("This contains test_keyword_12345 inside it")
            self.assertIn("test_category", result)

    def test_visual_detector_uses_config(self):
        """Test that VisualPatternDetector uses thresholds from config"""
        full_config = {
            "thresholds": {
                "visual": {
                    "min_contrast": 0.9,
                    "attention_threshold": 0.6,
                    "tiny_element_ratio": 0.001
                }
            },
            "keywords": {},
            "dom": {}
        }
        
        with patch('config.get_config', return_value=full_config):
            detector = VisualPatternDetector()
            self.assertEqual(detector.visual_thresholds["min_contrast"], 0.9)
            
    def test_dom_analyzer_uses_config(self):
        """Test that DOMAnalyzer uses settings from config"""
        mock_dom_config = {
            "target_elements": ["custom_element_tag"],
            "text_attributes": ["custom_attr"],
            "page_load_timeout": 5,
            "screenshot_width": 800,
            "screenshot_height": 600
        }
        
        full_config = {
            "dom": mock_dom_config,
            "keywords": {},
            "thresholds": {}
        }
        
        with patch('config.get_config', return_value=full_config):
            analyzer = DOMAnalyzer(headless=True)
            
            # Verify settings were loaded
            self.assertEqual(analyzer.target_elements, ["custom_element_tag"])
            self.assertEqual(analyzer.text_attributes, ["custom_attr"])

if __name__ == '__main__':
    unittest.main()
