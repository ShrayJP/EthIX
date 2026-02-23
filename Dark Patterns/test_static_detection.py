"""
Unit tests for ETHIX Static Detection Module
"""

import unittest
import numpy as np
import cv2
from static_detection_module import (
    TextPatternDetector, 
    VisualPatternDetector, 
    StaticDetectionModule
)


class TestTextPatternDetector(unittest.TestCase):
    """Test cases for TextPatternDetector"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = TextPatternDetector()
    
    def test_urgency_detection(self):
        """Test detection of urgency patterns"""
        text = "Hurry! Only 2 hours left to buy!"
        result = self.detector.detect_keyword_patterns(text)
        
        self.assertIn("urgency", result)
        self.assertIn("scarcity", result)
    
    def test_confirmshaming_detection(self):
        """Test detection of confirmshaming"""
        text = "No thanks, I don't want to save money"
        result = self.detector.detect_keyword_patterns(text)
        
        self.assertIn("confirmshaming", result)
    
    def test_social_proof_detection(self):
        """Test detection of social proof"""
        text = "Join 10,000 people who bought this"
        result = self.detector.detect_keyword_patterns(text)
        
        self.assertIn("social_proof", result)
    
    def test_forced_action_detection(self):
        """Test detection of forced action language"""
        text = "You must provide your email to continue"
        result = self.detector.detect_keyword_patterns(text)
        
        self.assertIn("forced_action", result)
    
    def test_no_pattern_in_normal_text(self):
        """Test that normal text doesn't trigger false positives"""
        text = "Welcome to our website. We hope you enjoy your visit."
        result = self.detector.detect_keyword_patterns(text)
        
        self.assertEqual(len(result), 0)
    
    def test_analyze_text_returns_dict(self):
        """Test that analyze_text returns expected structure"""
        text = "Buy now!"
        result = self.detector.analyze_text(text)
        
        self.assertIn("text", result)
        self.assertIn("detected_patterns", result)
        self.assertIn("confidence_scores", result)
        self.assertIsInstance(result["detected_patterns"], list)


class TestVisualPatternDetector(unittest.TestCase):
    """Test cases for VisualPatternDetector"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = VisualPatternDetector()
    
    def create_test_image(self, size=(400, 600, 3)):
        """Helper to create test image"""
        return np.ones(size, dtype=np.uint8) * 200
    
    def test_low_contrast_detection(self):
        """Test detection of low contrast (hidden) elements"""
        image = self.create_test_image()
        
        # Create low contrast element
        bbox = (100, 100, 200, 150)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     (205, 205, 205), -1)
        
        contrast = self.detector.analyze_color_contrast(image, bbox)
        
        # Low contrast should be detected (score < 0.15)
        self.assertLess(contrast, 0.2)
    
    def test_high_contrast_detection(self):
        """Test that high contrast elements are not flagged"""
        image = self.create_test_image()
        
        # Create high contrast element (black on gray)
        bbox = (100, 100, 200, 150)
        # Add some texture/noise so std dev is not 0
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     (0, 0, 0), -1)
        # Add a white line to create contrast variance within the element
        cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 5)
        
        contrast = self.detector.analyze_color_contrast(image, bbox)
        
        # High contrast should be detected
        self.assertGreater(contrast, 0.3)
    
    def test_attention_grabbing_bright_colors(self):
        """Test detection of attention-grabbing bright colors"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Create bright red element
        bbox = (100, 100, 200, 150)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     (0, 0, 255), -1)
        
        result = self.detector.detect_attention_grabbers(image, bbox)
        
        self.assertTrue(result["is_attention_grabbing"])
        self.assertGreater(result["score"], 0.6)
    
    def test_normal_colors_not_attention_grabbing(self):
        """Test that normal colors are not flagged"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Create gray element
        bbox = (100, 100, 200, 150)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     (150, 150, 150), -1)
        
        result = self.detector.detect_attention_grabbers(image, bbox)
        
        self.assertFalse(result["is_attention_grabbing"])
    
    def test_tiny_element_detection(self):
        """Test detection of very small elements"""
        page_size = (1200, 800)  # width, height
        
        # Tiny element (10x10 pixels in 1200x800 page)
        tiny_bbox = (1000, 50, 1010, 60)
        
        result = self.detector.analyze_element_size(tiny_bbox, page_size)
        
        self.assertTrue(result["is_tiny"])
        self.assertLess(result["relative_size"], 0.001)
    
    def test_normal_size_element(self):
        """Test that normal-sized elements are not flagged as tiny"""
        page_size = (1200, 800)
        
        # Normal element (200x100 pixels)
        normal_bbox = (100, 100, 300, 200)
        
        result = self.detector.analyze_element_size(normal_bbox, page_size)
        
        self.assertFalse(result["is_tiny"])


class TestStaticDetectionModule(unittest.TestCase):
    """Test cases for integrated StaticDetectionModule"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = StaticDetectionModule()
    
    def test_text_only_analysis(self):
        """Test analysis with text only"""
        result = self.detector.analyze_element(
            text="Hurry! Limited time offer!",
            element_type="button"
        )
        
        self.assertIsNotNone(result["text_analysis"])
        self.assertIsNone(result["visual_analysis"])
        self.assertGreater(len(result["combined_patterns"]), 0)
    
    def test_visual_only_analysis(self):
        """Test analysis with image only"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        bbox = (100, 100, 200, 150)
        
        result = self.detector.analyze_element(
            image=image,
            element_bbox=bbox,
            element_type="button"
        )
        
        self.assertIsNone(result["text_analysis"])
        self.assertIsNotNone(result["visual_analysis"])
    
    def test_combined_text_visual_analysis(self):
        """Test analysis combining text and visual"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        # Bright red button
        bbox = (100, 100, 200, 150)
        cv2.rectangle(image, bbox[:2], bbox[2:], (0, 0, 255), -1)
        
        result = self.detector.analyze_element(
            text="BUY NOW! LIMITED TIME!",
            image=image,
            element_bbox=bbox,
            element_type="button"
        )
        
        self.assertIsNotNone(result["text_analysis"])
        self.assertIsNotNone(result["visual_analysis"])
        self.assertGreater(result["overall_confidence"], 0)
    
    def test_page_analysis_multiple_elements(self):
        """Test full page analysis with multiple elements"""
        page_elements = [
            {
                "text": "Only 2 left!",
                "bbox": (100, 100, 200, 150),
                "type": "label"
            },
            {
                "text": "Buy now",
                "bbox": (300, 100, 400, 150),
                "type": "button"
            },
            {
                "text": "No thanks, I'll pay more",
                "bbox": (300, 200, 500, 250),
                "type": "button"
            }
        ]
        
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        result = self.detector.analyze_page(page_elements, image)
        
        self.assertEqual(result["total_elements"], 3)
        self.assertGreater(len(result["detected_patterns_summary"]), 0)
        self.assertGreaterEqual(result["page_risk_score"], 0)
        self.assertLessEqual(result["page_risk_score"], 2.0)
    
    def test_page_analysis_risk_score(self):
        """Test that page risk score is calculated correctly"""
        # High-risk elements
        high_risk_elements = [
            {
                "text": "HURRY! LAST CHANCE! ONLY 1 LEFT!",
                "bbox": (100, 100, 300, 150),
                "type": "button"
            }
        ] * 5  # Multiple high-risk elements
        
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        result = self.detector.analyze_page(high_risk_elements, image)
        
        # Should have high risk score
        self.assertGreater(result["page_risk_score"], 0.5)
    
    def test_confidence_scores_in_range(self):
        """Test that confidence scores are between 0 and 1"""
        result = self.detector.analyze_element(
            text="Limited offer! Buy now!",
            element_type="button"
        )
        
        self.assertGreaterEqual(result["overall_confidence"], 0)
        self.assertLessEqual(result["overall_confidence"], 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = StaticDetectionModule()
    
    def test_empty_text(self):
        """Test handling of empty text"""
        result = self.detector.analyze_element(
            text="",
            element_type="button"
        )
        
        # Should not crash, should return valid structure
        self.assertIsInstance(result, dict)
    
    def test_none_text(self):
        """Test handling of None text"""
        result = self.detector.analyze_element(
            text=None,
            element_type="button"
        )
        
        self.assertIsNone(result["text_analysis"])
    
    def test_invalid_bbox(self):
        """Test handling of invalid bounding box"""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200
        
        # Bbox with zero area
        result = self.detector.analyze_element(
            image=image,
            element_bbox=(100, 100, 100, 100),
            element_type="button"
        )
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
    
    def test_very_long_text(self):
        """Test handling of very long text"""
        long_text = "word " * 1000  # Very long text
        
        result = self.detector.analyze_element(
            text=long_text,
            element_type="text"
        )
        
        # Should not crash
        self.assertIsInstance(result, dict)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextPatternDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualPatternDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestStaticDetectionModule))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
