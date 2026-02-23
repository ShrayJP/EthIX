
import unittest
import logging
from static_detection_module import TextPatternDetector

# Configure logging to see classifier output
logging.basicConfig(level=logging.INFO)

class TestContextAwareDetection(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("Loading Zero-Shot Classifier (this may take a moment)...")
        cls.detector = TextPatternDetector()
        
        if not cls.detector.classifier:
            print("WARNING: Zero-shot classifier failed to load. Tests will rely on keyword fallback.")
    
    def test_urgency_context(self):
        """Test distinction between real urgency and non-urgency"""
        
        # True Positive
        text_urgent = "Hurry up! This offer expires in 10 minutes!"
        result_urgent = self.detector.analyze_text(text_urgent)
        print(f"\nText: {text_urgent}\nDetected: {result_urgent['detected_patterns']}")
        
        self.assertIn("urgency", result_urgent["detected_patterns"])
        s_score = result_urgent["confidence_scores"].get("urgency", 0)
        self.assertGreater(s_score, 0.4)

        # True Negative (Context Flip)
        text_not_urgent = "Take your time, there is no need to hurry."
        result_not_urgent = self.detector.analyze_text(text_not_urgent)
        print(f"\nText: {text_not_urgent}\nDetected: {result_not_urgent['detected_patterns']}")
        
        # Keyword matcher would flag 'hurry', but semantics should ideally ignore or give low score
        # Note: Zero-shot might still find some urgency, but score should be lower or different labels
        # Ideally we want this NOT to be flagged, or flagged with very low confidence
        
        # Let's check if 'urgency' is NOT present or score is low
        if "urgency" in result_not_urgent["detected_patterns"]:
            score = result_not_urgent["confidence_scores"]["urgency"]
            print(f"Urgency score for negative case: {score}")
            # We hope the score is lower than the positive case
            self.assertLess(score, s_score)
            
    def test_semantic_detection_without_keywords(self):
        """Test detection where keywords are missing but meaning is present"""
        
        # "This deal expires in 5 minutes" -> "expires" is a keyword?
        # Let's try something without explicit keywords from the list
        # detailed keywords list: hurry, now, today only, expires, deadline, limited time...
        
        text_semantic = "You have 300 seconds before this opportunity vanishes."
        # 'vanishes' is not in standard list. '300 seconds' is not. 'opportunity' is not.
        
        result = self.detector.analyze_text(text_semantic)
        print(f"\nText: {text_semantic}\nDetected: {result['detected_patterns']}")
        
        # Should detect urgency or scarcity
        detected = set(result["detected_patterns"])
        self.assertTrue("urgency" in detected or "scarcity" in detected)

if __name__ == '__main__':
    unittest.main()
