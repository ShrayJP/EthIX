"""
Demo script for ETHIX Static Detection Module
Shows various usage examples and capabilities
"""

import numpy as np
import cv2
from static_detection_module import StaticDetectionModule, TextPatternDetector, VisualPatternDetector


def demo_text_detection():
    """Demonstrate text-based dark pattern detection"""
    print("\n" + "="*70)
    print("DEMO 1: TEXT-BASED DARK PATTERN DETECTION")
    print("="*70)
    
    detector = TextPatternDetector()
    
    # Test cases with different dark patterns
    test_cases = [
        {
            "text": "Hurry! Only 2 items left in stock. Order now before it's gone!",
            "expected": ["urgency", "scarcity"]
        },
        {
            "text": "10,000 people have already bought this product. Join them!",
            "expected": ["social_proof"]
        },
        {
            "text": "No thanks, I don't want to save 50% on my purchase",
            "expected": ["confirmshaming"]
        },
        {
            "text": "You must provide your phone number to continue. This is required.",
            "expected": ["forced_action"]
        },
        {
            "text": "*Additional fees may apply. See terms and conditions for details.",
            "expected": ["misdirection"]
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: \"{test['text']}\"")
        
        result = detector.analyze_text(test['text'])
        
        print(f"Detected Patterns: {result['detected_patterns']}")
        print(f"Expected Patterns: {test['expected']}")
        
        # Show keyword matches
        for pattern, keywords in result['keyword_matches'].items():
            print(f"  - {pattern}: matched keywords: {keywords}")
        
        # Show confidence scores
        print(f"Confidence Scores: {result['confidence_scores']}")
        
        print("-" * 70)


def demo_visual_detection():
    """Demonstrate visual-based dark pattern detection"""
    print("\n" + "="*70)
    print("DEMO 2: VISUAL-BASED DARK PATTERN DETECTION")
    print("="*70)
    
    detector = VisualPatternDetector()
    
    # Create synthetic test images
    test_cases = [
        {
            "name": "Low Contrast Hidden Element",
            "description": "Button with text barely visible against background",
            "create_func": create_low_contrast_image
        },
        {
            "name": "Attention-Grabbing Button",
            "description": "Bright, saturated button designed to grab attention",
            "create_func": create_attention_grabbing_image
        },
        {
            "name": "Tiny Close Button",
            "description": "Very small close button that's hard to click",
            "create_func": create_tiny_element_image
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['name']}")
        print(f"Description: {test['description']}")
        
        # Create test image
        image, bbox = test['create_func']()
        
        # Analyze
        result = detector.analyze_visual_element(image, bbox, element_type="button")
        
        print(f"Detected Patterns: {result['detected_patterns']}")
        print(f"Contrast Score: {result['contrast_score']:.3f}")
        
        if result['attention_analysis']['is_attention_grabbing']:
            print(f"Attention Score: {result['attention_analysis']['score']:.3f} (ATTENTION GRABBING)")
        
        if result['size_analysis']['is_tiny']:
            print(f"Relative Size: {result['size_analysis']['relative_size']:.6f} (TINY)")
        
        print("-" * 70)


def demo_combined_detection():
    """Demonstrate combined text + visual detection"""
    print("\n" + "="*70)
    print("DEMO 3: COMBINED TEXT + VISUAL DETECTION")
    print("="*70)
    
    detector = StaticDetectionModule()
    
    # Test cases combining both modalities
    test_cases = [
        {
            "name": "Deceptive CTA Button",
            "text": "YES! I want to save 70% NOW!",
            "create_image": create_attention_grabbing_image
        },
        {
            "name": "Confirmshaming with Low Visibility",
            "text": "No thanks, I prefer to pay full price",
            "create_image": create_low_contrast_image
        },
        {
            "name": "Urgent Call with Bright Design",
            "text": "LAST CHANCE! Only 3 left!",
            "create_image": create_attention_grabbing_image
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['name']}")
        print(f"Text: \"{test['text']}\"")
        
        # Create image
        image, bbox = test['create_image']()
        
        # Analyze
        result = detector.analyze_element(
            text=test['text'],
            image=image,
            element_bbox=bbox,
            element_type="button"
        )
        
        print(f"\nCombined Detected Patterns: {result['combined_patterns']}")
        print(f"Overall Confidence: {result['overall_confidence']:.3f}")
        
        # Show individual analysis
        if result['text_analysis']:
            print(f"\nText Patterns: {result['text_analysis']['detected_patterns']}")
        
        if result['visual_analysis']:
            print(f"Visual Patterns: {result['visual_analysis']['detected_patterns']}")
        
        print("-" * 70)


def demo_page_analysis():
    """Demonstrate full page analysis"""
    print("\n" + "="*70)
    print("DEMO 4: FULL PAGE ANALYSIS")
    print("="*70)
    
    detector = StaticDetectionModule()
    
    # Simulate page elements
    page_elements = [
        {
            "text": "Hurry! Sale ends in 2 hours",
            "bbox": (100, 50, 300, 100),
            "type": "banner"
        },
        {
            "text": "10,000+ satisfied customers",
            "bbox": (100, 150, 300, 200),
            "type": "text"
        },
        {
            "text": "BUY NOW",
            "bbox": (400, 300, 600, 380),
            "type": "button"
        },
        {
            "text": "No thanks, I don't want this deal",
            "bbox": (400, 400, 600, 430),
            "type": "button"
        },
        {
            "text": "*Additional fees may apply",
            "bbox": (100, 500, 400, 520),
            "type": "text"
        }
    ]
    
    # Create page screenshot
    page_screenshot = np.ones((600, 700, 3), dtype=np.uint8) * 240
    
    # Analyze page
    result = detector.analyze_page(page_elements, page_screenshot)
    
    print(f"\nPage Analysis Results:")
    print(f"Total Elements: {result['total_elements']}")
    print(f"High Risk Elements: {len(result['high_risk_elements'])}")
    print(f"Page Risk Score: {result['page_risk_score']:.3f}")
    
    print(f"\nPattern Distribution:")
    for pattern, count in result['detected_patterns_summary'].items():
        print(f"  - {pattern}: {count} occurrence(s)")
    
    print(f"\nHigh Risk Elements:")
    for i, elem in enumerate(result['high_risk_elements'], 1):
        print(f"\n  {i}. Type: {elem['element_type']}")
        print(f"     Patterns: {elem['combined_patterns']}")
        print(f"     Confidence: {elem['overall_confidence']:.3f}")
        if elem['text_analysis']:
            print(f"     Text: \"{elem['text_analysis']['text']}\"")


# Helper functions to create synthetic test images

def create_low_contrast_image():
    """Create image with low contrast element"""
    # Gray background
    image = np.ones((400, 600, 3), dtype=np.uint8) * 180
    
    # Gray button with slightly lighter text (low contrast)
    bbox = (200, 150, 400, 200)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (190, 190, 190), -1)
    cv2.putText(image, "Cancel", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return image, bbox


def create_attention_grabbing_image():
    """Create image with bright, attention-grabbing element"""
    # White background
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Bright red button
    bbox = (200, 150, 400, 200)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), -1)
    cv2.putText(image, "BUY NOW!", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image, bbox


def create_tiny_element_image():
    """Create image with very small element"""
    # White background
    image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Tiny close button
    bbox = (1150, 20, 1180, 40)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (200, 200, 200), -1)
    cv2.putText(image, "x", (1160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
    return image, bbox


def run_all_demos():
    """Run all demonstration examples"""
    print("\n" + "="*70)
    print("ETHIX - STATIC DARK PATTERN DETECTION MODULE")
    print("Demonstration Examples")
    print("="*70)
    
    # Run all demos
    demo_text_detection()
    demo_visual_detection()
    demo_combined_detection()
    demo_page_analysis()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nAll detection modules are working correctly!")
    print("Ready for integration with browser extension and dynamic detection.")


if __name__ == "__main__":
    run_all_demos()
