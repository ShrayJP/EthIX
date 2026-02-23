"""
Integration script for Static Dark Pattern Detection
Combines DOM extraction and pattern detection
"""

import json
import cv2
from typing import Dict, List
import logging
from dom_analyzer import DOMAnalyzer
from static_detection_module import StaticDetectionModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DarkPatternDetectionPipeline:
    """
    Complete pipeline for detecting dark patterns on webpages
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the detection pipeline
        
        Args:
            headless: Run browser in headless mode
        """
        self.dom_analyzer = DOMAnalyzer(headless=headless)
        self.detector = StaticDetectionModule()
        logger.info("Dark Pattern Detection Pipeline initialized")
    
    def analyze_webpage(self, url: str, save_results: bool = True) -> Dict:
        """
        Analyze a webpage for dark patterns
        
        Args:
            url: URL to analyze
            save_results: Save results to JSON file
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting analysis of: {url}")
        
        # Step 1: Extract page elements
        logger.info("Extracting page elements...")
        page_data = self.dom_analyzer.analyze_page(url)
        
        # Step 2: Run static detection
        logger.info("Running static detection...")
        detection_results = self.detector.analyze_page(
            page_elements=page_data['elements'],
            page_screenshot=page_data['screenshot']
        )
        
        # Step 3: Compile complete results
        complete_results = {
            "url": page_data['url'],
            "title": page_data['title'],
            "page_size": page_data['page_size'],
            "analysis": detection_results,
            "timestamp": None  # Add timestamp if needed
        }
        
        # Step 4: Generate summary
        summary = self.generate_summary(complete_results)
        complete_results['summary'] = summary
        
        # Step 5: Save results
        if save_results:
            self.save_results(complete_results, url)
        
        logger.info(f"Analysis complete. Risk score: {detection_results['page_risk_score']:.2f}")
        
        return complete_results
    
    def generate_summary(self, results: Dict) -> Dict:
        """
        Generate human-readable summary of findings
        
        Args:
            results: Complete analysis results
            
        Returns:
            Summary dictionary
        """
        analysis = results['analysis']
        
        summary = {
            "page_url": results['url'],
            "page_title": results['title'],
            "overall_risk": self.categorize_risk(analysis['page_risk_score']),
            "risk_score": round(analysis['page_risk_score'], 2),
            "total_elements_analyzed": analysis['total_elements'],
            "high_risk_elements_count": len(analysis['high_risk_elements']),
            "unique_patterns_detected": list(analysis['detected_patterns_summary'].keys()),
            "pattern_counts": analysis['detected_patterns_summary'],
            "top_concerns": []
        }
        
        # Identify top concerns
        for element in analysis['high_risk_elements'][:5]:  # Top 5
            concern = {
                "element_type": element['element_type'],
                "patterns": element['combined_patterns'],
                "confidence": round(element['overall_confidence'], 2)
            }
            
            if element['text_analysis']:
                concern['text_sample'] = element['text_analysis']['text'][:100]
            
            summary['top_concerns'].append(concern)
        
        return summary
    
    def categorize_risk(self, risk_score: float) -> str:
        """
        Categorize risk level
        
        Args:
            risk_score: Numerical risk score
            
        Returns:
            Risk category string
        """
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def save_results(self, results: Dict, url: str):
        """
        Save analysis results to file
        
        Args:
            results: Analysis results
            url: URL analyzed
        """
        # Create safe filename from URL
        safe_filename = url.replace("https://", "").replace("http://", "")
        safe_filename = safe_filename.replace("/", "_").replace(".", "_")
        filename = f"analysis_{safe_filename}.json"
        
        # Remove screenshot from JSON (too large)
        results_copy = results.copy()
        if 'screenshot' in results_copy:
            del results_copy['screenshot']
        
        # Remove numpy arrays from embeddings
        for elem in results_copy.get('analysis', {}).get('elements_analyzed', []):
            if elem.get('text_analysis') and 'embeddings' in elem['text_analysis']:
                del elem['text_analysis']['embeddings']
        
        # Save to JSON
        filepath = f"/home/claude/{filename}"
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def save_annotated_screenshot(self, results: Dict, screenshot_path: str):
        """
        Save screenshot with bounding boxes around detected dark patterns
        
        Args:
            results: Analysis results with screenshot
            screenshot_path: Path to save annotated screenshot
        """
        if 'screenshot' not in results:
            logger.warning("No screenshot available")
            return
        
        screenshot = results['screenshot'].copy()
        
        # Draw bounding boxes for high-risk elements
        for element in results['analysis']['high_risk_elements']:
            if element.get('visual_analysis') and 'element_bbox' in element['visual_analysis']:
                bbox = element['visual_analysis']['element_bbox']
                x1, y1, x2, y2 = bbox
                
                # Red box for high-risk elements
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f"{element['element_type']}: {element['overall_confidence']:.2f}"
                cv2.putText(screenshot, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save annotated screenshot
        cv2.imwrite(screenshot_path, screenshot)
        logger.info(f"Annotated screenshot saved to: {screenshot_path}")
    
    def close(self):
        """Clean up resources"""
        self.dom_analyzer.close_driver()


def demo_pipeline():
    """Demo the complete pipeline"""
    pipeline = DarkPatternDetectionPipeline(headless=True)
    
    try:
        # Analyze a webpage
        results = pipeline.analyze_webpage("https://example.com")
        
        # Print summary
        print("\n" + "="*60)
        print("DARK PATTERN ANALYSIS SUMMARY")
        print("="*60)
        
        summary = results['summary']
        print(f"\nPage: {summary['page_title']}")
        print(f"URL: {summary['page_url']}")
        print(f"\nOverall Risk: {summary['overall_risk']}")
        print(f"Risk Score: {summary['risk_score']}/1.0")
        print(f"\nElements Analyzed: {summary['total_elements_analyzed']}")
        print(f"High Risk Elements: {summary['high_risk_elements_count']}")
        
        print(f"\nDetected Patterns:")
        for pattern, count in summary['pattern_counts'].items():
            print(f"  - {pattern}: {count} instance(s)")
        
        if summary['top_concerns']:
            print(f"\nTop Concerns:")
            for i, concern in enumerate(summary['top_concerns'], 1):
                print(f"\n  {i}. {concern['element_type'].upper()}")
                print(f"     Patterns: {', '.join(concern['patterns'])}")
                print(f"     Confidence: {concern['confidence']}")
                if 'text_sample' in concern:
                    print(f"     Text: \"{concern['text_sample']}...\"")
        
        print("\n" + "="*60)
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    demo_pipeline()
