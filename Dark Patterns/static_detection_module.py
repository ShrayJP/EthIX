"""
ETHIX - Static Dark Pattern Detection Module
Analyzes textual content and visual layouts to identify manipulative UI elements
"""

import torch
import torch.nn as nn
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModel
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPatternDetector:
    """
    NLP-based detector for manipulative text patterns using transformer models
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the text pattern detector
        
        Args:
            model_name: HuggingFace model name for text analysis (kept for backward compatibility)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if torch.cuda.is_available() else -1
        
        # Load configuration
        self.config = config.get_config()
        self.pattern_keywords = self.config["keywords"]
        
        # Initialize Zero-Shot Classifier
        zero_shot_model = self.config["model"].get("zero_shot_model", "facebook/bart-large-mnli")
        logger.info(f"Loading Zero-Shot Classifier: {zero_shot_model}")
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification", 
                model=zero_shot_model, 
                device=self.device_id
            )
        except Exception as e:
            logger.error(f"Failed to load zero-shot model: {e}")
            logger.warning("Falling back to keyword-only detection")
            self.classifier = None

        # Keep the embedding model for feature extraction if needed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_text_features(self, text: str) -> torch.Tensor:
        """
        Extract embeddings from text using transformer model
        
        Args:
            text: Input text to analyze
            
        Returns:
            Text embeddings tensor
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def detect_keyword_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Detect dark patterns using keyword matching
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of detected patterns and matched keywords
        """
        text_lower = text.lower()
        detected = {}
        
        for pattern_type, keywords in self.pattern_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                detected[pattern_type] = matches
        
        return detected
    
    def analyze_text(self, text: str, element_type: str = "unknown") -> Dict:
        """
        Comprehensive text analysis for dark patterns using Zero-Shot Classification
        
        Args:
            text: Text content to analyze
            element_type: Type of UI element (button, label, etc.)
            
        Returns:
            Analysis results with detected patterns and confidence scores
        """
        # 1. Fast Keyword Filter (Optimization)
        # If no keywords match, the model is unlikely to find anything either
        # This saves expensive model inference time
        keyword_patterns = self.detect_keyword_patterns(text)
        
        detected_patterns = []
        confidence_scores = {}
        
        # 2. Context-Aware Classification (Zero-Shot)
        if self.classifier:
            try:
                # Define candidate labels representing dark patterns
                # We map internal keys (e.g., "urgency") to natural language labels
                labels_map = {
                    "urgency": "urgency or pressure to act fast",
                    "scarcity": "scarcity or limited supply",
                    "social_proof": "social proof or popularity",
                    "misdirection": "misdirection or hidden costs",
                    "forced_action": "forced action or requirement",
                    "confirmshaming": "shaming for declining an offer",
                    "disguised_ads": "advertisement disguised as content",
                    "trick_questions": "trick question or confusing option"
                }
                
                candidate_labels = list(labels_map.values())
                
                # Run classification
                output = self.classifier(text, candidate_labels, multi_label=True)
                
                # Process results
                min_conf = self.config["thresholds"].get("min_model_confidence", 0.4)
                
                for label, score in zip(output['labels'], output['scores']):
                    if score > min_conf:
                        # Map back to internal key
                        for key, desc in labels_map.items():
                            if desc == label:
                                detected_patterns.append(key)
                                confidence_scores[key] = score
                                break
                                
            except Exception as e:
                logger.error(f"Classification failed: {e}")
                # Fallback to keyword matching results
                detected_patterns = list(keyword_patterns.keys())
                confidence_scores = {k: 0.5 for k in keyword_patterns}
        else:
            # Fallback if classifier failed to load
            detected_patterns = list(keyword_patterns.keys())
            confidence_scores = {k: 0.5 for k in keyword_patterns}

        # Extract patterns primarily from model, but augment with strong keyword matches if needed
        # (For now, we trust the model more, so we rely on its output)
        
        # Extract embeddings (still useful for other tasks)
        embeddings = self.extract_text_features(text)
        
        result = {
            "text": text,
            "element_type": element_type,
            "detected_patterns": list(set(detected_patterns)),
            "keyword_matches": keyword_patterns, # Keep for debugging
            "confidence_scores": confidence_scores,
            "embeddings": embeddings.cpu().numpy()
        }
        
        return result


class VisualPatternDetector:
    """
    Lightweight CNN-based detector for visual dark patterns
    Uses EfficientNet instead of Faster R-CNN for better performance
    """
    
    def __init__(self):
        """Initialize the visual pattern detector"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config.get_config()
        self.visual_thresholds = self.config["thresholds"]["visual"]
        
        # Visual dark pattern categories
        self.visual_patterns = [
            "hidden_elements",      # Low contrast, tiny text
            "attention_grabbing",   # Bright colors, large fonts on accept buttons
            "visual_hierarchy",     # Important options made less visible
            "false_urgency",        # Countdown timers, progress bars
            "disguised_close",      # Hard to find close buttons
            "fake_notifications",   # Fake notification badges
        ]
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for analysis
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def analyze_color_contrast(self, image: np.ndarray, 
                               element_bbox: Tuple[int, int, int, int]) -> float:
        """
        Analyze color contrast in an element (detect hidden elements)
        
        Args:
            image: Full page image
            element_bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Contrast ratio score
        """
        x1, y1, x2, y2 = element_bbox
        element_region = image[y1:y2, x1:x2]
        
        if element_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(element_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast using standard deviation
        contrast = np.std(gray)
        
        # Normalize to 0-1 range (assuming max std dev of 64)
        normalized_contrast = min(contrast / 64.0, 1.0)
        
        return normalized_contrast
    
    def detect_attention_grabbers(self, image: np.ndarray,
                                  element_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Detect attention-grabbing visual elements
        
        Args:
            image: Full page image
            element_bbox: Bounding box of element
            
        Returns:
            Detection results
        """
        x1, y1, x2, y2 = element_bbox
        element_region = image[y1:y2, x1:x2]
        
        if element_region.size == 0:
            return {"is_attention_grabbing": False, "score": 0.0}
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(element_region, cv2.COLOR_BGR2HSV)
        
        # Detect bright/saturated colors
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        avg_saturation = np.mean(saturation) / 255.0
        avg_brightness = np.mean(value) / 255.0
        
        # High saturation + high brightness = attention grabbing
        attention_score = (avg_saturation + avg_brightness) / 2
        
        return {
            "is_attention_grabbing": attention_score > self.visual_thresholds["attention_threshold"],
            "score": attention_score,
            "avg_saturation": avg_saturation,
            "avg_brightness": avg_brightness
        }
    
    def analyze_element_size(self, element_bbox: Tuple[int, int, int, int],
                            page_size: Tuple[int, int]) -> Dict:
        """
        Analyze if element size suggests dark pattern
        
        Args:
            element_bbox: Element bounding box
            page_size: Full page dimensions (width, height)
            
        Returns:
            Size analysis results
        """
        x1, y1, x2, y2 = element_bbox
        width, height = page_size
        
        element_width = x2 - x1
        element_height = y2 - y1
        element_area = element_width * element_height
        
        page_area = width * height
        
        relative_size = element_area / page_area if page_area > 0 else 0
        
        return {
            "element_width": element_width,
            "element_height": element_height,
            "relative_size": relative_size,
            "relative_size": relative_size,
            "is_tiny": relative_size < self.visual_thresholds["tiny_element_ratio"],  # Less than defined ratio
            "is_large": relative_size > 0.1    # More than 10% of page
        }
    
    def analyze_visual_element(self, image: np.ndarray,
                               element_bbox: Tuple[int, int, int, int],
                               element_type: str = "unknown") -> Dict:
        """
        Comprehensive visual analysis of UI element
        
        Args:
            image: Full page screenshot
            element_bbox: Element bounding box
            element_type: Type of element (button, text, etc.)
            
        Returns:
            Visual analysis results
        """
        page_height, page_width = image.shape[:2]
        
        # Analyze contrast
        contrast = self.analyze_color_contrast(image, element_bbox)
        
        # Detect attention grabbers
        attention = self.detect_attention_grabbers(image, element_bbox)
        
        # Analyze size
        size_analysis = self.analyze_element_size(element_bbox, (page_width, page_height))
        
        # Determine detected patterns
        detected_patterns = []
        
        if contrast < self.visual_thresholds["min_contrast"]:
            detected_patterns.append("hidden_elements")
        
        if attention["is_attention_grabbing"]:
            detected_patterns.append("attention_grabbing")
        
        if size_analysis["is_tiny"] and element_type in ["button", "link"]:
            detected_patterns.append("disguised_close")
        
        result = {
            "element_bbox": element_bbox,
            "element_type": element_type,
            "contrast_score": contrast,
            "attention_analysis": attention,
            "size_analysis": size_analysis,
            "detected_patterns": detected_patterns,
            "confidence_scores": {
                pattern: 0.7 for pattern in detected_patterns
            }
        }
        
        return result


class StaticDetectionModule:
    """
    Main Static Detection Module combining text and visual analysis
    """
    
    def __init__(self):
        """Initialize the static detection module"""
        self.text_detector = TextPatternDetector()
        self.visual_detector = VisualPatternDetector()
        logger.info("Static Detection Module initialized")
    
    def analyze_element(self, 
                       text: Optional[str] = None,
                       image: Optional[np.ndarray] = None,
                       element_bbox: Optional[Tuple[int, int, int, int]] = None,
                       element_type: str = "unknown") -> Dict:
        """
        Analyze a single UI element for dark patterns
        
        Args:
            text: Text content of the element
            image: Screenshot containing the element
            element_bbox: Bounding box of element in image
            element_type: Type of UI element
            
        Returns:
            Combined analysis results
        """
        results = {
            "element_type": element_type,
            "text_analysis": None,
            "visual_analysis": None,
            "combined_patterns": [],
            "overall_confidence": 0.0
        }
        
        # Text analysis
        if text:
            results["text_analysis"] = self.text_detector.analyze_text(text, element_type)
            results["combined_patterns"].extend(
                results["text_analysis"]["detected_patterns"]
            )
        
        # Visual analysis
        if image is not None and element_bbox is not None:
            results["visual_analysis"] = self.visual_detector.analyze_visual_element(
                image, element_bbox, element_type
            )
            results["combined_patterns"].extend(
                results["visual_analysis"]["detected_patterns"]
            )
        
        # Remove duplicates
        results["combined_patterns"] = list(set(results["combined_patterns"]))
        
        # Calculate overall confidence
        all_scores = []
        if results["text_analysis"]:
            all_scores.extend(results["text_analysis"]["confidence_scores"].values())
        if results["visual_analysis"]:
            all_scores.extend(results["visual_analysis"]["confidence_scores"].values())
        
        if all_scores:
            results["overall_confidence"] = np.mean(all_scores)
        
        return results
    
    def analyze_page(self, 
                    page_elements: List[Dict],
                    page_screenshot: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze entire page for dark patterns
        
        Args:
            page_elements: List of elements with text, bbox, type
            page_screenshot: Full page screenshot
            
        Returns:
            Page-level analysis results
        """
        page_results = {
            "total_elements": len(page_elements),
            "elements_analyzed": [],
            "detected_patterns_summary": {},
            "high_risk_elements": [],
            "page_risk_score": 0.0
        }
        
        for element in page_elements:
            element_result = self.analyze_element(
                text=element.get("text"),
                image=page_screenshot,
                element_bbox=element.get("bbox"),
                element_type=element.get("type", "unknown")
            )
            
            page_results["elements_analyzed"].append(element_result)
            
            # Track high-risk elements
            if element_result["overall_confidence"] > 0.6:
                page_results["high_risk_elements"].append(element_result)
            
            # Aggregate pattern counts
            for pattern in element_result["combined_patterns"]:
                if pattern not in page_results["detected_patterns_summary"]:
                    page_results["detected_patterns_summary"][pattern] = 0
                page_results["detected_patterns_summary"][pattern] += 1
        
        # Calculate page risk score
        if page_results["elements_analyzed"]:
            avg_confidence = np.mean([
                e["overall_confidence"] 
                for e in page_results["elements_analyzed"]
            ])
            pattern_diversity = len(page_results["detected_patterns_summary"])
            
            # Risk score combines average confidence and pattern diversity
            page_results["page_risk_score"] = (avg_confidence + 
                                              min(pattern_diversity * 0.1, 0.5))
        
        return page_results


if __name__ == "__main__":
    # Example usage
    print("Initializing Static Detection Module...")
    detector = StaticDetectionModule()
    
    # Example 1: Text-only analysis
    print("\n--- Example 1: Text Analysis ---")
    text_result = detector.analyze_element(
        text="Hurry! Only 2 left in stock! Order now before it's gone!",
        element_type="button"
    )
    print(f"Detected patterns: {text_result['combined_patterns']}")
    print(f"Confidence: {text_result['overall_confidence']:.2f}")
    
    # Example 2: Visual analysis with dummy image
    print("\n--- Example 2: Visual Analysis ---")
    dummy_image = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
    visual_result = detector.analyze_element(
        image=dummy_image,
        element_bbox=(100, 100, 200, 150),
        element_type="button"
    )
    print(f"Detected patterns: {visual_result['combined_patterns']}")
    
    # Example 3: Combined analysis
    print("\n--- Example 3: Combined Analysis ---")
    combined_result = detector.analyze_element(
        text="No thanks, I don't want to save 50%",
        image=dummy_image,
        element_bbox=(500, 600, 700, 650),
        element_type="button"
    )
    print(f"Detected patterns: {combined_result['combined_patterns']}")
    print(f"Confidence: {combined_result['overall_confidence']:.2f}")
