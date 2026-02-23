"""
DOM Analyzer for extracting UI elements from webpages
Extracts text, bounding boxes, and element types for static analysis
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import base64
import io
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DOMAnalyzer:
    """
    Analyzes DOM structure to extract UI elements for dark pattern detection
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the DOM analyzer
        
        Args:
            headless: Run browser in headless mode
        """
        self.headless = headless
        self.driver = None
        
        # Load configuration
        self.config = config.get_config()["dom"]
        
        # Interactive elements to analyze
        self.target_elements = self.config["target_elements"]
        
        # Attributes that might contain dark pattern text
        self.text_attributes = self.config["text_attributes"]
    
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        logger.info("WebDriver initialized")
    
    def close_driver(self):
        """Close WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")
    
    def load_page(self, url: str, wait_time: int = 5):
        """
        Load webpage
        
        Args:
            url: URL to load
            wait_time: Time to wait for page load
        """
        if not self.driver:
            self.setup_driver()
        
        self.driver.get(url)
        
        # Wait for page to load
        WebDriverWait(self.driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        logger.info(f"Page loaded: {url}")
    
    def get_element_bbox(self, element) -> Tuple[int, int, int, int]:
        """
        Get bounding box of element
        
        Args:
            element: Selenium WebElement
            
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        try:
            location = element.location
            size = element.size
            
            x1 = int(location['x'])
            y1 = int(location['y'])
            x2 = x1 + int(size['width'])
            y2 = y1 + int(size['height'])
            
            return (x1, y1, x2, y2)
        except:
            return (0, 0, 0, 0)
    
    def get_element_text(self, element) -> str:
        """
        Extract all text content from element
        
        Args:
            element: Selenium WebElement
            
        Returns:
            Combined text from all relevant attributes
        """
        texts = []
        
        # Get visible text
        try:
            text = element.text.strip()
            if text:
                texts.append(text)
        except:
            pass
        
        # Get text from attributes
        for attr in self.text_attributes:
            try:
                attr_text = element.get_attribute(attr)
                if attr_text and attr_text.strip():
                    texts.append(attr_text.strip())
            except:
                pass
        
        # Remove duplicates and combine
        unique_texts = list(set(texts))
        return " | ".join(unique_texts)
    
    def get_element_styles(self, element) -> Dict:
        """
        Get computed styles for element
        
        Args:
            element: Selenium WebElement
            
        Returns:
            Dictionary of relevant CSS properties
        """
        try:
            return {
                "color": element.value_of_css_property("color"),
                "background-color": element.value_of_css_property("background-color"),
                "font-size": element.value_of_css_property("font-size"),
                "font-weight": element.value_of_css_property("font-weight"),
                "opacity": element.value_of_css_property("opacity"),
                "display": element.value_of_css_property("display"),
                "visibility": element.value_of_css_property("visibility"),
            }
        except:
            return {}
    
    def is_visible(self, element) -> bool:
        """
        Check if element is visible
        
        Args:
            element: Selenium WebElement
            
        Returns:
            True if visible
        """
        try:
            return element.is_displayed() and element.size['width'] > 0 and element.size['height'] > 0
        except:
            return False
    
    def extract_elements(self) -> List[Dict]:
        """
        Extract all relevant UI elements from current page
        
        Returns:
            List of element dictionaries with text, bbox, type, etc.
        """
        elements = []
        
        for tag in self.target_elements:
            web_elements = self.driver.find_elements(By.TAG_NAME, tag)
            
            for element in web_elements:
                # Skip if not visible
                if not self.is_visible(element):
                    continue
                
                # Get element info
                text = self.get_element_text(element)
                bbox = self.get_element_bbox(element)
                styles = self.get_element_styles(element)
                
                # Skip if no text and no meaningful size
                if not text and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 100:
                    continue
                
                element_data = {
                    "type": tag,
                    "text": text,
                    "bbox": bbox,
                    "styles": styles,
                    "html_class": element.get_attribute("class") or "",
                    "html_id": element.get_attribute("id") or "",
                }
                
                elements.append(element_data)
        
        logger.info(f"Extracted {len(elements)} elements")
        return elements
    
    def take_screenshot(self) -> np.ndarray:
        """
        Take screenshot of current page
        
        Returns:
            Screenshot as numpy array (BGR format for OpenCV)
        """
        # Get screenshot as base64
        screenshot_base64 = self.driver.get_screenshot_as_base64()
        
        # Decode and convert to numpy array
        screenshot_bytes = base64.b64decode(screenshot_base64)
        image = Image.open(io.BytesIO(screenshot_bytes))
        
        # Convert to numpy array (RGB)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV compatibility
        image_bgr = image_np[:, :, ::-1].copy()
        
        return image_bgr
    
    def analyze_page(self, url: str) -> Dict:
        """
        Complete page analysis: extract elements and screenshot
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary with elements and screenshot
        """
        self.load_page(url)
        
        # Extract elements
        elements = self.extract_elements()
        
        # Take screenshot
        screenshot = self.take_screenshot()
        
        # Get page info
        page_info = {
            "url": url,
            "title": self.driver.title,
            "elements": elements,
            "screenshot": screenshot,
            "page_size": (screenshot.shape[1], screenshot.shape[0])  # (width, height)
        }
        
        return page_info


def demo_dom_analyzer():
    """Demo the DOM analyzer"""
    analyzer = DOMAnalyzer(headless=True)
    
    try:
        # Analyze a sample page
        print("Analyzing page...")
        page_data = analyzer.analyze_page("https://example.com")
        
        print(f"\nPage: {page_data['title']}")
        print(f"Page size: {page_data['page_size']}")
        print(f"Elements found: {len(page_data['elements'])}")
        
        # Show first few elements
        print("\nSample elements:")
        for i, elem in enumerate(page_data['elements'][:5]):
            print(f"\n{i+1}. Type: {elem['type']}")
            print(f"   Text: {elem['text'][:100]}...")
            print(f"   BBox: {elem['bbox']}")
    
    finally:
        analyzer.close_driver()


if __name__ == "__main__":
    demo_dom_analyzer()
