import cv2
import pytesseract
from PIL import Image
import numpy as np
import os
import google.generativeai as genai
import hashlib
import pickle
import enchant
from difflib import SequenceMatcher
import time
import requests

# Global timeout variable - change this as needed
GEMINI_TIMEOUT = 10  # seconds

# Google API Configuration
GOOGLE_API_KEY = "AIzaSyAoPywSoxoTbytGwQYI0DN7wBrYGdMYeHk"
genai.configure(api_key=GOOGLE_API_KEY)
img_path = "1.jpg"

class ImageTextRecognizer:
    def __init__(self):
        # Set tesseract path if needed (uncomment and modify for Windows)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Softwares\Tesseract\tesseract.exe"
        
        # Initialize spell checker
        try:
            self.spell_checker = enchant.Dict("en_US")
        except:
            print("Warning: enchant not available. Spell correction disabled.")
            self.spell_checker = None
        
        # Cache file for storing results
        self.cache_file = "ocr_cache.pkl"
        self.cache = self._load_cache()
        
        # Gemini model
        try:
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            print("Warning: Gemini model initialization failed")
            self.gemini_model = None
    
    def _load_cache(self):
        """Load cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except:
            pass
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _get_image_hash(self, image_path):
        """Generate hash for image to use as cache key"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return hashlib.md5(image_data).hexdigest()
        except:
            return None
    
    def _check_internet_connection(self):
        """Check if internet connection is available"""
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except:
            return False
    
    def _preserve_line_breaks(self, text):
        """Clean up text while preserving line breaks"""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def spell_correct_word(self, word):
        """Correct a single word using spell checker"""
        if not self.spell_checker or not word.isalpha():
            return word
        
        # If word is already correct, return it
        if self.spell_checker.check(word):
            return word
        
        # Get suggestions
        suggestions = self.spell_checker.suggest(word)
        if not suggestions:
            return word
        
        # Return the suggestion with highest similarity
        best_match = max(suggestions, key=lambda x: SequenceMatcher(None, word.lower(), x.lower()).ratio())
        
        # Only return suggestion if similarity is high enough
        if SequenceMatcher(None, word.lower(), best_match.lower()).ratio() > 0.6:
            # Preserve original case pattern
            if word.isupper():
                return best_match.upper()
            elif word.istitle():
                return best_match.capitalize()
            else:
                return best_match
        
        return word
    
    def spell_correct_text(self, text):
        """Apply spell correction to entire text while preserving line breaks"""
        if not self.spell_checker or not text:
            return text
        
        lines = text.split('\n')
        corrected_lines = []
        
        for line in lines:
            words = line.split()
            corrected_words = []
            
            for word in words:
                # Extract alphabetic part and preserve punctuation
                cleaned_word = ''.join(c for c in word if c.isalpha())
                if cleaned_word:
                    corrected = self.spell_correct_word(cleaned_word)
                    # Replace the alphabetic part in the original word
                    corrected_word = word.replace(cleaned_word, corrected, 1)
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            
            corrected_lines.append(' '.join(corrected_words))
        
        return '\n'.join(corrected_lines)
    
    def extract_text_with_gemini(self, image_path):
        """Extract text using Google Gemini API"""
        try:
            if not self.gemini_model:
                return None, "Gemini model not available"
            
            # Check internet connection
            if not self._check_internet_connection():
                return None, "No internet connection"
            
            # Load image
            img = Image.open(image_path)
            
            instruction = "Recognize and extract all text, numbers, symbols, and alphabet characters from this image. Preserve the original line breaks and formatting. Return the text exactly as it appears in the image, maintaining the layout and structure."
            
            # Set timeout for the API call
            response = self.gemini_model.generate_content(
                [instruction, img],
                request_options={"timeout": GEMINI_TIMEOUT}
            )
            
            # Check if response is valid
            if response and response.text:
                result_text = self._preserve_line_breaks(response.text)
                return result_text, "Success"
            else:
                return None, "Empty response from API"
                
        except Exception as e:
            return None, f"API error: {str(e)}"
    
    def preprocess_image(self, image_path):
        """Preprocess the image to improve OCR accuracy"""
        try:
            # Read image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply noise removal
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply threshold to get binary image
            thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            return processed
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return None
    
    def extract_text_with_tesseract(self, image_path):
        """Extract text using Tesseract OCR with preprocessing and spell correction"""
        try:
            # Try with preprocessing first
            processed_img = self.preprocess_image(image_path)
            
            if processed_img is not None:
                # Custom OCR configuration for better line break preservation
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(processed_img, config=custom_config)
            else:
                # Fallback to original image
                img = Image.open(image_path)
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(img, config=custom_config)
            
            # Preserve line breaks and clean up text
            text = self._preserve_line_breaks(text)
            
            # Apply spell correction
            if text:
                text = self.spell_correct_text(text)
            
            return text, "Success"
            
        except Exception as e:
            return None, f"Tesseract error: {str(e)}"
    
    def extract_text(self, image_path):
        """Main method that automatically processes image with fallback logic"""
        if not os.path.exists(image_path):
            print("❌ Image file not found!")
            return None
        
        print(f"📷 Processing image: {os.path.basename(image_path)}")
        print("="*60)
        
        # Step 1: Check cache first
        image_hash = self._get_image_hash(image_path)
        if image_hash and image_hash in self.cache:
            print("📋 Result source: CACHED")
            print("-" * 40)
            print(self.cache[image_hash])
            return self.cache[image_hash]
        
        # Step 2: Try Gemini API
        gemini_result, gemini_status = self.extract_text_with_gemini(image_path)
        
        if gemini_result:
            print("🌐 Result source: ONLINE (Gemini AI)")
            print("-" * 40)
            print(gemini_result)
            
            # Cache the successful result
            if image_hash:
                self.cache[image_hash] = gemini_result
                self._save_cache()
            
            return gemini_result
        
        else:
            print(f"⚠️  Gemini API failed: {gemini_status}")
            print("🔄 Falling back to offline OCR...")
            
            # Step 3: Fallback to Tesseract OCR
            tesseract_result, tesseract_status = self.extract_text_with_tesseract(image_path)
            
            if tesseract_result:
                print("💻 Result source: OFFLINE (Tesseract + Dictionary)")
                print("-" * 40)
                print(tesseract_result)
                
                # Cache offline results too (in case API is temporarily down)
                if image_hash:
                    self.cache[image_hash] = tesseract_result
                    self._save_cache()
                
                return tesseract_result
            else:
                print(f"❌ Tesseract also failed: {tesseract_status}")
                print("💔 No text could be extracted from the image")
                return None
    
    def get_text_bounding_boxes(self, image_path, output_path="text_detection_result.jpg"):
        """Generate image with bounding boxes around detected text"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Get bounding box data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Draw bounding boxes
            n_boxes = len(data['level'])
            boxes_drawn = 0
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Only draw boxes for confident detections
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add text label
                    text = data['text'][i].strip()
                    if text:
                        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        boxes_drawn += 1
            
            if boxes_drawn > 0:
                cv2.imwrite(output_path, img)
                print(f"✅ Bounding boxes image saved as: {output_path}")
                return True
            else:
                print("⚠️  No text regions detected for bounding boxes")
                return False
        
        except Exception as e:
            print(f"❌ Error creating bounding boxes: {str(e)}")
            return False

def main():
    # Initialize the recognizer
    recognizer = ImageTextRecognizer()
    
    print("🚀 Enhanced OCR System - Auto Processing Mode")
    print(f"⏱️  Gemini API timeout: {GEMINI_TIMEOUT} seconds")
    print("="*60)
    
    # Get image path from user
    image_path = img_path
    
    # Process the image automatically
    result = recognizer.extract_text(image_path)
    
    if result:
        print("\n" + "="*60)
        # Ask if user wants bounding boxes
        save_boxes = input("📦 Generate image with text bounding boxes? (y/n): ").lower().strip()
        if save_boxes == 'y':
            recognizer.get_text_bounding_boxes(image_path)
    
    print("\n✨ Processing complete!")

if __name__ == "__main__":
    main()


# import cv2
# import pytesseract
# from PIL import Image
# import numpy as np
# import os
# import google.generativeai as genai
# import hashlib
# import pickle
# import enchant
# from difflib import SequenceMatcher
# import time
# import requests

# # Global timeout variable - change this as needed
# GEMINI_TIMEOUT = 30  # seconds
# img_path = "1.jpg"

# # Google API Configuration
# GOOGLE_API_KEY = "AIzaSyAoPywSoxoTbytGwQYI0DN7wBrYGdMYeHk"
# genai.configure(api_key=GOOGLE_API_KEY)

# class ImageTextRecognizer:
#     def __init__(self):
#         pytesseract.pytesseract.tesseract_cmd = r"C:\Softwares\Tesseract\tesseract.exe"

#         try:
#             self.spell_checker = enchant.Dict("en_US")
#         except:
#             print("Warning: enchant not available. Spell correction disabled.")
#             self.spell_checker = None
        
#         # Cache file for storing results
#         self.cache_file = "ocr_cache.pkl"
#         self.cache = self._load_cache()
        
#         # Gemini model
#         try:
#             self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
#         except:
#             print("Warning: Gemini model initialization failed")
#             self.gemini_model = None
    
#     def _load_cache(self):
#         """Load cache from file"""
#         try:
#             if os.path.exists(self.cache_file):
#                 with open(self.cache_file, 'rb') as f:
#                     return pickle.load(f)
#         except:
#             pass
#         return {}
    
#     def _save_cache(self):
#         """Save cache to file"""
#         try:
#             with open(self.cache_file, 'wb') as f:
#                 pickle.dump(self.cache, f)
#         except Exception as e:
#             print(f"Warning: Could not save cache: {e}")
    
#     def _get_image_hash(self, image_path):
#         """Generate hash for image to use as cache key"""
#         try:
#             with open(image_path, 'rb') as f:
#                 image_data = f.read()
#             return hashlib.md5(image_data).hexdigest()
#         except:
#             return None
    
#     def _check_internet_connection(self):
#         """Check if internet connection is available"""
#         try:
#             requests.get("https://www.google.com", timeout=5)
#             return True
#         except:
#             return False
    
#     def _preserve_line_breaks(self, text):
#         """Clean up text while preserving line breaks"""
#         if not text:
#             return ""
        
#         lines = text.split('\n')
#         cleaned_lines = []
        
#         for line in lines:
#             cleaned_line = line.strip()
#             if cleaned_line:  # Only add non-empty lines
#                 cleaned_lines.append(cleaned_line)
        
#         return '\n'.join(cleaned_lines)
    
#     def spell_correct_word(self, word):
#         """Correct a single word using spell checker"""
#         if not self.spell_checker or not word.isalpha():
#             return word
        
#         # If word is already correct, return it
#         if self.spell_checker.check(word):
#             return word
        
#         # Get suggestions
#         suggestions = self.spell_checker.suggest(word)
#         if not suggestions:
#             return word
        
#         # Return the suggestion with highest similarity
#         best_match = max(suggestions, key=lambda x: SequenceMatcher(None, word.lower(), x.lower()).ratio())
        
#         # Only return suggestion if similarity is high enough
#         if SequenceMatcher(None, word.lower(), best_match.lower()).ratio() > 0.6:
#             # Preserve original case pattern
#             if word.isupper():
#                 return best_match.upper()
#             elif word.istitle():
#                 return best_match.capitalize()
#             else:
#                 return best_match
        
#         return word
    
#     def spell_correct_text(self, text):
#         """Apply spell correction to entire text while preserving line breaks"""
#         if not self.spell_checker or not text:
#             return text
        
#         lines = text.split('\n')
#         corrected_lines = []
        
#         for line in lines:
#             words = line.split()
#             corrected_words = []
            
#             for word in words:
#                 # Extract alphabetic part and preserve punctuation
#                 cleaned_word = ''.join(c for c in word if c.isalpha())
#                 if cleaned_word:
#                     corrected = self.spell_correct_word(cleaned_word)
#                     # Replace the alphabetic part in the original word
#                     corrected_word = word.replace(cleaned_word, corrected, 1)
#                     corrected_words.append(corrected_word)
#                 else:
#                     corrected_words.append(word)
            
#             corrected_lines.append(' '.join(corrected_words))
        
#         return '\n'.join(corrected_lines)
    
#     def extract_text_with_gemini(self, image_path):
#         """Extract text using Google Gemini API"""
#         try:
#             if not self.gemini_model:
#                 return None, "Gemini model not available"
            
#             # Check internet connection
#             if not self._check_internet_connection():
#                 return None, "No internet connection"
            
#             # Load image
#             img = Image.open(image_path)
            
#             instruction = "Recognize and extract all text, numbers, symbols, and alphabet characters from this image. Preserve the original line breaks and formatting. Return the text exactly as it appears in the image, maintaining the layout and structure."
            
#             # Set timeout for the API call
#             response = self.gemini_model.generate_content(
#                 [instruction, img],
#                 request_options={"timeout": GEMINI_TIMEOUT}
#             )
            
#             # Check if response is valid
#             if response and response.text:
#                 result_text = self._preserve_line_breaks(response.text)
#                 return result_text, "Success"
#             else:
#                 return None, "Empty response from API"
                
#         except Exception as e:
#             return None, f"API error: {str(e)}"
    
#     def preprocess_image(self, image_path):
#         """Advanced preprocessing to improve OCR accuracy for various image types"""
#         try:
#             # Read image using OpenCV
#             img = cv2.imread(image_path)
#             if img is None:
#                 return None
            
#             # Convert to grayscale
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
#             # Apply Gaussian blur to reduce noise
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
#             # Apply adaptive threshold for better handling of varying lighting
#             adaptive_thresh = cv2.adaptiveThreshold(
#                 blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#             )
            
#             # Also try OTSU thresholding
#             _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
#             # Combine both thresholding methods
#             combined = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
            
#             # Apply morphological operations to clean up
#             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#             processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
#             processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
#             # Remove small noise
#             kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#             processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_noise)
            
#             return processed
#         except Exception as e:
#             print(f"Error in image preprocessing: {e}")
#             return None
    
#     def extract_text_with_tesseract(self, image_path):
#         """Extract text using Tesseract OCR with multiple approaches and spell correction"""
#         try:
#             # Load original image
#             original_img = cv2.imread(image_path)
#             if original_img is None:
#                 return None, "Could not load image"
            
#             best_text = ""
#             best_confidence = 0
            
#             # Method 1: Try with advanced preprocessing
#             processed_img = self.preprocess_image(image_path)
#             if processed_img is not None:
#                 configs_to_try = [
#                     r'--oem 3 --psm 6 -c preserve_interword_spaces=1',  # Uniform block
#                     r'--oem 3 --psm 8 -c preserve_interword_spaces=1',  # Single word
#                     r'--oem 3 --psm 13 -c preserve_interword_spaces=1', # Raw line
#                 ]
                
#                 for config in configs_to_try:
#                     try:
#                         text = pytesseract.image_to_string(processed_img, config=config)
#                         # Get confidence for this result
#                         data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
#                         confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
#                         avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
#                         if avg_confidence > best_confidence and text.strip():
#                             best_text = text
#                             best_confidence = avg_confidence
#                     except:
#                         continue
            
#             # Method 2: Try with original image and different PSM modes
#             pil_img = Image.open(image_path)
#             configs_to_try_original = [
#                 r'--oem 3 --psm 6',  # Uniform block
#                 r'--oem 3 --psm 7',  # Single text line
#                 r'--oem 3 --psm 8',  # Single word
#                 r'--oem 3 --psm 11', # Sparse text
#                 r'--oem 3 --psm 12', # Sparse text with OSD
#             ]
            
#             for config in configs_to_try_original:
#                 try:
#                     text = pytesseract.image_to_string(pil_img, config=config)
#                     # Get confidence
#                     data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
#                     confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
#                     avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
#                     if avg_confidence > best_confidence and text.strip():
#                         best_text = text
#                         best_confidence = avg_confidence
#                 except:
#                     continue
            
#             # Method 3: Try with image scaling for small text
#             height, width = original_img.shape[:2]
#             if height < 500 or width < 500:  # If image is small, scale it up
#                 scale_factor = 3
#                 scaled_img = cv2.resize(original_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
#                 scaled_gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
                
#                 try:
#                     text = pytesseract.image_to_string(scaled_gray, config=r'--oem 3 --psm 6')
#                     data = pytesseract.image_to_data(scaled_gray, config=r'--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
#                     confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
#                     avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
#                     if avg_confidence > best_confidence and text.strip():
#                         best_text = text
#                         best_confidence = avg_confidence
#                 except:
#                     pass
            
#             # If no good result, try one last method with edge detection
#             if best_confidence < 30:
#                 try:
#                     gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#                     edges = cv2.Canny(gray, 50, 150)
#                     # Dilate edges to connect text components
#                     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#                     edges = cv2.dilate(edges, kernel, iterations=1)
#                     # Invert so text is black on white
#                     edges = cv2.bitwise_not(edges)
                    
#                     text = pytesseract.image_to_string(edges, config=r'--oem 3 --psm 6')
#                     if text.strip():
#                         best_text = text
#                 except:
#                     pass
            
#             # Clean up and preserve line breaks
#             if best_text:
#                 best_text = self._preserve_line_breaks(best_text)
                
#                 # Apply spell correction
#                 best_text = self.spell_correct_text(best_text)
                
#                 # Filter out obvious garbage (too many non-alphanumeric characters)
#                 words = best_text.split()
#                 valid_words = []
#                 for word in words:
#                     # Keep word if it has at least 50% alphanumeric characters
#                     alnum_count = sum(1 for c in word if c.isalnum())
#                     if len(word) == 0 or alnum_count / len(word) >= 0.3:
#                         valid_words.append(word)
                
#                 final_text = ' '.join(valid_words)
#                 if final_text.strip():
#                     return final_text, f"Success (confidence: {best_confidence:.1f}%)"
            
#             return None, "No reliable text found"
            
#         except Exception as e:
#             return None, f"Tesseract error: {str(e)}"
    
#     def extract_text(self, image_path):
#         """Main method that automatically processes image with fallback logic"""
#         if not os.path.exists(image_path):
#             print("❌ Image file not found!")
#             return None
        
#         print(f"📷 Processing image: {os.path.basename(image_path)}")
#         print("="*60)
        
#         # Step 1: Check cache first
#         image_hash = self._get_image_hash(image_path)
#         if image_hash and image_hash in self.cache:
#             print("📋 Result source: CACHED")
#             print("-" * 40)
#             print(self.cache[image_hash])
#             return self.cache[image_hash]
        
#         # Step 2: Try Gemini API
#         gemini_result, gemini_status = self.extract_text_with_gemini(image_path)
        
#         if gemini_result:
#             print("🌐 Result source: ONLINE (Gemini AI)")
#             print("-" * 40)
#             print(gemini_result)
            
#             # Cache the successful result
#             if image_hash:
#                 self.cache[image_hash] = gemini_result
#                 self._save_cache()
            
#             return gemini_result
        
#         else:
#             print(f"⚠️  Gemini API failed: {gemini_status}")
#             print("🔄 Falling back to offline OCR...")
            
#             # Step 3: Fallback to Tesseract OCR
#             tesseract_result, tesseract_status = self.extract_text_with_tesseract(image_path)
            
#             if tesseract_result:
#                 print("💻 Result source: OFFLINE (Tesseract + Dictionary)")
#                 print("-" * 40)
#                 print(tesseract_result)
                
#                 # Cache offline results too (in case API is temporarily down)
#                 if image_hash:
#                     self.cache[image_hash] = tesseract_result
#                     self._save_cache()
                
#                 return tesseract_result
#             else:
#                 print(f"❌ Tesseract also failed: {tesseract_status}")
#                 print("💔 No text could be extracted from the image")
#                 return None
    
#     def get_text_bounding_boxes(self, image_path, output_path="text_detection_result.jpg"):
#         """Generate image with bounding boxes around detected text"""
#         try:
#             img = cv2.imread(image_path)
#             if img is None:
#                 return False
            
#             # Get bounding box data
#             data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
#             # Draw bounding boxes
#             n_boxes = len(data['level'])
#             boxes_drawn = 0
            
#             for i in range(n_boxes):
#                 if int(data['conf'][i]) > 30:  # Only draw boxes for confident detections
#                     (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#                     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
#                     # Add text label
#                     text = data['text'][i].strip()
#                     if text:
#                         cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#                         boxes_drawn += 1
            
#             if boxes_drawn > 0:
#                 cv2.imwrite(output_path, img)
#                 print(f"✅ Bounding boxes image saved as: {output_path}")
#                 return True
#             else:
#                 print("⚠️  No text regions detected for bounding boxes")
#                 return False
        
#         except Exception as e:
#             print(f"❌ Error creating bounding boxes: {str(e)}")
#             return False

# def main():
#     # Initialize the recognizer
#     recognizer = ImageTextRecognizer()
    
#     print("🚀 Enhanced OCR System - Auto Processing Mode")
#     print(f"⏱️  Gemini API timeout: {GEMINI_TIMEOUT} seconds")
#     print("="*60)
    
#     # Get image path from user
#     image_path = img_path
    
#     # Process the image automatically
#     result = recognizer.extract_text(image_path)
    
#     if result:
#         print("\n" + "="*60)
#         # Ask if user wants bounding boxes
#         save_boxes = input("📦 Generate image with text bounding boxes? (y/n): ").lower().strip()
#         if save_boxes == 'y':
#             recognizer.get_text_bounding_boxes(image_path)
    
#     print("\n✨ Processing complete!")

# if __name__ == "__main__":
#     main()
