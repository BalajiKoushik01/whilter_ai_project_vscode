"""
License Plate Recognition Module
=================================

This module extracts license plate text from vehicle bounding boxes using EasyOCR.

It includes functionality for:
- Reading license plate text from vehicle images
- Saving cropped license plate regions
- Handling various error conditions
"""

import easyocr
import cv2
import os
import logging
import time
from typing import List, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OCR Reader with error handling
reader = None
try:
    reader = easyocr.Reader(['en'])
    logger.info("[SUCCESS] EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"[ERROR] Failed to initialize EasyOCR: {e}")


def load_ocr_reader() -> Optional[Any]:
    """
    Load the EasyOCR reader with error handling.

    Returns:
        EasyOCR reader object or None if initialization fails
    """
    global reader
    if reader is not None:
        return reader

    try:
        reader = easyocr.Reader(['en'])
        logger.info("[SUCCESS] EasyOCR initialized successfully")
        return reader
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize EasyOCR: {e}")
        return None


def extract_license_plates(image_path: str, car_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Extract license plate texts from detected vehicle boxes and save cropped regions.

    Args:
        image_path: Path to the input image
        car_boxes: List of vehicle bounding boxes as (x1, y1, x2, y2) tuples

    Returns:
        List of tuples with (plate_text, bounding_box)
    """
    plates = []

    # Validate inputs
    if not image_path or not isinstance(image_path, str):
        logger.error("[ERROR] Invalid image path provided")
        return plates

    if not car_boxes or not isinstance(car_boxes, list):
        logger.info("[INFO] No vehicle boxes provided for license plate extraction")
        return plates

    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"[ERROR] Image not found: {image_path}")
        return plates

    # Load OCR reader if not already loaded
    ocr_reader = load_ocr_reader()
    if ocr_reader is None:
        logger.error("[ERROR] OCR reader not available. Cannot perform plate recognition.")
        return plates

    # Read image with error handling
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"[ERROR] Failed to read image: {image_path}")
            return plates
    except Exception as e:
        logger.error(f"[ERROR] Error reading image: {e}")
        return plates

    # Create output directory
    try:
        os.makedirs("outputs/plates/cropped", exist_ok=True)
    except Exception as e:
        logger.error(f"[ERROR] Failed to create output directory: {e}")
        return plates

    # Process each vehicle box
    start_time = time.time()
    valid_plates = 0

    for idx, box in enumerate(car_boxes):
        try:
            # Validate box format
            if len(box) != 4:
                logger.warning(f"[WARNING] Invalid box format at index {idx}, skipping")
                continue

            x1, y1, x2, y2 = box

            # Validate coordinates
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                logger.warning(f"[WARNING] Invalid box coordinates at index {idx}: {box}, skipping")
                continue

            # Crop image
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                logger.warning(f"[WARNING] Empty crop region at index {idx}, skipping")
                continue

            # Save cropped image
            try:
                crop_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_plate{idx}.jpg"
                cropped_path = os.path.join("outputs/plates/cropped", crop_name)
                cv2.imwrite(cropped_path, cropped)
                logger.debug(f"[DEBUG] Saved cropped plate image to {cropped_path}")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to save cropped image: {e}")
                # Continue processing even if saving fails

            # Perform OCR with error handling
            try:
                results = ocr_reader.readtext(cropped)
                best_text = ''
                max_conf = 0

                for (_, text, conf) in results:
                    if conf > max_conf and text.strip():
                        best_text = text.strip()
                        max_conf = conf

                if best_text:
                    plates.append((best_text, box))
                    valid_plates += 1
                    logger.info(f"[SUCCESS] Detected license plate: {best_text} (confidence: {max_conf:.2f})")
                else:
                    logger.debug(f"[DEBUG] No text detected in vehicle box {idx}")
            except Exception as e:
                logger.warning(f"[WARNING] OCR failed for vehicle box {idx}: {e}")

        except Exception as e:
            logger.warning(f"[WARNING] Error processing vehicle box {idx}: {e}")
            continue

    process_time = time.time() - start_time
    if valid_plates > 0:
        logger.info(f"[SUCCESS] Extracted {valid_plates} license plates in {process_time:.2f} seconds")
    else:
        logger.info(f"[INFO] No license plates detected in {process_time:.2f} seconds")

    return plates
