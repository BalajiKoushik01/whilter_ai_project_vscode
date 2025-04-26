"""
Vehicle Detector Module
======================

This module provides functionality for detecting vehicles in images using YOLOv8.

It includes functions for:
- Loading the YOLOv8 model
- Detecting vehicles in single images
- Processing batches of images
- Customizing visualization options
"""

from ultralytics import YOLO
import cv2
import os
import platform
import time
import logging
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vehicle_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Define vehicle classes
VEHICLE_CLASSES = {
    2: "Car", 
    3: "Motorcycle", 
    5: "Bus", 
    7: "Truck"
}

# Default colors for different vehicle types
DEFAULT_COLORS = {
    "Car": (0, 255, 0),       # Green
    "Motorcycle": (0, 165, 255),  # Orange
    "Bus": (255, 0, 0),       # Blue
    "Truck": (0, 0, 255),     # Red
    "Vehicle": (255, 255, 0)  # Cyan (fallback)
}


def load_model(model_path: str = "yolov8n.pt", timeout: int = 30) -> Optional[Any]:
    """
    Load the YOLO model with timeout and error handling

    Args:
        model_path: Path to the model file
        timeout: Maximum time to wait for model loading in seconds

    Returns:
        YOLO model or None if loading fails
    """
    if not os.path.exists(model_path):
        logger.error(f"[ERROR] Model file not found: {model_path}")
        return None

    try:
        start_time = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start_time
        logger.info(f"[SUCCESS] Model loaded successfully in {load_time:.2f} seconds")
        return model
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        return None


# Load model once at module level with error handling
model = load_model()


def generate_random_colors(num_colors: int = 5) -> Dict[str, Tuple[int, int, int]]:
    """
    Generate random colors for visualization
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        Dictionary mapping class names to RGB color tuples
    """
    colors = {}
    for cls_name in VEHICLE_CLASSES.values():
        colors[cls_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    colors["Vehicle"] = (255, 255, 0)  # Default color
    return colors


def detect_vehicles(
    image_path: str, 
    save_path: Optional[str] = None, 
    show: bool = False,
    conf_threshold: float = 0.25,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    open_image: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> Tuple[List, Dict[str, int]]:
    """
    Detect vehicles in an image

    Args:
        image_path: Path to the input image
        save_path: Path to save the output image (optional)
        show: Whether to display the image with detections
        conf_threshold: Confidence threshold for detections
        custom_colors: Custom colors for different vehicle types
        open_image: Whether to open the saved image
        line_thickness: Thickness of bounding box lines
        font_scale: Scale of the font for labels

    Returns:
        Tuple containing list of vehicle boxes and dictionary with vehicle counts by type
    """
    # Check if model was loaded successfully
    if model is None:
        logger.error("[ERROR] Model not available. Cannot perform detection.")
        return [], {}

    # Validate input path
    if not os.path.exists(image_path):
        logger.error(f"[ERROR] Input image not found: {image_path}")
        return [], {}

    # Read image with error handling
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"[ERROR] Couldn't read image at: {image_path}")
            return [], {}
    except Exception as e:
        logger.error(f"[ERROR] Error reading image: {e}")
        return [], {}

    # Use colors provided or defaults
    colors = custom_colors if custom_colors else DEFAULT_COLORS

    # Process image with timeout
    try:
        start_time = time.time()
        results = model(image_path, conf=conf_threshold)
        process_time = time.time() - start_time
        logger.info(f"[SUCCESS] Image processed in {process_time:.2f} seconds")
    except Exception as e:
        logger.error(f"[ERROR] YOLOv8 failed to process the image: {e}")
        return [], {}

    # Get first result
    if not results or len(results) == 0:
        logger.warning("[WARNING] No results returned from model")
        return [], {}

    result = results[0]

    # Filter boxes for vehicle classes only
    vehicle_boxes = []
    vehicle_counts = {cls_name: 0 for cls_name in VEHICLE_CLASSES.values()}
    vehicle_counts["Total"] = 0

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Skip if confidence is below threshold
        if conf < conf_threshold:
            continue

        # Check if it's a vehicle class
        if cls in VEHICLE_CLASSES:
            vehicle_boxes.append(box)
            class_name = VEHICLE_CLASSES.get(cls, "Vehicle")
            vehicle_counts[class_name] += 1
            vehicle_counts["Total"] += 1

    if len(vehicle_boxes) == 0:
        logger.info("[INFO] No vehicles detected.")
        return [], vehicle_counts

    logger.info(f"[SUCCESS] Detected {len(vehicle_boxes)} vehicle(s).")

    # Create a copy of the image for drawing
    output_image = image.copy()

    # Draw bounding boxes for detected vehicles
    for box in vehicle_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        # Get class name
        class_name = VEHICLE_CLASSES.get(cls, "Vehicle")

        # Get color for this class
        color = colors.get(class_name, colors["Vehicle"])

        # Draw rectangle and label
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, line_thickness)

        # Draw background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)[0]
        cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)

        # Draw text
        cv2.putText(
            output_image, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255), 
            line_thickness
        )

    # Show detection statistics
    stats_text = []
    for cls_name, count in vehicle_counts.items():
        if count > 0 or cls_name == "Total":
            stats_text.append(f"{cls_name}: {count}")

    logger.info("Vehicle counts: " + ", ".join(stats_text))

    # Implement show parameter functionality
    if show:
        cv2.imshow("Detected Vehicles", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save the image
            cv2.imwrite(save_path, output_image)
            logger.info(f"[SUCCESS] Output image saved to: {save_path}")

            # Open the image if requested
            if open_image:
                # Convert to absolute path to ensure it can be found
                abs_save_path = os.path.abspath(save_path)
                if os.path.exists(abs_save_path):
                    try:
                        if platform.system() == "Windows":
                            os.startfile(abs_save_path)
                        elif platform.system() == "Darwin":  # macOS
                            import subprocess
                            subprocess.call(["open", abs_save_path])
                        else:  # Linux
                            import subprocess
                            subprocess.call(["xdg-open", abs_save_path])
                    except Exception as e:
                        logger.warning(f"[WARNING] Couldn't open image: {e}")
                else:
                    logger.warning(f"[WARNING] Output image not found at: {abs_save_path}")
        except PermissionError as e:
            logger.error(f"[ERROR] Permission error when saving image: {e}")
        except Exception as e:
            logger.error(f"[ERROR] Error saving image: {e}")

    return vehicle_boxes, vehicle_counts


def batch_process(
    input_dir: str, 
    output_dir: str, 
    conf_threshold: float = 0.25,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    show: bool = False,
    open_images: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Process all images in a directory

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        conf_threshold: Confidence threshold for detections
        custom_colors: Custom colors for different vehicle types
        extensions: List of valid file extensions to process
        show: Whether to display images with detections
        open_images: Whether to open saved images

    Returns:
        Dictionary mapping filenames to vehicle counts
    """
    if not os.path.exists(input_dir):
        logger.error(f"[ERROR] Input directory not found: {input_dir}")
        return {}

    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"[ERROR] Failed to create output directory: {e}")
        return {}

    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    if not image_files:
        logger.warning(f"[WARNING] No image files found in {input_dir}")
        return {}

    logger.info(f"[INFO] Found {len(image_files)} images to process")

    # Process each image
    results = {}
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_detected{os.path.splitext(filename)[1]}")

        logger.info(f"[INFO] Processing image {i+1}/{len(image_files)}: {filename}")

        try:
            _, vehicle_counts = detect_vehicles(
                image_path=image_path,
                save_path=output_path,
                show=show,
                conf_threshold=conf_threshold,
                custom_colors=custom_colors,
                open_image=open_images
            )
            results[filename] = vehicle_counts
        except Exception as e:
            logger.error(f"[ERROR] Error processing {filename}: {e}")
            results[filename] = {"error": str(e)}

    # Print summary
    total_vehicles = sum(counts.get("Total", 0) for counts in results.values())
    logger.info(f"[SUCCESS] Batch processing complete. Detected {total_vehicles} vehicles across {len(image_files)} images.")

    return results