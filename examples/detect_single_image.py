"""
Example: Detect vehicles in a single image

This example demonstrates how to use the Vehicle Detection System to detect
vehicles in a single image and display the results.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vehicle_detection.detector import detect_vehicles


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Detect vehicles in a single image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default="output",
                        help="Path to the output directory")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--show", action="store_true",
                        help="Display the image with detections")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Ensure the input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        return 1
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Determine the output path
    filename = os.path.basename(args.image)
    output_path = os.path.join(args.output, f"{os.path.splitext(filename)[0]}_detected{os.path.splitext(filename)[1]}")
    
    print(f"Processing image: {args.image}")
    print(f"Output will be saved to: {output_path}")
    
    # Detect vehicles in the image
    boxes, counts = detect_vehicles(
        image_path=args.image,
        save_path=output_path,
        show=args.show,
        conf_threshold=args.conf
    )
    
    # Print the results
    print(f"\nDetection Results:")
    print(f"Total vehicles detected: {counts.get('Total', 0)}")
    
    for vehicle_type, count in counts.items():
        if vehicle_type != "Total" and count > 0:
            print(f"  - {vehicle_type}: {count}")
    
    print(f"\nOutput image saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())