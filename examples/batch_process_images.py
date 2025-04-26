"""
Example: Batch process multiple images

This example demonstrates how to use the Vehicle Detection System to process
multiple images in a directory and generate detection results for each.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vehicle_detection.detector import batch_process, generate_random_colors, DEFAULT_COLORS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch process multiple images")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input directory containing images")
    parser.add_argument("--output", type=str, default="output",
                        help="Path to the output directory")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--random-colors", action="store_true",
                        help="Use random colors for different vehicle types")
    parser.add_argument("--show", action="store_true",
                        help="Display images with detections")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Ensure the input directory exists
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        print(f"Error: Input directory not found or not a directory: {args.input}")
        return 1
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Processing images in directory: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    # Choose colors
    colors = generate_random_colors() if args.random_colors else DEFAULT_COLORS
    
    # Process all images in the directory
    results = batch_process(
        input_dir=args.input,
        output_dir=args.output,
        conf_threshold=args.conf,
        custom_colors=colors,
        show=args.show
    )
    
    # Print the results
    print(f"\nBatch Processing Results:")
    print(f"Processed {len(results)} images")
    
    total_vehicles = sum(counts.get("Total", 0) for counts in results.values())
    print(f"Total vehicles detected: {total_vehicles}")
    
    # Count vehicles by type
    vehicle_types = {}
    for counts in results.values():
        for vehicle_type, count in counts.items():
            if vehicle_type != "Total":
                vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + count
    
    # Print counts by vehicle type
    for vehicle_type, count in vehicle_types.items():
        print(f"  - {vehicle_type}: {count}")
    
    print(f"\nDetailed results by image:")
    for image, counts in results.items():
        print(f"  {image}: {counts.get('Total', 0)} vehicles")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())