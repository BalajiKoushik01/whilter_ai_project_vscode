"""
Unit tests for the detector module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

from vehicle_detection.detector import (
    load_model,
    generate_random_colors,
    detect_vehicles,
    batch_process,
    VEHICLE_CLASSES,
    DEFAULT_COLORS
)


def test_vehicle_classes():
    """Test that VEHICLE_CLASSES contains the expected classes."""
    assert 2 in VEHICLE_CLASSES
    assert 3 in VEHICLE_CLASSES
    assert 5 in VEHICLE_CLASSES
    assert 7 in VEHICLE_CLASSES
    assert VEHICLE_CLASSES[2] == "Car"
    assert VEHICLE_CLASSES[3] == "Motorcycle"
    assert VEHICLE_CLASSES[5] == "Bus"
    assert VEHICLE_CLASSES[7] == "Truck"


def test_default_colors():
    """Test that DEFAULT_COLORS contains the expected colors."""
    assert "Car" in DEFAULT_COLORS
    assert "Motorcycle" in DEFAULT_COLORS
    assert "Bus" in DEFAULT_COLORS
    assert "Truck" in DEFAULT_COLORS
    assert "Vehicle" in DEFAULT_COLORS
    assert len(DEFAULT_COLORS["Car"]) == 3  # RGB tuple


def test_generate_random_colors():
    """Test that generate_random_colors returns the expected format."""
    colors = generate_random_colors()
    assert isinstance(colors, dict)
    assert "Car" in colors
    assert "Motorcycle" in colors
    assert "Bus" in colors
    assert "Truck" in colors
    assert "Vehicle" in colors
    assert len(colors["Car"]) == 3  # RGB tuple


@patch("vehicle_detection.detector.YOLO")
def test_load_model_success(mock_yolo):
    """Test that load_model returns a model when successful."""
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    # Create a temporary model file
    with open("temp_model.pt", "w") as f:
        f.write("dummy model")
    
    try:
        model = load_model("temp_model.pt")
        assert model is not None
        assert model == mock_model
        mock_yolo.assert_called_once_with("temp_model.pt")
    finally:
        # Clean up
        if os.path.exists("temp_model.pt"):
            os.remove("temp_model.pt")


def test_load_model_file_not_found():
    """Test that load_model returns None when the model file is not found."""
    model = load_model("nonexistent_model.pt")
    assert model is None


@patch("vehicle_detection.detector.model")
@patch("vehicle_detection.detector.cv2.imread")
def test_detect_vehicles_image_not_found(mock_imread, mock_model):
    """Test that detect_vehicles handles missing images correctly."""
    mock_imread.return_value = None
    
    boxes, counts = detect_vehicles("nonexistent_image.jpg")
    
    assert boxes == []
    assert counts == {}
    mock_imread.assert_called_once_with("nonexistent_image.jpg")


@patch("vehicle_detection.detector.os.path.exists")
def test_batch_process_input_dir_not_found(mock_exists):
    """Test that batch_process handles missing input directories correctly."""
    mock_exists.return_value = False
    
    results = batch_process("nonexistent_dir", "output_dir")
    
    assert results == {}
    mock_exists.assert_called_once_with("nonexistent_dir")


if __name__ == "__main__":
    pytest.main()