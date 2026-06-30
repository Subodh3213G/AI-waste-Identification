import numpy as np
import pytest
from PIL import Image
from app.preprocessor import preprocess_for_mobilenet

def test_preprocess_for_mobilenet_shape():
    """Test that the preprocessing pipeline correctly reshapes images."""
    # Create a dummy image (e.g., 500x300, RGB)
    dummy_img = Image.new("RGB", (500, 300), color=(255, 0, 0))
    
    # Process it
    tensor = preprocess_for_mobilenet(dummy_img)
    
    # Assert correct batch shape for MobileNetV2 (1, 224, 224, 3)
    assert tensor.shape == (1, 224, 224, 3), f"Expected (1, 224, 224, 3), got {tensor.shape}"

def test_preprocess_normalization():
    """Test that pixel values are correctly normalized to [-1, 1]."""
    # Create a white image (all pixels 255)
    white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    tensor = preprocess_for_mobilenet(white_img)
    
    # The max value should be around 1.0 (since 255 -> 1.0 in MobileNetV2 scaling)
    assert np.max(tensor) <= 1.01, f"Max value too high: {np.max(tensor)}"
    
    # Create a black image (all pixels 0)
    black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    tensor_black = preprocess_for_mobilenet(black_img)
    
    # The min value should be around -1.0 (since 0 -> -1.0)
    assert np.min(tensor_black) >= -1.01, f"Min value too low: {np.min(tensor_black)}"
