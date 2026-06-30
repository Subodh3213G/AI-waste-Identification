"""Image preprocessing module for EcoSort AI.

Handles image loading, validation, resizing, normalization,
and tensor shaping for MobileNetV2 inference.
"""
import numpy as np
from PIL import Image


def load_and_validate_image(uploaded_file) -> Image.Image:
    """Load an uploaded file and return a validated RGB PIL Image."""
    try:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")


def preprocess_for_mobilenet(pil_image: Image.Image) -> np.ndarray:
    """Resize, normalize, and shape an image into a MobileNetV2 input tensor.

    MobileNetV2 expects 224x224 RGB images scaled to the [-1, 1] range.

    Args:
        pil_image: A PIL Image in RGB mode.

    Returns:
        A NumPy array with shape (1, 224, 224, 3) and float32 dtype.
    """
    # 1. Resize while maintaining aspect ratio (scale shortest edge to 224)
    width, height = pil_image.size
    target_size = 224
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    
    img = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 2. Center crop exactly 224x224 to avoid stretching/distorting the waste object
    left = (new_width - target_size) / 2
    top = (new_height - target_size) / 2
    right = (new_width + target_size) / 2
    bottom = (new_height + target_size) / 2
    
    img = img.crop((left, top, right, bottom))
    img_array = np.array(img)

    # Handle grayscale images by stacking channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Handle RGBA images by dropping the alpha channel
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Expand dimensions to batch shape: (1, 224, 224, 3)
    img_tensor = np.expand_dims(img_array, axis=0)

    # Scale pixel values to MobileNetV2 expectations [-1, 1]
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return preprocess_input(img_tensor.astype(np.float32))


def get_image_metadata(pil_image: Image.Image) -> dict:
    """Extract basic metadata from a PIL Image."""
    return {
        "width": pil_image.width,
        "height": pil_image.height,
        "mode": pil_image.mode,
        "format": getattr(pil_image, "format", None) or "Unknown",
    }
