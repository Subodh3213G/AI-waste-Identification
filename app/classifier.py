"""MobileNetV2 waste classification module for EcoSort AI.

Loads a pre-trained MobileNetV2 model and maps ImageNet predictions
to four waste categories: Organic, Recyclable, Hazardous, E-waste.
"""
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# ImageNet label → Waste category mapping
# ---------------------------------------------------------------------------
WASTE_CATEGORY_MAP = {
    # Organic
    "banana": "Organic", "orange": "Organic", "lemon": "Organic",
    "apple": "Organic", "strawberry": "Organic", "pineapple": "Organic",
    "corn": "Organic", "mushroom": "Organic", "broccoli": "Organic",
    "cauliflower": "Organic", "bell_pepper": "Organic", "cucumber": "Organic",
    "head_cabbage": "Organic", "zucchini": "Organic", "fig": "Organic",
    "pomegranate": "Organic", "jackfruit": "Organic", "custard_apple": "Organic",
    "carbonara": "Organic", "pizza": "Organic", "burrito": "Organic",
    "meat_loaf": "Organic", "bagel": "Organic", "pretzel": "Organic",
    "cheeseburger": "Organic", "hotdog": "Organic", "ice_cream": "Organic",
    "potpie": "Organic", "espresso": "Organic",

    # Recyclable
    "water_bottle": "Recyclable", "pop_bottle": "Recyclable",
    "wine_bottle": "Recyclable", "beer_bottle": "Recyclable",
    "bottle_cap": "Recyclable", "beer_glass": "Recyclable",
    "goblet": "Recyclable", "cup": "Recyclable", "coffee_mug": "Recyclable",
    "plastic_bag": "Recyclable", "envelope": "Recyclable",
    "packet": "Recyclable", "carton": "Recyclable",
    "paper_towel": "Recyclable", "pitcher": "Recyclable",
    "bucket": "Recyclable", "shopping_basket": "Recyclable",
    "crate": "Recyclable", "tray": "Recyclable", "pot": "Recyclable",
    "vase": "Recyclable", "jug": "Recyclable", "rain_barrel": "Recyclable",
    "wooden_spoon": "Recyclable", "ladle": "Recyclable",
    "mixing_bowl": "Recyclable", "soup_bowl": "Recyclable",
    "tennis_ball": "Recyclable", "basketball": "Recyclable",
    "volleyball": "Recyclable", "soccer_ball": "Recyclable",

    # Hazardous
    "syringe": "Hazardous", "pill_bottle": "Hazardous",
    "gas_pump": "Hazardous", "lighter": "Hazardous",
    "matchstick": "Hazardous", "candle": "Hazardous",
    "torch": "Hazardous", "fire_engine": "Hazardous",

    # E-waste
    "cellular_telephone": "E-waste", "laptop": "E-waste",
    "desktop_computer": "E-waste", "monitor": "E-waste",
    "television": "E-waste", "screen": "E-waste",
    "mouse": "E-waste", "remote_control": "E-waste",
    "iPod": "E-waste", "digital_watch": "E-waste",
    "digital_clock": "E-waste", "power_drill": "E-waste",
    "vacuum": "E-waste", "electric_fan": "E-waste",
    "iron": "E-waste", "toaster": "E-waste",
    "microwave": "E-waste", "refrigerator": "E-waste",
    "washer": "E-waste", "dishwasher": "E-waste",
    "printer": "E-waste", "joystick": "E-waste",
    "loudspeaker": "E-waste", "CD_player": "E-waste",
    "cassette_player": "E-waste", "hard_disc": "E-waste",
    "modem": "E-waste", "projector": "E-waste",
    "typewriter_keyboard": "E-waste", "space_heater": "E-waste",
    "notebook": "E-waste", "hand_held_computer": "E-waste",
    "web_site": "E-waste",
}

DEFAULT_CATEGORY = "Recyclable"

CATEGORY_COLORS = {
    "Organic": "#84CC16",
    "Recyclable": "#3B82F6",
    "Hazardous": "#EF4444",
    "E-waste": "#8B5CF6",
}

BIN_COLORS = {
    "Organic": "Green 🟢",
    "Recyclable": "Blue 🔵",
    "Hazardous": "Red 🔴",
    "E-waste": "Orange 🟠",
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model():
    """Load the pre-trained MobileNetV2 model (ImageNet weights)."""
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    return MobileNetV2(weights="imagenet")


def classify_image(model, preprocessed_tensor: np.ndarray) -> list[dict]:
    """Run inference and return top-5 predictions with waste categories."""
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

    preds = model.predict(preprocessed_tensor, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]

    results = []
    for class_id, label, confidence in decoded:
        clean_label = label.replace("_", " ").title()
        waste_cat = WASTE_CATEGORY_MAP.get(label, DEFAULT_CATEGORY)
        results.append({
            "label": clean_label,
            "raw_label": label,
            "confidence": round(float(confidence), 4),
            "waste_category": waste_cat,
            "category_color": CATEGORY_COLORS.get(waste_cat, "#3B82F6"),
            "bin_color": BIN_COLORS.get(waste_cat, "Blue 🔵"),
        })
    return results


def get_mock_classification() -> list[dict]:
    """Return mock predictions for demo mode (no TensorFlow)."""
    import random
    primary_options = [
        ("Water Bottle", "water_bottle", "Recyclable", 0.87),
        ("Banana", "banana", "Organic", 0.93),
        ("Cellular Telephone", "cellular_telephone", "E-waste", 0.81),
        ("Syringe", "syringe", "Hazardous", 0.76),
    ]
    choice = random.choice(primary_options)
    primary = {
        "label": choice[0], "raw_label": choice[1],
        "confidence": choice[3], "waste_category": choice[2],
        "category_color": CATEGORY_COLORS[choice[2]],
        "bin_color": BIN_COLORS[choice[2]],
    }
    others = [
        {"label": "Plastic Bag", "raw_label": "plastic_bag", "confidence": 0.06,
         "waste_category": "Recyclable", "category_color": "#3B82F6", "bin_color": "Blue 🔵"},
        {"label": "Envelope", "raw_label": "envelope", "confidence": 0.03,
         "waste_category": "Recyclable", "category_color": "#3B82F6", "bin_color": "Blue 🔵"},
        {"label": "Paper Towel", "raw_label": "paper_towel", "confidence": 0.02,
         "waste_category": "Recyclable", "category_color": "#3B82F6", "bin_color": "Blue 🔵"},
        {"label": "Cup", "raw_label": "cup", "confidence": 0.01,
         "waste_category": "Recyclable", "category_color": "#3B82F6", "bin_color": "Blue 🔵"},
    ]
    return [primary] + others
