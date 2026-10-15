from pathlib import Path
import argparse
import json
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATHS = [
    MODELS_DIR / "best_waste_classifier.keras",
    MODELS_DIR / "final_waste_classifier.keras",
    MODELS_DIR / "model.keras",
]
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)

def load_resources():
    model = None
    used_path = None
    for p in MODEL_PATHS:
        if p.exists():
            model = tf.keras.models.load_model(p)
            used_path = p
            break

    if model is None:
        raise FileNotFoundError(f"Trained model not found in {MODELS_DIR}. Run: python src/train.py")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"class_names.json not found in {MODELS_DIR}. Run: python src/train.py")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names, used_path

def predict_image(image_path):
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, class_names, model_path = load_resources()

    try:
        # Load and resize image; note model graph handles internal mobilenet_v2 preprocessing
        image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)
    except Exception as e:
        raise ValueError(f"Failed to load or parse image '{image_path}': {e}")

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    ranked_indices = np.argsort(probabilities)[::-1]
    ranked_predictions = [
        {"class": class_names[i], "confidence": float(probabilities[i] * 100.0)}
        for i in ranked_indices
    ]

    return {
        "model_used": model_path.name,
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index] * 100.0),
        "rankings": ranked_predictions,
    }

def main():
    parser = argparse.ArgumentParser(description="Classify waste category from an input image.")
    parser.add_argument("--image", required=True, help="Path to the image file to classify.")
    arguments = parser.parse_args()

    result = predict_image(arguments.image)

    print("\n" + "=" * 36)
    print("       ECOSORT AI PREDICTION")
    print("=" * 36)
    print(f"Model: {result['model_used']}")
    print(f"Predicted Class : {result['predicted_class'].upper()}")
    print(f"Confidence      : {result['confidence']:.2f}%")
    print("-" * 36)
    print("Ranked Probabilities:")
    for item in result["rankings"]:
        bar = "█" * int(item["confidence"] / 5)
        print(f"  {item['class']:<10} {item['confidence']:>6.2f}%  {bar}")
    print("=" * 36 + "\n")

if __name__ == "__main__":
    main()
