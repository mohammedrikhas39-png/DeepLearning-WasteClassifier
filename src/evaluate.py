from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_datasets, get_class_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "best_waste_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

def get_target_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return get_class_names()

def main():
    if not MODEL_PATH.exists():
        # Fall back to final_waste_classifier or model.keras if best doesn't exist
        fallback_path = MODELS_DIR / "final_waste_classifier.keras"
        if fallback_path.exists():
            active_model_path = fallback_path
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python src/train.py")
    else:
        active_model_path = MODEL_PATH

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    class_names = get_target_class_names()

    print(f"[INFO] Loading test dataset...")
    # Do not mutate class_names during evaluation
    _, _, test_dataset, _ = load_datasets(save_classes=False)

    print(f"[INFO] Loading model from {active_model_path}...")
    model = tf.keras.models.load_model(active_model_path)

    print("[INFO] Running vectorized evaluation on test dataset...")
    # Vectorized inference for maximum speed
    raw_predictions = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(raw_predictions, axis=1)

    # Extract true labels
    true_labels = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)

    # Truncate if batches differ slightly in cardinality
    num_samples = min(len(true_labels), len(predicted_labels))
    true_labels = true_labels[:num_samples]
    predicted_labels = predicted_labels[:num_samples]

    report = classification_report(
        true_labels, predicted_labels,
        target_names=class_names[:max(np.max(true_labels) + 1, np.max(predicted_labels) + 1)],
        digits=4, zero_division=0,
    )
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(report)

    report_path = OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
    print(f"[INFO] Saved report to {report_path}")

    # Confusion matrix
    matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Waste Classifier - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {cm_path}")

    accuracy = float(np.mean(true_labels == predicted_labels))
    metrics = {
        "test_accuracy": accuracy,
        "class_names": class_names,
        "number_of_test_images": int(num_samples),
        "model_file": str(active_model_path.name),
    }
    metrics_path = OUTPUTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nFinal Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()