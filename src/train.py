from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from preprocess import load_datasets, IMAGE_SIZE, save_class_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 6
LEARNING_RATE_INITIAL = 0.0002
LEARNING_RATE_FINE_TUNE = 0.00002

def create_model(number_of_classes):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,), name="input_image")
    x = data_augmentation(inputs)
    # Preprocess internally so inference expects standard [0, 255] pixels
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        number_of_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="waste_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, base_model

def calculate_class_weights(train_dataset, num_classes):
    """Compute balanced class weights to handle dataset skew (e.g. fewer trash images)."""
    labels = []
    for _, batch_labels in train_dataset:
        labels.extend(batch_labels.numpy())
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    print(f"[INFO] Balanced Class Weights: {class_weight_dict}")
    return class_weight_dict

def save_training_plot(history_initial, history_fine=None):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    acc = history_initial.history.get("accuracy", [])
    val_acc = history_initial.history.get("val_accuracy", [])
    loss = history_initial.history.get("loss", [])
    val_loss = history_initial.history.get("val_loss", [])

    if history_fine:
        acc = acc + history_fine.history.get("accuracy", [])
        val_acc = val_acc + history_fine.history.get("val_accuracy", [])
        loss = loss + history_fine.history.get("loss", [])
        val_loss = val_loss + history_fine.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(13, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2E7D32", lw=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#1565C0", lw=2)
    if history_fine:
        plt.axvline(x=INITIAL_EPOCHS, color="gray", linestyle="--", label="Start Fine Tuning")
    plt.title("Classification Accuracy Over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#C62828", lw=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#E65100", lw=2)
    if history_fine:
        plt.axvline(x=INITIAL_EPOCHS, color="gray", linestyle="--", label="Start Fine Tuning")
    plt.title("Model Loss Over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = OUTPUTS_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"[INFO] Saved training curves to {plot_path}")

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("        ECOSORT AI - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load datasets and save clean class names
    train_dataset, validation_dataset, test_dataset, class_names = load_datasets(save_classes=True)
    num_classes = len(class_names)
    print(f"[INFO] Classes to train ({num_classes}): {class_names}")

    class_weights = calculate_class_weights(train_dataset, num_classes)
    model, base_model = create_model(num_classes)

    best_model_path = MODELS_DIR / "best_waste_classifier.keras"
    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n[PHASE 1] Training Top Classification Head (Backbone Frozen)...")
    history_initial = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks_phase1,
        class_weight=class_weights,
    )

    # Phase 2: Fine-Tuning Top Layers of MobileNetV2
    print("\n[PHASE 2] Unfreezing top layers of MobileNetV2 for Fine-Tuning...")
    base_model.trainable = True
    # Freeze the initial layers, only train top 30 layers
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    history_fine = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        initial_epoch=history_initial.epoch[-1] + 1,
        epochs=total_epochs,
        callbacks=callbacks_phase2,
        class_weight=class_weights,
    )

    # Save final model
    final_model_path = MODELS_DIR / "final_waste_classifier.keras"
    model.save(final_model_path)
    print(f"[INFO] Saved final model to {final_model_path}")

    # Generate and save combined training plots
    save_training_plot(history_initial, history_fine)

    # Evaluate best checkpoint on test set
    best_model = tf.keras.models.load_model(best_model_path)
    test_loss, test_accuracy = best_model.evaluate(test_dataset, verbose=0)

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classes": class_names,
        "initial_epochs": INITIAL_EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
    }
    with open(MODELS_DIR / "training_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 45)
    print("          FINAL EVALUATION SUMMARY")
    print("=" * 45)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print("=" * 45)

if __name__ == "__main__":
    main()