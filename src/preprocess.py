from pathlib import Path
import json
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
AUTOTUNE = tf.data.AUTOTUNE
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_class_names():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Create data/raw/ and add one folder per waste class."
        )
    # Only keep directories that contain at least one valid image file
    valid_classes = []
    for folder in sorted(DATA_DIR.iterdir()):
        if folder.is_dir():
            image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            if len(image_files) > 0:
                valid_classes.append(folder.name)
            else:
                print(f"[WARNING] Skipping empty or non-image directory: {folder.name}")

    if len(valid_classes) < 2:
        raise ValueError(
            f"At least two non-empty class folders are required in {DATA_DIR}. "
            f"Found: {valid_classes}"
        )
    return valid_classes

def save_class_names(class_names, target_dir=MODELS_DIR):
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    print(f"[INFO] Saved {len(class_names)} classes to {target_dir / 'class_names.json'}")

def load_datasets(save_classes=False):
    class_names = get_class_names()
    if save_classes:
        save_class_names(class_names)

    train_dataset, val_full_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        validation_split=0.30,
        subset="both",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_batches = tf.data.experimental.cardinality(val_full_dataset).numpy()
    if validation_batches < 2:
        raise ValueError("Dataset is too small for validation and testing split.")

    test_batches = max(1, validation_batches // 2)

    # Cache before splitting to guarantee deterministic, disjoint partitions
    val_full_cached = val_full_dataset.cache()
    test_dataset = val_full_cached.take(test_batches).prefetch(AUTOTUNE)
    validation_dataset = val_full_cached.skip(test_batches).prefetch(AUTOTUNE)

    train_dataset = train_dataset.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names

if __name__ == "__main__":
    train_ds, val_ds, test_ds, names = load_datasets(save_classes=False)
    print(f"Classes ({len(names)}):", names)
    print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())