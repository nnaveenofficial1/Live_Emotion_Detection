import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "data/train"

print("1. Creating Data Augmentation Generator for Training...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

print("2. Loading Training Data with Augmentation...")
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='training'
)

print("\n3. Creating Clean Generator for Validation...")
# Crucial: Validation should NOT have augmentation applied. Just scaling.
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Loading Validation Data...")
validation_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='validation'
)

print("\n4. Verifying Batch Shapes...")
x_train_batch, y_train_batch = next(train_generator)
x_val_batch, y_val_batch = next(validation_generator)

print(f"Training Batch X shape:   {x_train_batch.shape}")
print(f"Training Batch y shape:   {y_train_batch.shape}")
print(f"Validation Batch X shape: {x_val_batch.shape}")
print(f"Validation Batch y shape: {y_val_batch.shape}")

# Optional check just to be sure scaling went through
print(f"\nTrain Batch Max Val: {x_train_batch.max()}")
print(f"Val Batch Max Val:   {x_val_batch.max()}")
