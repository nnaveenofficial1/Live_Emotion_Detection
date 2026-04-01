import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the path to the dataset folder
dataset_path = "data/train" # Path where emotion folders (angry, happy, etc.) are located

# 1. Use ImageDataGenerator to load the images
# 2. Apply normalization using rescale = 1./255
# 3. Split the dataset into training (80%) and validation (20%) using validation_split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Loading training dataset...")
# Load training set (80%)
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),      # 4. Set image size to (48, 48)
    color_mode='grayscale',    # 5. Use grayscale mode
    class_mode='categorical',  # 6. Use categorical class mode
    batch_size=32,             # 7. Batch size = 32
    subset='training'          # Subset for training
)

print("\nLoading validation dataset...")
# Load validation set (20%)
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),      # 4. Set image size to (48, 48)
    color_mode='grayscale',    # 5. Use grayscale mode
    class_mode='categorical',  # 6. Use categorical class mode
    batch_size=32,             # 7. Batch size = 32
    subset='validation'        # Subset for validation
)

# 8. Print the class indices to verify label mapping
print("\nVerified Label Mapping (Class Indices):")
print(train_generator.class_indices)
