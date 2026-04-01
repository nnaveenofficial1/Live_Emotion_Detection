import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset (replicating train_data setup)
dataset_path = "data/train"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

print("Loading dataset...")
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='training'
)

# 1. Use train_data.classes to get labels
labels = train_data.classes

# 2. Compute class weights using sklearn's compute_class_weight
# 'balanced' mode automatically adjusts weights inversely proportional to class frequencies.
unique_classes = np.unique(labels)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=labels
)

# 3. Map weights to each class index
class_indices = train_data.class_indices
index_to_class = {index: class_name for class_name, index in class_indices.items()}

# Create dictionary mapping class indices to class weights (format expected by Keras)
class_weights_dict = {i: class_weights_array[i] for i in unique_classes}

# 4. Print class weights clearly along with class names
print("\n--- Computed Class Weights ---")
for index, weight in class_weights_dict.items():
    class_name = index_to_class[index]
    print(f"Emotion: {class_name.ljust(10)} | Class Index: {index} | Weight: {weight:.4f}")
