import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Assuming the dataset is located at data/train
dataset_path = "data/train"

# Re-loading the dataset as train_data to match your context
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

# 1. Get class labels using train_data.classes
labels = train_data.classes

# 2. Get class names using train_data.class_indices
class_indices = train_data.class_indices
# Map indices back to class names
index_to_class = {index: class_name for class_name, index in class_indices.items()}

# 3. Count the number of samples in each class
# np.bincount works well for counting non-negative integer arrays like class labels
class_counts = np.bincount(labels)

# 4. Print each emotion label with its corresponding image count clearly
print("\n--- Emotion Class Imbalance Check ---")
for i, count in enumerate(class_counts):
    class_name = index_to_class[i]
    print(f"Emotion: {class_name.ljust(10)} | Count: {count} images")

# 5. Plot a bar chart using matplotlib to visualize the distribution
class_names_list = [index_to_class[i] for i in range(len(class_counts))]

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names_list, class_counts, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700', '#B19CD9', '#FFB6C1'], edgecolor='black')

# Enhance the bar chart by adding the exact counts above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(class_counts) * 0.01), int(yval), ha='center', va='bottom', fontsize=10)

plt.title('Emotion Detection Dataset - Class Distribution (Train Data)', fontsize=14, fontweight='bold')
plt.xlabel('Emotion Classes', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot so it can be viewed as an image file
output_image = 'class_distribution.png'
plt.savefig(output_image, dpi=300)
print(f"\nBar chart successfully generated and saved to '{output_image}'.")
