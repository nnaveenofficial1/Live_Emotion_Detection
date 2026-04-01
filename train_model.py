import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from build_model import create_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Setup Generators (train_data and val_data)
dataset_path = "data/train"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)
train_data = train_datagen.flow_from_directory(
    dataset_path, target_size=(48, 48), color_mode='grayscale',
    class_mode='categorical', batch_size=32, subset='training'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data = val_datagen.flow_from_directory(
    dataset_path, target_size=(48, 48), color_mode='grayscale',
    class_mode='categorical', batch_size=32, subset='validation'
)

# 2. Setup Class Weights Dict
labels = train_data.classes
unique_classes = np.unique(labels)
class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=labels)
class_weights_dict = {i: class_weights_array[i] for i in unique_classes}

# 3. Instantiate model
model = create_model()

# 4. Setup Callbacks (EarlyStopping & ModelCheckpoint)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='best_emotion_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 5. Train the Model
print("\nStarting Training with parameters: 40 Epochs, EarlyStopping, and ModelCheckpoint...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1  # Automatically prints training and validation loss/accuracy per epoch
)

print("\nTraining execution completed and best weights are restored if EarlyStopping triggered.")

# Print final training and validation accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# Optionally save model
model.save("live_emotion_detection_model.keras")
print("Saved best model to 'live_emotion_detection_model.keras'")
