import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.compiler.mlcompute import mlcompute
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Select the GPU device on M1 Mac
mlcompute.set_mlc_device(device_name='gpu')
print(mlcompute.set_mlc_device(device_name='gpu'))
# Define constants
TRAIN_DIR = 'chest_xray/train'
TEST_DIR = 'chest_xray/test'
VALID_DIR = 'chest_xray/val'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

# Load the VGG19 model with ImageNet weights, excluding the top layers
base_model = VGG19(weights='imagenet', include_top=False, input_shape=TARGET_SIZE + (3,))

# Add custom layers on top of VGG19
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Unfreeze the last few layers of the base model for fine-tuning
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators with more augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
valid_generator = test_datagen.flow_from_directory(VALID_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

# Train the model for more epochs
history = model.fit(train_generator, epochs=10, validation_data=valid_generator)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# Predict and calculate classification report and confusion matrix
y_true = test_generator.classes
y_pred = model.predict(test_generator) > 0.5
y_pred = y_pred.astype(int).flatten()

print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save the model
model.save('vgg19_chest_xray_pneumonia_optimized.h5')
