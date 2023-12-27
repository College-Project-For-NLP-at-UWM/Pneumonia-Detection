import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

# Load the saved model
model = models.load_model('vgg19_chest_xray_pneumonia_optimized.h5')

# Load the test dataset
TEST_DIR = 'chest_xray/test'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)


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

# Load an image for explanation
image_path = 'chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'  # Example pneumonia image
original_image = load_img(image_path, target_size=TARGET_SIZE)
image_array = img_to_array(original_image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()

# Define a custom prediction function for LIME
def predict_fn(images):
    return model.predict(images)

# Explain the prediction
segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
explanation = explainer.explain_instance(
    image_array[0], 
    predict_fn, 
    top_labels=2, 
    hide_color=None, 
    num_samples=1000, 
    segmentation_fn=segmentation_fn
)

# Display the explanation
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=False, 
    num_features=10, 
    hide_rest=False
)
plt.imshow(mark_boundaries(temp, mask))
plt.show()

import shap

# Load an image for explanation
image_path = 'chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'  # Example pneumonia image
original_image = load_img(image_path, target_size=TARGET_SIZE)
image_array = img_to_array(original_image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Prepare a background dataset
num_background_samples = 50
background_data = np.zeros((num_background_samples, *TARGET_SIZE, 3))
counter = 0
for images, _ in test_generator:
    num_images = images.shape[0]
    remaining_space = num_background_samples - counter

    if remaining_space >= num_images:
        background_data[counter:counter + num_images] = images
        counter += num_images
    else:
        background_data[counter:counter + remaining_space] = images[:remaining_space]
        counter += remaining_space
    if counter >= num_background_samples:
        break


# Initialize the GradientExplainer with the VGG19 model and background data
explainer = shap.GradientExplainer(model, background_data)

# Calculate SHAP values for the given image
shap_values, _ = explainer.shap_values(image_array, ranked_outputs=2)

# Plot the SHAP values
shap.image_plot(shap_values, image_array)

