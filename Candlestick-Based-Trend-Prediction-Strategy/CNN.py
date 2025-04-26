# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:09:16 2024

@author: edree
"""
import numpy as np
import splitfolders
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalMaxPooling2D
from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
# if gpus: 
#     for gpu in gpus:
#           tf.config.experimental.set_memory_growth(gpu, True)
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7492)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
input_folder = './chart_BTC'
output_folder = './output_BTC'

#Split with a ratio of Train:Val:Test = 60%:20%:20%
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.7, .1, .2), group_prefix=None)  # Default values

batch_size = 64
img_height = 150
img_width = 150

train_datagen = ImageDataGenerator(rescale=1./255,
                               # rotation_range=40,
                               # width_shift_range=0.2,
                                #height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    output_folder + '/train/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    interpolation='bilinear')

validation_generator = test_val_datagen.flow_from_directory(
    output_folder + '/val/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    interpolation='bilinear',
    shuffle=False)

test_generator = test_val_datagen.flow_from_directory(
    output_folder + '/test/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    interpolation='bilinear',
    shuffle=False)

model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'), 
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

eval_result = model.evaluate(test_generator)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")



test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)
true_classes = test_generator.classes
true_classes = true_classes[:len(predicted_classes)]

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, digits=3))

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve points
fpr, tpr, thresholds = roc_curve(true_classes, predictions)

# Calculate the AUC (Area under the ROC Curve)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
model.save("chart_classification_model_BTC.h5")