# ai/train_model.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure dataset path
DATASET_PATH = 'dataset/'

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 220, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output: 0 or 1
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image generator: auto rescale + validation split
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(150, 220),
    color_mode='grayscale',
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(150, 220),
    color_mode='grayscale',
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=15)

# Save the trained model
os.makedirs("model", exist_ok=True)
model.save("model/signature_cnn.h5")
print("✅ Model saved to model/signature_cnn.h5")
