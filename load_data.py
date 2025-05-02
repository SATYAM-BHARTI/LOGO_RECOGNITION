# load_data.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, image_size=(224, 224), batch_size=32, max_classes=None):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Train generator
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation generator
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    class_names = list(train_gen.class_indices.keys())
    if max_classes:
        class_names = class_names[:max_classes]

    return train_gen, val_gen, class_names
