import seaborn as sns

import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import cv2
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import h5py
import matplotlib.pyplot as plt


base_dir = "cell_images"

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "val")

train_parasitized_dir = os.path.join(train_dir, "Parasitized")
train_uninfected_dir = os.path.join(train_dir, "Uninfected")
test_parasitized_dir = os.path.join(test_dir, "Parasitized")
test_uninfected_dir = os.path.join(test_dir, "Uninfected")
val_parasitized_dir = os.path.join(val_dir, "Parasitized")
val_uninfected_dir = os.path.join(val_dir, "Uninfected")


os.makedirs(train_parasitized_dir, exist_ok=True)
os.makedirs(train_uninfected_dir, exist_ok=True)
os.makedirs(val_parasitized_dir, exist_ok=True)
os.makedirs(val_uninfected_dir, exist_ok=True)
os.makedirs(test_parasitized_dir, exist_ok=True)
os.makedirs(test_uninfected_dir, exist_ok=True)

parasitized_dir = 'cell_images/Parasitized'
uninfected_dir = 'cell_images/Uninfected'

parasitized_images = os.listdir(parasitized_dir)
uninfected_images = os.listdir(uninfected_dir)


parasitized_train, parasitized_temp = train_test_split(parasitized_images, test_size=0.2, random_state=42)
parasitized_test, parasitized_val = train_test_split(parasitized_temp, test_size=0.5, random_state=42)

uninfected_train, uninfected_temp = train_test_split(uninfected_images, test_size=0.2, random_state=42)
uninfected_test, uninfected_val = train_test_split(uninfected_temp, test_size=0.5, random_state=42)

def copy_files_with_prefix(file_list, source_dir, target_dir, prefix="_copy_"):
    for file in file_list:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, prefix + file)
        shutil.copy(src, dst)


copy_files_with_prefix(parasitized_train, parasitized_dir, train_parasitized_dir)
copy_files_with_prefix(parasitized_val, parasitized_dir, val_parasitized_dir)
copy_files_with_prefix(parasitized_test, parasitized_dir, test_parasitized_dir)

copy_files_with_prefix(uninfected_train, uninfected_dir, train_uninfected_dir)
copy_files_with_prefix(uninfected_val, uninfected_dir, val_uninfected_dir)
copy_files_with_prefix(uninfected_test, uninfected_dir, test_uninfected_dir)

def load_images_from_folder(folder, save_path=None):
    if save_path and os.path.exists(save_path):
        data = np.load(save_path)
        return data['images'], data['labels']

    images = []
    labels = []
    for label_folder in ["Parasitized", "Uninfected"]:
        path = os.path.join(folder, label_folder)  # Subfolder path
        label = 0 if label_folder == "Parasitized" else 1  # Labels: 0 = Parasitized, 1 = Uninfected
        for filename in os.listdir(path):  # Loop through each file in the folder
            img_path = os.path.join(path, filename)  # File path
            img = cv2.imread(img_path)  # Read the image
            if img is not None:
                img = cv2.resize(img, (100, 115))  # Resize images to fixed size
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    if save_path:
        np.savez(save_path, images=images, labels=labels)

    return images, labels


train_save_path = 'train_images_labels.npz'
val_save_path = 'val_images_labels.npz'
test_save_path = 'test_images_labels.npz'

train_images, train_labels = load_images_from_folder('cell_images/train', train_save_path)
val_images, val_labels = load_images_from_folder('cell_images/val', val_save_path)
test_images, test_labels = load_images_from_folder('cell_images/test', test_save_path)


mean_train = np.mean(train_images, axis=(0, 1, 2), keepdims=True)


train_images = train_images - mean_train
val_images = val_images - mean_train  
test_images = test_images - mean_train

from keras import layers, models

from tensorflow.keras import layers, models, regularizers

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(115,100, 3),
                  kernel_regularizer=regularizers.l2(0.001),
                  kernel_initializer=("glorot_uniform")),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.30),

    layers.Conv2D(64, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001),
                  kernel_initializer=("glorot_uniform")),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.35),

    layers.Conv2D(128, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001),
                  kernel_initializer=("glorot_uniform")),

    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.45),

    layers.Flatten(),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001),
                 kernel_initializer=("glorot_uniform")),
    layers.Dropout(0.45),
    layers.Dense(1, activation='sigmoid')
])

model.summary()


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15)


history = model.fit(
    train_images, train_labels,
    epochs=200,
    validation_data=(val_images, val_labels),
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test doğruluğu: {test_acc}")


