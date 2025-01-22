import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras import layers, initializers, Input
from keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, BatchNormalization, Activation, Concatenate
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.regularizers import l2
from keras.applications import ResNet50
from create_datasets import create_dataset_simclr
from create_train_val_list import create_file_paths_all
from config import metadata_path, image_directory


def build_simclr_model(img_size, num_channels):
    def encoder():
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)
        representations = layers.Dense(128)(x)  # Embedding output
        return Model(inputs=base_model.input, outputs=representations)

    encoder_model = encoder()
    input1 = layers.Input(shape=(img_size, img_size, num_channels), name="input1")
    input2 = layers.Input(shape=(img_size, img_size, num_channels), name="input2")

    rep1 = encoder_model(input1)
    rep2 = encoder_model(input2)
    print(rep1.shape)
    print(rep2.shape)

    model = Model(inputs={"input1": input1, "input2": input2}, outputs=[rep1, rep2])

    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=contrastive_loss)
    return model


def contrastive_loss(y_true, y_pred, temperature=0.1):
    # Normalize the embeddings (optional)
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)

    # Compute cosine similarity (dot product)
    dot_product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1)

    # Apply temperature scaling
    similarity = dot_product / temperature

    # Return the mean similarity loss
    return -tf.reduce_mean(similarity)


def extract_features(model, dataset):
    features = []
    for img_batch in dataset:
        batch_features = model.predict(img_batch)
        features.append(batch_features)
    return np.concatenate(features, axis=0)


file_paths_all = create_file_paths_all(metadata_path, image_directory)
print(file_paths_all.head())
simclr_dataset = create_dataset_simclr(file_paths_all, 32)
print(len(simclr_dataset))

# Get the first batch from the dataset


for inputs, targets in simclr_dataset.take(1):
    print(inputs)
    print(inputs["input1"].shape)
    # print(inputs["input2"].shape)

    # print(inputs["input1"])
    # print(inputs["input2"])

# print(img1, img2)


simclr_model = build_simclr_model(224, 3)
simclr_model.fit(simclr_dataset, epochs=10, steps_per_epoch=len(simclr_dataset))
features = extract_features(simclr_model, simclr_dataset)

