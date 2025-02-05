import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.utils import register_keras_serializable

from keras.models import Sequential, Model, load_model
from keras import layers, initializers, Input
from keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, BatchNormalization, Activation, Concatenate
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.regularizers import l2
from keras.applications import ResNet50
import os
import logging

logger = logging.getLogger("MainLogger")


def build_simclr_model(img_size, num_channels, learning_rate):
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

    loss_layer = ContrastiveLossLayer()([rep1, rep2])

    model = Model(inputs={"input1": input1, "input2": input2}, outputs=[loss_layer, rep1, rep2])

    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


@register_keras_serializable()
class ContrastiveLossLayer(keras.layers.Layer):
    def __init__(self, temperature=0.1, **kwargs):
        super(ContrastiveLossLayer, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        rep1, rep2 = inputs
        rep1 = tf.math.l2_normalize(rep1, axis=-1)
        rep2 = tf.math.l2_normalize(rep2, axis=-1)
        logits = tf.matmul(rep1, tf.transpose(rep2)) / self.temperature

        labels = tf.range(tf.shape(logits)[0])  # Assume a simple case where indices are correct
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        self.add_loss(loss)
        return loss  # Returning loss as output for better integration with compile()

    def compute_output_shape(self, input_shape):
        return ()  # Scalar output for loss

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


def extract_features(model, dataset, batch_size):
    features_all = []
    for inputs, dummy in dataset:
        input1 = inputs["input1"]
        input2 = inputs["input2"]
        _, features1, features2 = model.predict({"input1": input1, "input2": input2}, batch_size=batch_size, verbose=0)
        features = features1 + features2
        features_all.append(features)
    features_concat = np.concatenate(features_all, axis=0)

    logger.info(f"Length of predicted features list: {len(features_concat)}")
    return features_concat


def save_features(features, file_paths_simclr, features_output):
    features_df = pd.DataFrame(features, columns=[f"feature_{i + 1}" for i in range(features.shape[1])])
    combined_df = pd.concat([file_paths_simclr, features_df], axis=1)
    logger.info(f"Number of rows in features file {len(combined_df)}")
    combined_df.to_csv(features_output, index=False)
    logger.info(f"Saved predicted simclr feautures to {features_output}.")
    return combined_df


def save_simclr_training_history(history, simclr_history_output):
    metrics = history.history
    data = {"epoch": list(range(1, len(metrics["loss"]) + 1)),
            "loss": metrics["loss"]}

    metrics_df = pd.DataFrame(data)
    metrics_df.to_csv(simclr_history_output, index=False)

    logger.info(f"Simclr training saved to {simclr_history_output}.")


