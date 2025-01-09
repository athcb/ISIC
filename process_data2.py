from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from io import BytesIO
import pandas as pd
import random
import os

import h5py
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.src.layers import MaxPooling2D

image_directory = "../ISIC_data/ISIC_2020_Training_JPEG/train"
metadata_directory = "../ISIC_data/ISIC_2020_Training_GroundTruth_v2.csv"

metadata = pd.read_csv(metadata_directory)
metadata["image_path"] = '../ISIC_data/ISIC_2020_Training_JPEG/train/' + metadata['image_name'] + '.jpg'
train_files, val_files = train_test_split(metadata[["image_path", "target"]], test_size = 0.2, shuffle=True, random_state = 10, stratify=metadata["target"])

print(f"Number of train images: {len(train_files)}")
print(f"Number of val images: {len(val_files)}")
pd.DataFrame({"image_path": train_files["image_path"], "label": train_files["target"]}).to_csv("train.csv", index=False)
pd.DataFrame({"image_path": val_files["image_path"], "label": val_files["target"]}).to_csv("val.csv", index=False)

train_paths = pd.read_csv("train.csv")
print(train_paths.label.value_counts(normalize=True))

val_paths = pd.read_csv("val.csv")
print(val_paths.label.value_counts(normalize=True))



def load_metadata():
    """ Load training labels from csv with image metadata """
    metadata = pd.read_csv(metadata_directory)

    # Replace NAs in Sex columns with a random choice between male and female (only 65 out of 33127, should not bias model)
    metadata.sex = metadata.sex.apply(lambda x: random.choice(["male", "female"]) if pd.isna(x) else x)
    metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1})

    mean_age = round(metadata["age_approx"].mean())
    metadata.fillna({"age_approx": mean_age}, inplace=True)
    metadata["image_path"] = '../ISIC_data/ISIC_2020_Training_JPEG/train/' + metadata['image_name'] + '.jpg'

    metadata["anatom_site_general_challenge"] = metadata["anatom_site_general_challenge"].replace(
        {"lower extremity": "lower_extremity",
         "upper extremity": "upper_extremity"})
    metadata = pd.get_dummies(metadata,
                              columns=["diagnosis", "anatom_site_general_challenge"],
                              prefix=["diagnosis", "site"],
                              drop_first=True)

    print(metadata.image_path.head())
    return metadata[["image_path", "age_approx", "sex", "target"]].head(1000)


def load_image(file_path, label):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # Normalize the pixel values
    img = img / 255.0
    return img, label

def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label


def create_train_val_datasets(file_paths, labels, batch_size, training = False):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image)
    if training == True:
        print("Starting augmentation...")
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print("Starting shuffling and batching...")
        dataset = dataset.shuffle(1000).batch(batch_size)
    else:
        print("Starting batching...")
        dataset = dataset.batch(batch_size)
    return dataset

train_dataset = create_train_val_datasets(train_paths.image_path.tolist(),
                                          train_paths.label.tolist(),
                                          batch_size = 32,
                                          training = True)
val_dataset = create_train_val_datasets(val_paths.image_path.tolist(),
                                          val_paths.label.tolist(),
                                          batch_size = 32,
                                          training = False)


#print("Loading metadata...")
#metadata = load_metadata()
#print(f"Metadata for {metadata.shape[0]} images")

def check_class_distribution(dataset, dataset_name):
    labels = []
    for img, label in dataset.unbatch():
        labels.append(label.numpy())
    # print(labels)

    label_counts = Counter(labels)
    #print(label_counts)
    total_num = sum(label_counts.values())

    print(f"Class distribution in {dataset_name}:")
    for label, count in label_counts.items():
        print("Class ", label, " ratio: ", count / total_num)
    print("Total number of samples: ", total_num)


#check_class_distribution(train_dataset, "train dataset")
#check_class_distribution(val_dataset, "val dataset")

def f1_score(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(y_pred)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, "float"))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, "float"))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), "float"))

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1_score


def design_model_conv(img_size, num_channels,
                      num_conv_layers, num_filters,
                      filter_size, padding_type,
                      activation_fn, l2_reg_conv, stride_num,
                      pool_s, pool_stride, pool_padding,
                      dropout_val, num_dense_layers,
                      num_dense_units, activation_dense, l2_reg_dense,
                      nodes_output, activation_output,
                      learning_rate):

    # initialize model instance
    model = Sequential()
    # input layer
    model.add(layers.InputLayer(shape=(img_size, img_size, num_channels)))

    for i in range(1, num_conv_layers+1):

        model.add(Conv2D(num_filters[i-1], (filter_size, filter_size),
                         strides = stride_num,
                         padding = padding_type,
                         activation= activation_fn,
                         kernel_regularizer=l2(l2_reg_conv)))

        if (i % 2) == 0:
            model.add(MaxPooling2D(pool_size=(pool_s, pool_s),
                                   strides=(pool_stride, pool_stride),
                                   padding=pool_padding))

    model.add(MaxPooling2D(pool_size=(pool_s, pool_s),
                           strides=(pool_stride, pool_stride),
                           padding=pool_padding))
    model.add(Flatten())
    model.add(Dropout(dropout_val))

    for j in range(num_dense_layers):
        model.add(Dense(num_dense_units[j], activation = activation_dense, kernel_regularizer=l2(l2_reg_dense)))

    # output layer
    model.add(Dense(nodes_output, activation=activation_output))

    print(model.summary())

    opt = Adam(learning_rate = learning_rate)
    model.compile(loss = "binary_crossentropy",
                  metrics =[f1_score, "precision", "recall", "AUC"],
                  optimizer = opt )

    return model

def fit_model(model, train_dataset, validation_dataset, num_epochs, verbose):

   class_weight = {0: 1.0, 1: 40}

   history = model.fit(train_dataset,
              steps_per_epoch = train_dataset.cardinality().numpy(),
              epochs = num_epochs,
              validation_data = validation_dataset,
              validation_steps = validation_dataset.cardinality().numpy(),
              class_weight = class_weight,
              verbose = verbose)

   return model, history

def save_training_history(history, output_training_history):
    if not os.path.exists(output_training_history):
        data = {"epoch": list(range(1, len(history.history["loss"]) + 1)),
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "precision": history.history["precision"],
                "val_precision": history.history["val_precision"],
                "recall": history.history["recall"],
                "val_recall": history.history["val_recall"],
                "f1_score": history.history["f1_score"],
                "val_f1_score": history.history["val_f1_score"],
                "AUC": history.history["AUC"],
                "val_AUC": history.history["val_AUC"]}

        history_vals = pd.DataFrame(data)
        history_vals.to_csv(output_training_history, index=False)

        print("Training history values: ")
        print(history_vals)

def create_history_plots(input_history_vals):

    history_vals = pd.read_csv(input_history_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(history_vals.epoch, history_vals.loss, label="train")
    axes[0].plot(history_vals.epoch, history_vals.val_loss, label="validation")
    #axes[0].plot(history_vals.epoch, [baseline_loss] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    #axes[0].plot(history_vals.epoch, [random_loss] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    #axes[0].plot(history_vals.epoch, [test_loss] * len(history_vals.epoch), label="test", color="green")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Train & Validation Loss")
    axes[0].legend()
    # Plot the MAE on the training set and validation set
    axes[1].plot(history_vals.epoch, history_vals.f1_score, label="train")
    axes[1].plot(history_vals.epoch, history_vals.val_f1_score, label="validation")
    #axes[1].plot(history_vals.epoch, [baseline_accuracy] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    #axes[1].plot(history_vals.epoch, [random_accuracy] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    #axes[1].plot(history_vals.epoch, [test_accuracy] * len(history_vals.epoch), label="test", color="green")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("f1 score")
    axes[1].set_title("Train & Validation F1-Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


model = design_model_conv(img_size=224, num_channels=3,
                          num_conv_layers=4, num_filters=[8, 16, 32, 64],
                          filter_size=3, padding_type="same",
                          activation_fn="relu", l2_reg_conv=0.001, stride_num=1,
                          pool_s=2, pool_stride=2, pool_padding="valid",
                          dropout_val=0.3, num_dense_layers=3,
                          num_dense_units=[128, 64, 32], activation_dense="relu",
                          l2_reg_dense=0.005,
                          nodes_output=1, activation_output="sigmoid",
                          learning_rate=0.001)

model, history = fit_model(model, train_dataset, val_dataset, num_epochs=2, verbose=1)
output_training_history = "training_history.csv"
save_training_history(history, output_training_history)
create_history_plots(output_training_history)
