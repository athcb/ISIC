#from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from io import BytesIO
import pandas as pd
import random
import os
import h5py
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import  make_scorer, log_loss, classification_report, precision_score, recall_score
from scipy.stats import loguniform
import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute import batch_sizes_for_worker

print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
from KerasWrapper import KerasWrapper
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

image_directory = "../ISIC_data/ISIC_2020_Training_JPEG/train"
metadata_directory = "../ISIC_data/ISIC_2020_Training_GroundTruth_v2.csv"

metadata = pd.read_csv(metadata_directory)
print(metadata.columns)
print(metadata.benign_malignant.value_counts(normalize=True))

metadata["image_path"] = image_directory + "/"  + metadata['image_name'] + '.jpg'
train_files, val_files = train_test_split(metadata[["image_path", "target"]], test_size = 0.2, shuffle=True, random_state = 10, stratify=metadata["target"])

num_samples = len(train_files)
print(f"Number of train images: {num_samples}")
print(f"Number of val images: {len(val_files)}")
pd.DataFrame({"image_path": train_files["image_path"], "label": train_files["target"]}).to_csv("train.csv", index=False)
pd.DataFrame({"image_path": val_files["image_path"], "label": val_files["target"]}).to_csv("val.csv", index=False)

print("Images per class in training set (ratios and total number):")
train_paths = pd.read_csv("train.csv")
print(train_paths.label.value_counts(normalize=True))
print(train_paths.label.value_counts())

print("Images per class in val set (ratios and total number):")
val_paths = pd.read_csv("val.csv")
print(val_paths.label.value_counts(normalize=True))
print(val_paths.label.value_counts())



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


def create_train_val_datasets(file_paths, labels, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image)
    if training == True:
        print("Starting augmentation...")
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print("Starting shuffling and batching...")
        # dataset = dataset.shuffle(1000).batch(batch_size)
        dataset = dataset.batch(batch_size)
    else:
        print("Starting batching...")
        dataset = dataset.batch(batch_size)
    return dataset



# print("Loading metadata...")
# metadata = load_metadata()
# print(f"Metadata for {metadata.shape[0]} images")

def check_class_distribution(dataset, dataset_name):
    labels = []
    for img, label in dataset.unbatch():
        labels.append(label.numpy())
    # print(labels)

    label_counts = Counter(labels)
    # print(label_counts)
    total_num = sum(label_counts.values())

    print(f"Class distribution in {dataset_name}:")
    for label, count in label_counts.items():
        print("Class ", label, " ratio: ", count / total_num)
    print("Total number of samples: ", total_num)


# check_class_distribution(train_dataset, "train dataset")
# check_class_distribution(val_dataset, "val dataset")

def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, "float"))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, "float"))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), "float"))

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1_score


import tensorflow as tf


def focal_loss(gamma_val, alpha_val):
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Compute cross entropy loss
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Compute focal loss term
        loss = alpha_val * tf.pow(1 - y_pred, gamma_val) * cross_entropy

        return tf.reduce_mean(loss)

    return focal_loss_fixed


def design_model_conv(img_size, num_channels,
                      num_filters,
                      filter_size, padding_type,
                      activation_fn, l2_reg_conv, stride_num,
                      pool_s, pool_stride, pool_padding,
                      dropout_val,
                      num_dense_units, activation_dense, l2_reg_dense,
                      nodes_output, activation_output,
                      learning_rate):
    # initialize model instance
    model = Sequential()
    # input layer
    model.add(layers.InputLayer(shape=(img_size, img_size, num_channels)))

    for i in range(1, len(num_filters) + 1):

        model.add(Conv2D(num_filters[i - 1], (filter_size, filter_size),
                         strides=stride_num,
                         padding=padding_type,
                         # activation= activation_fn,
                         kernel_regularizer=l2(l2_reg_conv)))
        model.add(BatchNormalization())
        model.add(Activation(activation_fn))

        if (i % 2) == 0:
            model.add(MaxPooling2D(pool_size=(pool_s, pool_s),
                                   strides=(pool_stride, pool_stride),
                                   padding=pool_padding))

    # model.add(MaxPooling2D(pool_size=(pool_s, pool_s),
    #                       strides=(pool_stride, pool_stride),
    #                       padding=pool_padding))
    # model.add(Flatten())
    model.add(GlobalMaxPooling2D())
    # model.add(Dropout(dropout_val))

    for j in range(len(num_dense_units)):
        model.add(Dense(num_dense_units[j],
                        # activation = activation_dense,
                        kernel_regularizer=l2(l2_reg_dense)))
        model.add(BatchNormalization())
        model.add(Activation(activation_dense))
        model.add(Dropout(dropout_val))

    # output layer
    model.add(Dense(nodes_output, activation=activation_output))

    print(model.summary())

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy",
                  # loss = focal_loss(gamma_val=0.5, alpha_val=0.75),
                  metrics=[f1_score, "precision", "recall", "AUC"],
                  optimizer=opt)

    return model

def save_randomized_search_results(grid_result, output_search_best_params):

    best_params_df = pd.DataFrame([grid_result.best_params_])
    best_score = grid_result.best_score_
    best_precision = grid_result.cv_results_['mean_test_precision'][grid_result.best_index_]  # Precision score
    best_recall = grid_result.cv_results_['mean_test_recall'][grid_result.best_index_]  # Recall score

    best_params_df["best_score"] = best_score
    best_params_df["best_precision"] = best_precision
    best_params_df["best_recall"] = best_recall

    print("Best params DF: ")
    print(best_params_df)
    best_params_df.to_csv(output_search_best_params, index=False)

def fit_model(model, train_dataset, validation_dataset, num_epochs, weight_positive, callbacks, verbose):

    class_weight = {0: 1.0, 1: weight_positive}

    history = model.fit(train_dataset,
                        steps_per_epoch=train_dataset.cardinality().numpy(),  #inefficient?
                        #steps_per_epoch=len(train_dataset),
                        epochs=num_epochs,
                        validation_data=validation_dataset,
                        validation_steps=validation_dataset.cardinality().numpy(),
                        #validation_steps= len(validation_dataset),
                        class_weight=class_weight,
                        callbacks = callbacks,
                        verbose=verbose)

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


def evaluate_model(model, train_dataset, test_dataset):

    test_loss, test_f1_score, test_precision, test_recall, test_auc = model.evaluate(test_dataset)
    #train_loss, train_accuracy, train_auc = model.evaluate(train_dataset)
    print("test_Loss ", test_loss)

    test_y_pred = model.predict(test_dataset)
    #train_y_pred = model.predict(train_dataset)

    return test_loss, test_f1_score, test_precision, test_recall, test_auc,  test_y_pred



def create_history_plots(input_history_vals):
    history_vals = pd.read_csv(input_history_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(history_vals.epoch, history_vals.loss, label="train")
    axes[0].plot(history_vals.epoch, history_vals.val_loss, label="validation")
    # axes[0].plot(history_vals.epoch, [baseline_loss] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    # axes[0].plot(history_vals.epoch, [random_loss] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    # axes[0].plot(history_vals.epoch, [test_loss] * len(history_vals.epoch), label="test", color="green")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Train & Validation Loss")
    axes[0].legend()
    # Plot the MAE on the training set and validation set
    axes[1].plot(history_vals.epoch, history_vals.f1_score, label="train")
    axes[1].plot(history_vals.epoch, history_vals.val_f1_score, label="validation")
    # axes[1].plot(history_vals.epoch, [baseline_accuracy] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    # axes[1].plot(history_vals.epoch, [random_accuracy] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    # axes[1].plot(history_vals.epoch, [test_accuracy] * len(history_vals.epoch), label="test", color="green")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("f1 score")
    axes[1].set_title("Train & Validation F1-Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()



param_grid = {"img_size": [224],
              "num_channels": [3],
              "num_filters": [[8, 16, 32], [32, 64, 128]],  # number of filter in the conv layers (length of list also defines number of conv layers)
              "filter_size": [3],
              "padding_type": ["same"],
              "activation_fn": ["relu"],
              "l2_reg_conv": loguniform(1e-4, 1e-3), #L2 regularization in the convolutional layers
              "stride_num": [1],
              "pool_s": [2], # filter size for pooling layer
              "pool_stride": [2], # strides for pooling layer
              "pool_padding": ["valid"], # padding type in pooling layer (valid: no padding, same: pad to keep same image dimensions)
              "dropout_val": [0.3, 0.4, 0.5], # dropout value for fully connected dense layers
              "num_dense_units": [ [64, 32], [128, 64], [32, 8] ], # number of units in the FC layers after the conv layers (length of list also defines number of FC layers)
              "activation_dense": ["relu"], # activation function in the FC layers
              "l2_reg_dense": loguniform(1e-4, 1e-3), #L2 regularization in the FC layers
              "nodes_output": [1], # number of nodes in the output layer (1 for binary classification, else number of classes)
              "activation_output": ["sigmoid"], # activation function for the output layer (sigmoid for binary classification, softmax for multi-classification)
              "learning_rate": loguniform(1e-4, 1e-3), # learning rate for gradient descent
              "num_epochs": [10], # number of training epochs
              "weight_positive": [20, 30, 40], # weight for the minority (positive) class in case of imbalanced datasets
              }

def custom_randomised_search(train_dataset, param_grid, num_iter, cvfolds, batch_size, train_paths):

    # define number of folds to split the training dataset
    skf = StratifiedKFold(n_splits = cvfolds, shuffle = True, random_state = 11)


    best_score = float("inf")
    best_model = None
    val_scores_best_model = {}
    mean_scores_best_model = {}
    best_params = {}


    for _ in range(num_iter):
        params = {key: np.random.choice(values) for key, values in param_grid.items()}
        print("Params in testing: ", params)

        fold_scores_val = {"loss": [],
                           "f1_score": [],
                           "precision": [],
                           "recall": [],
                           "auc": []}

        fold_scores_train = {"loss": [],
                           "f1_score": [],
                           "precision": [],
                           "recall": [],
                           "auc": []}

        mean_scores = {"loss": None,
                        "f1_score": None,
                        "precision": None,
                        "recall": None,
                        "auc": None}

        range_train_labels = np.arange(len(train_paths))
        train_labels = np.array(train_paths.labels)

        for i, (train_index, val_index) in enumerate(skf.split(range_train_labels, train_labels)):
            print(f"Fold {i} for params: {params}")

            train_index_tensor = tf.constant(train_index)
            val_index_tensor = tf.constant(val_index)

            train_data = train_dataset.enumerate().filter(lambda idx, _: tf.reduce_any(tf.equal((idx, train_index_tensor)))).map(lambda _, data: data)
            val_data = train_dataset.enumerate().filter(lambda idx, _: tf.reduce_any(tf.equal((idx, val_index_tensor)))).map(lambda _, data: data)

            train_data = train_data.batch(batch_size)
            val_data   = val_data.batch(batch_size)

            early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

            model = design_model_conv(img_size = params["img_size"],
                                  num_channels = params["num_channels"],
                                  num_filters = params["num_filters"],
                                  filter_size = params["filter_size"],
                                  padding_type = params["padding_type"],
                                  activation_fn = params["activation_fn"],
                                  l2_reg_conv = params["l2_reg_conv"],
                                  stride_num = params["stride_num"],
                                  pool_s = params["pool_s"],
                                  pool_stride = params["pool_stride"],
                                  pool_padding = params["pool_padding"],
                                  dropout_val = params["dropout_val"],
                                  num_dense_units = params["num_dense_units"],
                                  activation_dense = params["activation_dense"],
                                  l2_reg_dense = params["l2_reg_dense"],
                                  nodes_output = params["nodes_output"],
                                  activation_output = params["activation_output"],
                                  learning_rate = params["learning_rate"])
            # question: the train_dataset was already batched. Is the new train_data also automatically batched? in the fit_model function the steps per epoch and validation steps are computed from the cardinality of the dataset. is that still correct?
            model, history = fit_model(model, train_data, val_data, params["num_epochs"], params["weight_positive"], callbacks = [early_stop], verbose=1)

            # Calculate scores on validation set (and on training set for comparison)
            train_loss, train_f1_score, train_precision, train_recall, train_auc = model.evaluate(train_data)
            val_loss, val_f1_score, val_precision, val_recall, val_auc = model.evaluate(val_data)

            print(f"Results on validation set for fold {i}:" )
            print("Loss: ", val_loss, "F1 score: ", val_f1_score, "Precision: ", val_precision, "Recall: ", val_recall, "AUC: ", val_auc)

            print(f"Results on training set for fold {i}:")
            print("Loss: ", train_loss, "F1 score: ", train_f1_score, "Precision: ", train_precision, "Recall: ", train_recall,"AUC: ", train_auc)

            # Assign results to dictionary
            fold_scores_val["loss"].append(val_loss)
            fold_scores_val["f1_score"].append(val_f1_score)
            fold_scores_val["precision"].append(val_precision)
            fold_scores_val["recall"].append(val_recall)
            fold_scores_val["auc"].append(val_auc)

            fold_scores_train["loss"].append(train_loss)
            fold_scores_train["f1_score"].append(train_f1_score)
            fold_scores_train["precision"].append(train_precision)
            fold_scores_train["recall"].append(train_recall)
            fold_scores_train["auc"].append(train_auc)

        mean_score = np.mean(fold_scores_val["loss"])
        mean_f1_score = np.mean(fold_scores_val["f1_score"])
        mean_precision = np.mean(fold_scores_val["precision"])
        mean_recall = np.mean(fold_scores_val["recall"])
        mean_auc = np.mean(fold_scores_val["auc"])

        mean_scores["loss"] = mean_score
        mean_scores["f1_score"] = mean_f1_score
        mean_scores["precision"] = mean_precision
        mean_scores["recall"] = mean_recall
        mean_scores["auc"] = mean_auc

        # keep best model based on validation score (loss)
        if mean_score < best_score:

            best_score = mean_score
            best_model = model
            best_params = params

            mean_scores_best_model = mean_scores
            val_scores_best_model = fold_scores_val
            train_scores_best_model = fold_scores_train

    print(f"Best params found: {params}")
    return best_model,  best_params, mean_scores_best_model, val_scores_best_model, train_scores_best_model


train_dataset = create_train_val_datasets(train_paths.image_path.tolist(),
                                          train_paths.label.tolist(),
                                          batch_size=16,
                                          training=True)
val_dataset = create_train_val_datasets(val_paths.image_path.tolist(),
                                        val_paths.label.tolist(),
                                        batch_size=16,
                                        training=False)

print(len(train_dataset))
print(train_dataset.cardinality.numpy())


output_search_best_params = "./search_best_params.csv"
if not os.path.exists(output_search_best_params):
    print("Starting randomised search for hyperparameter tuning..")
    best_model,  best_params, val_scores_best_model, train_scores_best_model = custom_randomised_search(train_dataset,
                                    param_grid,
                                    num_iter=20,
                                    cvfolds=3,
                                    custom_scoring=log_loss,
                                    greaterisbetter_param=False)
    save_randomized_search_results(grid_result, output_search_best_params)
    print(f"Saved randomised search results to {output_search_best_params}")

'''
model = design_model_conv(img_size=224, 
                          num_channels=3,
                          num_conv_layers=3,
                          num_filters=[16, 32, 64],
                          filter_size=3,
                          padding_type="same",
                          activation_fn="relu",
                          # l2_reg_conv=0.001,
                          l2_reg_conv=0.0,
                          stride_num=1,
                          pool_s=2,
                          pool_stride=2,
                          pool_padding="valid",
                          dropout_val=0.3,
                          num_dense_layers=2,
                          num_dense_units=[64, 32],
                          activation_dense="relu",
                          l2_reg_dense=0.005,
                          nodes_output=1,
                          activation_output="sigmoid",
                          learning_rate=0.005)'''

#model, history = fit_model(model, train_dataset, val_dataset, num_epochs=15, weight_positive = 10., verbose=1)
#output_training_history = "training_history.csv"
#save_training_history(history, output_training_history)
#create_history_plots(output_training_history)
