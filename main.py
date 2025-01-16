## Standard libraries
#from PIL import Image
import argparse

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from io import BytesIO
import pandas as pd
import random
import h5py
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy.stats as ss
from scipy.stats import loguniform
import os
import logging

## Import third-party libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras import mixed_precision

## My modules
from create_train_val_list import create_train_val_list
from create_datasets import create_train_val_datasets
from randomised_search import randomised_search
from randomised_search_tl import randomised_search_tl
from save_randomised_search_results import save_randomised_search_results
from train_tl import train_model
from save_training_results import save_training_results
from create_history_plots import create_history_plots
from config import param_grid_tl, param_grid, image_directory, metadata_path, output_best_params, output_mean_scores, output_val_scores, output_model, output_training_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="ISIC skin cancer classification model")
    parser.add_argument("--batch_size", type=int, default = 32, help = "batch size for random search and training")
    parser.add_argument("--num_iter", type=int, default = 5, help="number of iterations for random search")
    parser.add_argument("--cvfolds", type=int, default=3, help="number of folds for cross-validation in random search")
    return parser.parse_args()

def main():
    logger.info(f"Tensorflow version {tf.__version__}")
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    tf.random.set_seed(11)

    args = parse_args()
    batch_size = args.batch_size
    num_iter = args.num_iter
    cvfolds = args.cvfolds

    # create dataframes with the local paths to training and validation images and labels
    train_paths, val_paths = create_train_val_list(metadata_path, image_directory)

    # run randomised search with (stratified) kfold cross validation on the training dataset if the file with the randomised search results does not exist
    if not os.path.exists(output_best_params):
        logger.info("Starting randomised search for hyperparameter tuning..")
        best_model, best_params, mean_scores_best_model, val_scores_best_model = randomised_search_tl(train_paths,
                                                                                                      param_grid_tl,
                                                                                                      num_iter= num_iter,
                                                                                                      cvfolds= cvfolds,
                                                                                                      batch_size=batch_size)

        save_randomised_search_results(best_model,  best_params, mean_scores_best_model, val_scores_best_model,
                                       output_best_params, output_mean_scores, output_val_scores)
        logger.info(f"Saved randomised search results to {output_best_params}")

    # train model with parameters from randomised search
    logger.info("Starting model training...")
    model, history = train_model(train_paths, val_paths, output_best_params, batch_size = batch_size)

    # save model and training history
    logger.info("Saving model and training history...")
    save_training_results(history, model, output_training_history, output_model)

    # create plots with loss and custom metrics
    create_history_plots(output_training_history)


if __name__ == "__main__":
    main()

"""
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
                          learning_rate=0.005)"""


