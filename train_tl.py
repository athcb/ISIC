import pandas as pd
import logging
## Import Tensorflow libraries
from tensorflow.keras.callbacks import EarlyStopping

from create_datasets import create_train_val_datasets
from design_model_tl import design_model_transfer_phase1, design_model_transfer_phase2
from fit_model import fit_model

logger = logging.getLogger("MainLogger")

# TODO attention mechanisms, multi-scale learning, exponential decay


def train_model(train_paths, val_paths, output_best_params, batch_size):
    # load best parameters values
    best_params = pd.read_csv(output_best_params)

    logger.info("creating training dataset with best parameters from randomised search...")
    train_dataset, train_steps = create_train_val_datasets(file_paths=train_paths["image_path"].to_numpy(),
                                              # access the paths to training images from train_paths df
                                              labels=train_paths["label"].to_numpy(),
                                              # access the paths to labels of training images from train_paths df
                                              batch_size= batch_size,
                                              num_epochs = best_params["num_epochs"][0],
                                              training=True  # set to True for training set
                                              )

    val_dataset, val_steps = create_train_val_datasets(file_paths=val_paths["image_path"].to_numpy(),
                                            # access the paths to validation images from val_paths df
                                            labels=val_paths["label"][0],
                                            # access the paths to labels of validation images from val_paths df
                                            batch_size= batch_size,
                                            num_epochs=best_params["num_epochs"][0],
                                            training=False  # set to False for validation set
                                            )
    logger.info("Fine Tuning Phase 1: Training the dense layers")
    model, base_model =  design_model_transfer_phase1(best_params["img_size"][0],
                          best_params["num_channels"][0],
                          best_params["dropout_val"][0],
                          best_params["num_dense_units"][0],
                          best_params["activation_dense"][0],
                          best_params["l2_reg_dense"][0],
                          best_params["nodes_output"][0],
                          best_params["activation_output"][0],
                          best_params["learning_rate"][0],
                          best_params["alpha"][0],
                          best_params["gamma"][0])

    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

    model, history = fit_model(model,
                               train_dataset=train_dataset,
                               steps_per_epoch=train_steps,
                               validation_dataset=val_dataset,
                               validation_steps=val_steps,
                               num_epochs=best_params["num_epochs"][0],
                               weight_positive=best_params["weight_positive"][0],
                               callbacks = [early_stop],
                               verbose=1)

    logger.info("Fine Tuning Phase 2: Training the last 3 Convolutional layers")

    model =  design_model_transfer_phase2(model,
                                          base_model,
                                          best_params["learning_rate"].to_numpy() * 0.5,
                                          best_params["alpha"].to_numpy(),
                                          best_params["gamma"].to_numpy())

    model, history = fit_model(model,
                               train_dataset=train_dataset,
                               steps_per_epoch=train_steps,
                               validation_dataset=val_dataset,
                               validation_steps=val_steps,
                               num_epochs= best_params["num_epochs"][0],
                               weight_positive = best_params["weight_positive"][0],
                               callbacks = [early_stop],
                               verbose=1)

    #val_loss, val_precision, val_recall, val_auc = model.evaluate(val_dataset)

    return model, history
