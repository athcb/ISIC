import pandas as pd
import logging
import ast
## Import Tensorflow libraries
from tensorflow.keras.callbacks import EarlyStopping

from create_datasets import create_train_val_datasets
from design_model_tl import design_model_transfer_phase1, design_model_transfer_phase2
from fit_model import fit_model

logger = logging.getLogger("MainLogger")

# TODO attention mechanisms, multi-scale learning


def train_model(train_paths, val_paths, output_best_params, batch_size):
    # load best parameters values
    best_params = pd.read_csv(output_best_params)
    params = best_params.iloc[0]
    img_size = int(params["img_size"])
    num_channels = int(params["num_channels"])
    dropout_val = float(params["dropout_val"])
    num_dense_units = ast.literal_eval(params["num_dense_units"])
    activation_dense = params["activation_dense"]
    l2_reg_dense = float(params["l2_reg_dense"])
    nodes_output = int(params["nodes_output"])
    activation_output = params["activation_output"]
    learning_rate = float(params["learning_rate"])
    alpha = float(params["alpha"])
    gamma = float(params["gamma"])
    num_epochs = int(params["num_epochs"])
    weight_positive = float(params["weight_positive"])
    num_dense_units_metadata = int(params["num_dense_units_metadata"])
    num_unfrozen_layers = int(params["num_unfrozen_layers"])
    decay_rate = float(params["decay_rate"])

    metadata_train = train_paths.iloc[:, 2:].to_numpy()
    metadata_val = val_paths.iloc[:, 2:].to_numpy()
    num_metadata_features = int(metadata_train.shape[1])
    logger.info(f"metadata number of features {num_metadata_features}")
    logger.info(f"metadata number of features {metadata_train.head()}")

    logger.info("creating training dataset with best parameters from randomised search...")
    train_dataset, train_steps = create_train_val_datasets(file_paths=train_paths["image_path"].to_numpy(),
                                              # access the paths to training images from train_paths df
                                              labels=train_paths["label"].to_numpy(),
                                              metadata = metadata_train,
                                              # access the paths to labels of training images from train_paths df
                                              batch_size= batch_size,
                                              num_epochs = num_epochs,
                                              training=True  # set to True for training set
                                              )

    val_dataset, val_steps = create_train_val_datasets(file_paths=val_paths["image_path"].to_numpy(),
                                            # access the paths to validation images from val_paths df
                                            labels=val_paths["label"].to_numpy(),
                                            metadata=metadata_val,
                                            # access the paths to labels of validation images from val_paths df
                                            batch_size= batch_size,
                                            num_epochs= num_epochs,
                                            training=False  # set to False for validation set
                                            )
    logger.info("Fine Tuning Phase 1: Training the dense layers")
    model, base_model =  design_model_transfer_phase1(img_size = img_size,
                                                      num_channels = num_channels,
                                                      dropout_val = dropout_val,
                                                      num_dense_units = num_dense_units,
                                                      activation_dense = activation_dense,
                                                      l2_reg_dense = l2_reg_dense,
                                                      nodes_output = nodes_output,
                                                      activation_output = activation_output,
                                                      learning_rate = learning_rate,
                                                      alpha = alpha,
                                                      gamma = gamma,
                                                      num_metadata_features=num_metadata_features,
                                                      num_dense_units_metadata= num_dense_units_metadata)

    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

    model, history_phase1 = fit_model(model,
                               train_dataset=train_dataset,
                               steps_per_epoch=train_steps,
                               validation_dataset=val_dataset,
                               validation_steps=val_steps,
                               num_epochs=num_epochs,
                               weight_positive=weight_positive,
                               callbacks = [early_stop],
                               verbose=1)

    val_loss, val_precision, val_recall, val_auc = model.evaluate(val_dataset)
    loss, precision, recall, auc = model.evaluate(train_dataset)

    logger.info(f"Train dataset overall results Phase 1: {loss}, {precision}, {recall}, {auc}")
    logger.info(f"Val dataset overall results Phase 1: {val_loss}, {val_precision}, {val_recall}, {val_auc}")


    logger.info("Fine Tuning Phase 2: Training the last 3 Convolutional layers")

    model =  design_model_transfer_phase2(model,
                                          base_model,
                                          learning_rate = learning_rate,
                                          alpha = alpha,
                                          gamma = gamma,
                                          num_unfrozen_layers=num_unfrozen_layers,
                                          decay_steps= train_steps,
                                          decay_rate= decay_rate)

    model, history_phase2 = fit_model(model,
                               train_dataset=train_dataset,
                               steps_per_epoch=train_steps,
                               validation_dataset=val_dataset,
                               validation_steps=val_steps,
                               num_epochs= num_epochs,
                               weight_positive = weight_positive,
                               callbacks = [early_stop],
                               verbose=1)

    val_loss, val_precision, val_recall, val_auc = model.evaluate(val_dataset)
    loss, precision, recall, auc = model.evaluate(train_dataset)

    logger.info(f"Train dataset overall results Phase 2: {loss}, {precision}, {recall}, {auc}")
    logger.info(f"Val dataset overall results Phase 2: {val_loss}, {val_precision}, {val_recall}, {val_auc}")

    return model, history_phase1, history_phase2
