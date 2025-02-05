import pandas as pd
import logging
import ast
## Import Tensorflow libraries
from tensorflow.keras.callbacks import EarlyStopping

from create_datasets import create_train_val_datasets
from build_conv_model import build_model_phase4,build_model_phase3, build_model_phase1, build_model_phase2
from fit_model import fit_model
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

logger = logging.getLogger("MainLogger")


def train_model(train_paths, val_paths, output_best_params, batch_size):
    # load best parameters values
    best_params = pd.read_csv(output_best_params)
    params = best_params.iloc[0]
    img_size = int(params["img_size"])
    num_channels = int(params["num_channels"])
    dropout_val = 0.3  # float(params["dropout_val"])
    num_dense_units = ast.literal_eval(params["num_dense_units"])
    activation_dense = params["activation_dense"]
    l2_reg_dense = 0.00005  # float(params["l2_reg_dense"])
    nodes_output = int(params["nodes_output"])
    activation_output = params["activation_output"]
    learning_rate = 0.0001  # float(params["learning_rate"])
    # learning_rate = 0.0001 # float(params["learning_rate"])
    alpha = 0.5  # float(params["alpha"])
    gamma = 2.  # float(params["gamma"])
    num_epochs = 12  # int(params["num_epochs"])
    weight_positive = 1.  # float(params["weight_positive"])
    num_dense_units_metadata = ast.literal_eval(params["num_dense_units_metadata"])
    num_dense_units_combined = ast.literal_eval(params["num_dense_units_metadata"])
    num_dense_units_features = [128]
    pooling_type = "global_avg"  # params["pooling_type"]
    num_unfrozen_layers = int(params["num_unfrozen_layers"])
    decay_rate = float(params["decay_rate"])
    batch_norm = int(params["batch_norm"])
    lr_scaling_factor_phase2 = 0.5  # 0.1
    lr_scaling_factor_phase3 = 0.3
    lr_scaling_factor_phase4 = 0.1
    pretrained_model = "densenet121"
    crop_size = 200

    metadata_train = train_paths.iloc[:, 3:11].to_numpy()
    metadata_val = val_paths.iloc[:, 3:11].to_numpy()
    num_metadata_features = int(metadata_train.shape[1])

    features_train = train_paths.iloc[:, 11:].to_numpy()
    features_val = val_paths.iloc[:, 11:].to_numpy()

    logger.info(f"metadata number of features {metadata_train.shape[1]}")
    logger.info(f"simcl number of features {features_train.shape[1]}")

    logger.info("creating training dataset with best parameters from randomised search...")
    train_dataset, train_steps = create_train_val_datasets(file_paths=train_paths["image_path"].to_numpy(),
                                                           labels=train_paths["label"].to_numpy(),
                                                           image_weight=train_paths["image_weight"].to_numpy(),
                                                           metadata=metadata_train,
                                                           features=features_train,
                                                           batch_size=batch_size,
                                                           num_epochs=num_epochs,
                                                           pretrained_model=pretrained_model,
                                                           img_size=img_size,
                                                           num_channels=num_channels,
                                                           crop_size=crop_size,
                                                           training=True
                                                           )

    val_dataset, val_steps = create_train_val_datasets(file_paths=val_paths["image_path"].to_numpy(),
                                                       # access the paths to validation images from val_paths df
                                                       labels=val_paths["label"].to_numpy(),
                                                       image_weight=val_paths["image_weight"].to_numpy(),
                                                       metadata=metadata_val,
                                                       features=features_val,
                                                       # access the paths to labels of validation images from val_paths df
                                                       batch_size=batch_size,
                                                       num_epochs=num_epochs,
                                                       pretrained_model=pretrained_model,
                                                       img_size=img_size,
                                                       num_channels=num_channels,
                                                       crop_size=crop_size,
                                                       training=False)  # set to False for validation set)

    logger.info("Feature Extraction Phase 1: Training the dense layers")
    model, base_model = build_model_phase1(img_size=img_size,
                                           num_channels=num_channels,
                                           dropout_val=dropout_val,
                                           num_dense_units=num_dense_units,
                                           activation_dense=activation_dense,
                                           l2_reg_dense=l2_reg_dense,
                                           nodes_output=nodes_output,
                                           activation_output=activation_output,
                                           learning_rate=learning_rate,
                                           alpha=alpha,
                                           gamma=gamma,
                                           num_metadata_features=num_metadata_features,
                                           num_dense_units_metadata=num_dense_units_metadata,
                                           num_dense_units_features=num_dense_units_features,
                                           num_dense_units_combined=num_dense_units_combined,
                                           pooling_type=pooling_type,
                                           batch_norm=batch_norm,
                                           decay_steps=train_steps,
                                           decay_rate=decay_rate,
                                           pretrained_model=pretrained_model)

    early_stop = EarlyStopping(monitor="val_auc_pr", mode="max", verbose=1, patience=10)

    model, history_phase1 = fit_model(model,
                                      train_dataset=train_dataset,
                                      steps_per_epoch=train_steps,
                                      validation_dataset=val_dataset,
                                      validation_steps=val_steps,
                                      num_epochs=num_epochs,
                                      weight_positive=weight_positive,
                                      callbacks=[early_stop],
                                      verbose=1)

    val_loss, val_precision, val_recall, val_auc_pr, val_accuracy, val_auc = model.evaluate(val_dataset)
    logger.info(
        f"Val dataset overall results Phase 1: loss: {val_loss}, precision: {val_precision}, recall: {val_recall}, auc pr: {val_auc_pr}, accuracy: {val_accuracy}, auc: {val_auc}")
    logger.info(
        f"Phase 1 F1 Score: {2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())}")

    logger.info(f"Fine Tuning Phase 2:")

    model = build_model_phase2(model,
                               base_model,
                               learning_rate=learning_rate,
                               alpha=alpha,
                               gamma=gamma,
                               decay_steps=train_steps,
                               decay_rate=decay_rate,
                               lr_scaling_factor_phase2=lr_scaling_factor_phase2,
                               pretrained_model=pretrained_model)

    model, history_phase2 = fit_model(model,
                                      train_dataset=train_dataset,
                                      steps_per_epoch=train_steps,
                                      validation_dataset=val_dataset,
                                      validation_steps=val_steps,
                                      num_epochs=num_epochs // 2,
                                      weight_positive=weight_positive,
                                      callbacks=[early_stop],
                                      verbose=1)

    val_loss, val_precision, val_recall, val_auc_pr, val_accuracy, val_auc = model.evaluate(val_dataset)
    logger.info(
        f"Val dataset overall results Phase 2: loss: {val_loss}, precision: {val_precision}, recall: {val_recall}, auc pr: {val_auc_pr}, accuracy: {val_accuracy}, auc: {val_auc}")
    logger.info(
        f"Phase 2 F1 Score: {2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())}")

    logger.info(f"Fine Tuning Phase 3:")

    model = build_model_phase3(model,
                               base_model,
                               learning_rate=learning_rate,
                               alpha=alpha,
                               gamma=gamma,
                               decay_steps=train_steps,
                               decay_rate=decay_rate,
                               lr_scaling_factor_phase3=lr_scaling_factor_phase3,
                               pretrained_model=pretrained_model)

    model, history_phase3 = fit_model(model,
                                      train_dataset=train_dataset,
                                      steps_per_epoch=train_steps,
                                      validation_dataset=val_dataset,
                                      validation_steps=val_steps,
                                      num_epochs=num_epochs // 2,
                                      weight_positive=weight_positive,
                                      callbacks=[early_stop],
                                      verbose=1)

    val_loss, val_precision, val_recall, val_auc_pr, val_accuracy, val_auc = model.evaluate(val_dataset)

    logger.info(
        f"Val dataset overall results Phase 3: loss: {val_loss}, precision: {val_precision}, recall: {val_recall}, auc pr: {val_auc_pr}, accuracy: {val_accuracy}, auc: {val_auc}")
    logger.info(
        f"Phase 3 F1 Score: {2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())}")

    return model, history_phase1, history_phase2, history_phase3