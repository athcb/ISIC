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
    num_epochs = 15  # int(params["num_epochs"])
    weight_positive = float(params["weight_positive"])
    num_dense_units_metadata = ast.literal_eval(params["num_dense_units_metadata"])
    num_dense_units_combined = ast.literal_eval(params["num_dense_units_metadata"])
    num_dense_units_features = [128]
    pooling_type = params["pooling_type"]
    num_unfrozen_layers = int(params["num_unfrozen_layers"])
    decay_rate = float(params["decay_rate"])
    batch_norm = int(params["batch_norm"])
    lr_scaling_factor_phase2 = 1.

    metadata_train = train_paths.iloc[:, 2:10].to_numpy()
    metadata_val = val_paths.iloc[:, 2:10].to_numpy()
    num_metadata_features = int(metadata_train.shape[1])

    features_train = train_paths.iloc[:, 10:].to_numpy()
    features_val = val_paths.iloc[:, 10:].to_numpy()

    logger.info(f"metadata number of features {metadata_train.shape[1]}")
    logger.info(f"simcl number of features {features_train.shape[1]}")

    logger.info("creating training dataset with best parameters from randomised search...")
    train_dataset, train_steps = create_train_val_datasets(file_paths=train_paths["image_path"].to_numpy(),
                                                           # access the paths to training images from train_paths df
                                                           labels=train_paths["label"].to_numpy(),
                                                           metadata=metadata_train,
                                                           features=features_train,
                                                           # access the paths to labels of training images from train_paths df
                                                           batch_size=batch_size,
                                                           num_epochs=num_epochs,
                                                           training=True  # set to True for training set
                                                           )

    val_dataset, val_steps = create_train_val_datasets(file_paths=val_paths["image_path"].to_numpy(),
                                                       # access the paths to validation images from val_paths df
                                                       labels=val_paths["label"].to_numpy(),
                                                       metadata=metadata_val,
                                                       features=features_val,
                                                       # access the paths to labels of validation images from val_paths df
                                                       batch_size=batch_size,
                                                       num_epochs=num_epochs,
                                                       training=False  # set to False for validation set
                                                       )
    logger.info("Feature Extraction Phase 1: Training the dense layers")
    model, base_model = design_model_transfer_phase1(img_size=img_size,
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
                                                     batch_norm=batch_norm)

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

    val_loss, val_precision, val_recall, val_auc = model.evaluate(val_dataset)
    loss, precision, recall, auc = model.evaluate(train_dataset)

    logger.info(
        f"Train dataset overall results Phase 1: loss: {loss}, precision: {precision}, recall: {recall}, auc: {auc}")
    logger.info(
        f"Val dataset overall results Phase 1: loss: {val_loss}, precision: {val_precision}, recall: {val_recall}, auc: {val_auc}")
    logger.info(
        f"Phase 1 F1 Score: {2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())}")

    logger.info(f"Fine Tuning Phase 2: Training the last {num_unfrozen_layers} Convolutional layers")

    model2 = design_model_transfer_phase2(model,
                                          base_model,
                                          learning_rate=learning_rate,
                                          alpha=alpha,
                                          gamma=gamma,
                                          num_unfrozen_layers=num_unfrozen_layers,
                                          decay_steps=train_steps,
                                          decay_rate=decay_rate,
                                          lr_scaling_factor_phase2=lr_scaling_factor_phase2)

    model2, history_phase2 = fit_model(model,
                                       train_dataset=train_dataset,
                                       steps_per_epoch=train_steps,
                                       validation_dataset=val_dataset,
                                       validation_steps=val_steps,
                                       num_epochs=1,
                                       weight_positive=weight_positive,
                                       callbacks=[early_stop],
                                       verbose=1)

    val_loss, val_precision, val_recall, val_auc = model.evaluate(val_dataset)
    loss, precision, recall, auc = model.evaluate(train_dataset)

    logger.info(
        f"Train dataset overall results Phase 2: loss: {loss}, precision: {precision}, recall: {recall}, auc: {auc}")
    logger.info(
        f"Val dataset overall results Phase 2: loss: {val_loss}, precision: {val_precision}, recall: {val_recall}, auc: {val_auc}")
    logger.info(
        f"Phase 2 F1 Score: {2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())}")

    y_prob = []  # List to store the predicted probabilities for the positive class (class 1)
    y_true = []  # List to store the true labels

    # Iterate through the tf.data.Dataset to get predictions batch by batch
    for inputs, batch_labels in val_dataset:
        batch_data = inputs["input_image"]
        batch_metadata = inputs["input_metadata"]
        # Ensure the model is receiving both image and metadata inputs
        batch_probs = model.predict(inputs)  # Using list for multi-input model

        y_prob.extend(batch_probs)  # Assuming class 1 is the positive class (adjust as needed)
        y_true.extend(batch_labels.numpy())  # Store true labels

    # Convert y_true and y_prob to numpy arrays for further processing
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Now you can use y_prob and y_true to calculate precision, recall, F1 score, etc.
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # Loop through different thresholds and compute metrics
    for threshold in thresholds:
        # Predict classes based on the threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate precision, recall, and F1 score for the current threshold
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Store the results
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Find the threshold that gives the highest F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision_scores[np.argmax(f1_scores)]
    best_recall = recall_scores[np.argmax(f1_scores)]
    best_f1_score = max(f1_scores)

    print(f"Best Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1_score}")
    print(f"Best Precision Score: {best_precision}")
    print(f"Best Recall Score: {best_recall}")

    return model, history_phase1, history_phase2
