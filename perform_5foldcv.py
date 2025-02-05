from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import glob
import scipy.stats as ss
import tensorflow as tf
from create_datasets import create_train_val_datasets
from create_train_val_list import oversample_minority
from build_conv_model import build_model_phase1, build_model_phase2, build_model_phase3, build_model_phase4
from fit_model import fit_model
import logging
from config import all_fold_results, param_final

logger = logging.getLogger("MainLogger")

## 5 Fold Cross Validation with best performing hyperparameters
def perform_5foldcv(train_paths, param_final, cvfolds, batch_size, oversampling_factor):

    # define number of folds to split the training dataset
    skf = StratifiedKFold(n_splits=cvfolds, shuffle=True, random_state=11)

    fold_scores_val = {"loss": [],
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
    train_labels = np.array(train_paths["label"])

    for i, (train_index, val_index) in enumerate(skf.split(range_train_labels, train_labels)):
        logger.info(f"Fold {i + 1}")

        # oversample minority class in the training set only
        train_paths_oversampled = oversample_minority(train_paths.iloc[train_index, :], oversampling_factor)

        file_paths_train = train_paths_oversampled["image_path"].to_numpy()
        labels_train = train_paths_oversampled["label"].to_numpy()
        metadata_train = train_paths_oversampled.iloc[:, 3:11].to_numpy()
        image_weight_train = train_paths_oversampled["image_weight"].to_numpy()
        features_train = train_paths_oversampled.iloc[:, 11:].to_numpy()

        ratio_minority_class = sum(labels_train) / file_paths_train.shape[0]
        logger.info(f"Minority ratio in oversampled training set: {ratio_minority_class}")

        file_paths_val = train_paths["image_path"][val_index].to_numpy()
        labels_val = train_paths["label"][val_index].to_numpy()
        image_weight_val = train_paths["image_weight"][val_index].to_numpy()
        metadata_val = train_paths.iloc[val_index, 3:11].to_numpy()
        features_val = train_paths.iloc[val_index, 11:].to_numpy()

        ratio_minority_class_val = sum(labels_val) / file_paths_val.shape[0]
        logger.info(f"Minority ratio in (non-oversampled) validation set: {ratio_minority_class_val}")

        # Check class distribution per fold
        prop_train = sum(labels_train) / len(labels_train)
        prop_val = sum(labels_val) / len(labels_val)
        print("Classes prop train: ", prop_train)
        print("Positive Class train: ", sum(labels_train))
        print("Classes prop val: ", prop_val)
        print("Positive Class val: ", sum(labels_val))

        print("creating training dataset for cv...")
        train_data, train_steps = create_train_val_datasets(file_paths=file_paths_train,
                                                            labels=labels_train,
                                                            image_weight=image_weight_train,
                                                            metadata=metadata_train,
                                                            features=features_train,
                                                            batch_size=batch_size,
                                                            num_epochs=params["num_epochs"],
                                                            pretrained_model=params["pretrained_model"],
                                                            img_size=params["img_size"],
                                                            num_channels=params["num_channels"],
                                                            crop_size=params["crop_size"],
                                                            training=True)

        logger.info("creating val dataset for cv...")
        val_data, val_steps = create_train_val_datasets(file_paths=file_paths_val,
                                                        labels=labels_val,
                                                        image_weight=image_weight_val,
                                                        metadata=metadata_val,
                                                        features=features_val,
                                                        batch_size=batch_size,
                                                        num_epochs=params["num_epochs"],
                                                        pretrained_model=params["pretrained_model"],
                                                        img_size=params["img_size"],
                                                        num_channels=params["num_channels"],
                                                        crop_size=params["crop_size"],
                                                        training=False)

        logger.info(f"Steps per epoch in train dataset: {len(labels_train) // batch_size}")
        logger.info(f"Steps per epoch in val dataset: {len(labels_val) // batch_size}")

        for img_met_feat, label in train_data.take(1):
            img, metadata, features = img_met_feat
            print(metadata)

        logger.info("Starting Phase 1 of Fine Tuning...")
        model, base_model = build_model_phase1(img_size=params["img_size"],
                                               num_channels=params["num_channels"],
                                               dropout_val=params["dropout_val"],
                                               num_dense_units=params["num_dense_units"],
                                               activation_dense=params["activation_dense"],
                                               l2_reg_dense=params["l2_reg_dense"],
                                               nodes_output=params["nodes_output"],
                                               activation_output=params["activation_output"],
                                               learning_rate=params["learning_rate"],
                                               alpha=params["alpha"],
                                               gamma=params["gamma"],
                                               num_metadata_features=metadata_train.shape[1],
                                               num_dense_units_metadata=params["num_dense_units_metadata"],
                                               num_dense_units_features=params["num_dense_units_features"],
                                               num_dense_units_combined=params["num_dense_units_combined"],
                                               pooling_type=params["pooling_type"],
                                               batch_norm=params["batch_norm"],
                                               pretrained_model=params["pretrained_model"])

        # initial_weights = model.get_weights()
        # model.set_weights(initial_weights) # reset model weights before model fitting

        model, history_phase1 = fit_model(model,
                                          train_dataset=train_data,
                                          steps_per_epoch=train_steps,
                                          validation_dataset=val_data,
                                          validation_steps=val_steps,
                                          num_epochs=params["num_epochs"],
                                          weight_positive=params["weight_positive"],
                                          callbacks=[],
                                          verbose=1)

        logger.info("Starting Phase 2 of Fine Tuning...")
        model = build_model_phase2(model,
                                   base_model,
                                   learning_rate=params["learning_rate"],
                                   alpha=params["alpha"],
                                   gamma=params["gamma"],
                                   decay_steps=train_steps,
                                   decay_rate=params["decay_rate"],
                                   lr_scaling_factor_phase2=params["lr_scaling_factor_phase2"],
                                   pretrained_model=params["pretrained_model"])

        model, history_phase2 = fit_model(model,
                                          train_dataset=train_data,
                                          steps_per_epoch=train_steps,
                                          validation_dataset=val_data,
                                          validation_steps=val_steps,
                                          num_epochs=params["num_epochs"] // 2,
                                          weight_positive=params["weight_positive"],
                                          callbacks=[early_stop],
                                          verbose=1)

        logger.info("Starting Phase 3 of Fine Tuning...")
        model = build_model_phase3(model,
                                   base_model,
                                   learning_rate=params["learning_rate"],
                                   alpha=params["alpha"],
                                   gamma=params["gamma"],
                                   decay_steps=train_steps,
                                   decay_rate=params["decay_rate"],
                                   lr_scaling_factor_phase3=params["lr_scaling_factor_phase3"],
                                   pretrained_model=params["pretrained_model"])

        model, history_phase3 = fit_model(model,
                                          train_dataset=train_data,
                                          steps_per_epoch=train_steps,
                                          validation_dataset=val_data,
                                          validation_steps=val_steps,
                                          num_epochs=params["num_epochs"] // 2,
                                          weight_positive=params["weight_positive"],
                                          callbacks=[early_stop],
                                          verbose=1)

        # Calculate scores on validation set (and on training set for comparison)
        logger.info(f"Calculating scores on validation set for fold {i + 1}:")

        val_loss, val_precision, val_recall, val_auc_pr, val_accuracy, val_auc = model.evaluate(val_data)
        val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())

        logger.info(f"Results on validation set for fold {i + 1}:")
        logger.info(f"loss: {val_loss}, precision: {val_precision}, recall: {val_recall}, auc pr: {val_auc_pr}, accuracy: {val_accuracy}, auc: {val_auc}")
        logger.info(f"F1 Score: {val_f1_score}")

        # Assign results to dictionary
        fold_scores_val["loss"].append(val_loss)
        fold_scores_val["f1_score"].append(val_f1_score)
        fold_scores_val["precision"].append(val_precision)
        fold_scores_val["recall"].append(val_recall)
        fold_scores_val["auc"].append(val_auc)

        fold_results = {"fold": i + 1, **fold_scores_val}
        pd.DataFrame([fold_results]).to_csv(f"./search_results/results_fold{i + 1}.csv", index=False)
        logger.info(f"Created csv file results_fold{i + 1}.csv with mean scores from {i + 1}")

    mean_scores["loss"] = np.mean(fold_scores_val["loss"])
    mean_scores["f1_score"] = np.mean(fold_scores_val["f1_score"])
    mean_scores["precision"] = np.mean(fold_scores_val["precision"])
    mean_scores["recall"] = np.mean(fold_scores_val["recall"])
    mean_scores["auc"] = np.mean(fold_scores_val["auc"])

    all_results = {**fold_scores_val, **mean_scores}
    all_results.to_csv(all_fold_results, index=False)

