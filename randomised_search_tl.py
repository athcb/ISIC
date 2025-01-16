from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import glob
import scipy.stats as ss
import tensorflow as tf
from create_datasets import create_train_val_datasets
from design_model_tl import design_model_transfer_phase1, design_model_transfer_phase2
from fit_model import fit_model
import logging

logger = logging.getLogger(__name__)

## Custom Randomized Search Function with Transfer Learning
def randomised_search_tl(train_paths, param_grid_tl, num_iter, cvfolds, batch_size):

    # define number of folds to split the training dataset
    skf = StratifiedKFold(n_splits = cvfolds, shuffle = True, random_state = 11)


    best_score = float("-inf")
    best_model = None
    val_scores_best_model = {}
    mean_scores_best_model = {}
    best_params = {}


    for itr in range(num_iter):

        logger.info(f"--------------ITERATION # {itr+1}---------------")

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

        params = {}

        for key, values in param_grid_tl.items():
            if isinstance(values, ss._distn_infrastructure.rv_continuous_frozen):
                params[key] = values.rvs() # random sample from distribution
            elif isinstance(values[0], list):
                params[key] = values[np.random.randint(len(values))] # randomly select a list from within the list
            else:
                params[key] = np.random.choice(values)  # randomly select a value from the list

        #early_stop = EarlyStopping(monitor="val_f1_score", mode="max", verbose=1, patience=5)

        range_train_labels = np.arange(len(train_paths))
        train_labels = np.array(train_paths["label"])

        for i, (train_index, val_index) in enumerate(skf.split(range_train_labels, train_labels)):
            logger.info(f"Fold {i+1} in iteration {itr+1} for params set: {params}")

            file_paths_train = train_paths["image_path"][train_index].to_numpy()
            labels_train = train_paths["label"][train_index].to_numpy()

            file_paths_val = train_paths["image_path"][val_index].to_numpy()
            labels_val = train_paths["label"][val_index].to_numpy()

            # Check class distribution per fold
            prop_train = sum(labels_train) / len(labels_train)
            prop_val = sum(labels_val) / len(labels_val)
            logger.info("Classes prop train: ", prop_train)
            logger.info("Positive Class train: ", sum(labels_train))
            logger.info("Classes prop val: ", prop_val)
            logger.info("Positive Class val: ", sum(labels_val))

            logger.info("creating training dataset for cv...")
            train_data, train_steps = create_train_val_datasets(file_paths_train, labels_train, batch_size, params["num_epochs"], training=True)
            logger.info("creating val dataset for cv...")
            val_data, val_steps   = create_train_val_datasets(file_paths_val,   labels_val,   batch_size, params["num_epochs"], training=False)

            logger.info("Steps per epoch train: ", len(labels_train) // batch_size)
            logger.info("val steps: ", len(labels_val) // batch_size)

            logger.info("Starting Phase 1 of Fine Tuning...")
            model, base_model = design_model_transfer_phase1(img_size=params["img_size"],
                                                 num_channels=params["num_channels"],
                                                 dropout_val=params["dropout_val"],
                                                 num_dense_units=params["num_dense_units"],
                                                 activation_dense=params["activation_dense"],
                                                 l2_reg_dense=params["l2_reg_dense"],
                                                 nodes_output=params["nodes_output"],
                                                 activation_output=params["activation_output"],
                                                 learning_rate=params["learning_rate"],
                                                 alpha=params["alpha"],
                                                 gamma=params["gamma"])

            #initial_weights = model.get_weights()

            #model.set_weights(initial_weights) # reset model weights before model fitting


            model, history = fit_model(model,
                                       train_dataset = train_data,
                                       steps_per_epoch = train_steps,
                                       validation_dataset = val_data,
                                       validation_steps = val_steps,
                                       num_epochs = params["num_epochs"],
                                       weight_positive =params["weight_positive"],
                                       #callbacks = [early_stop],
                                       callbacks=[],
                                       verbose=1)

            logger.info("Starting Phase 2 of Fine Tuning...")
            model = design_model_transfer_phase2(model,
                                                 base_model,
                                                 learning_rate=params["learning_rate"]*0.5,
                                                 alpha=params["alpha"],
                                                 gamma=params["gamma"])

            model, history = fit_model(model,
                                       train_dataset=train_data,
                                       steps_per_epoch=train_steps,
                                       validation_dataset=val_data,
                                       validation_steps=val_steps,
                                       num_epochs= params["num_epochs"] // 2,
                                       weight_positive=params["weight_positive"],
                                       # callbacks = [early_stop],
                                       callbacks=[],
                                       verbose=1)

            # Calculate scores on validation set (and on training set for comparison)
            logger.info(f"Calculating scores on validation set for fold {i+1}, iteration {itr+1}:")
            val_loss, val_precision, val_recall, val_auc = model.evaluate(val_data)
            val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + tf.keras.backend.epsilon())

            logger.info(f"Results on validation set for fold {i+1}, iteration {itr+1}:" )
            logger.info("Loss: ", val_loss, "F1 score: ", val_f1_score, "Precision: ", val_precision, "Recall: ", val_recall, "AUC: ", val_auc)

            # Assign results to dictionary
            fold_scores_val["loss"].append(val_loss)
            fold_scores_val["f1_score"].append(val_f1_score)
            fold_scores_val["precision"].append(val_precision)
            fold_scores_val["recall"].append(val_recall)
            fold_scores_val["auc"].append(val_auc)

        mean_scores["loss"] = np.mean(fold_scores_val["loss"])
        mean_scores["f1_score"]  = np.mean(fold_scores_val["f1_score"])
        mean_scores["precision"] = np.mean(fold_scores_val["precision"])
        mean_scores["recall"] = np.mean(fold_scores_val["recall"])
        mean_scores["auc"] = np.mean(fold_scores_val["auc"])

        itr_results = {"itr": itr+1, **mean_scores, **params}
        pd.DataFrame([itr_results]).to_csv(f"results_iter{itr+1}.csv", index=False)
        logger.info(f"Created csv file results_iter{itr+1}.csv with mean scores from {itr+1}")

        # keep best model based on validation score (loss)
        if mean_scores["f1_score"] > best_score:

            best_score = mean_scores["f1_score"]
            best_model = model
            best_params = params

            mean_scores_best_model = mean_scores
            val_scores_best_model = fold_scores_val

    # combine all itr files into one:
    all_itr = glob.glob("results_*.csv")
    all_results = pd.concat([pd.read_csv(file) for file in all_itr], ignore_index=True)
    all_results.to_csv("all_itr_results.csv", index = False)

    logger.info(f"Best params found: {params}")
    return best_model,  best_params, mean_scores_best_model, val_scores_best_model
