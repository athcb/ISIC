import pandas as pd
## Import Tensorflow libraries
from tensorflow.keras.callbacks import EarlyStopping



def train_model(train_paths, val_paths, batch_size):
    # load best parameters values
    best_params = pd.read_csv(output_best_params)

    print("creating training dataset with best parameters from randomised search...")
    train_dataset, train_steps = create_train_val_datasets(file_paths=train_paths["image_path"].to_numpy(),
                                              # access the paths to training images from train_paths df
                                              labels=train_paths["label"].to_numpy(),
                                              # access the paths to labels of training images from train_paths df
                                              batch_size= batch_size,
                                              num_epochs = best_params["num_epochs"],
                                              training=True  # set to True for training set
                                              )

    val_dataset, val_steps = create_train_val_datasets(file_paths=val_paths["image_path"].to_numpy(),
                                            # access the paths to validation images from val_paths df
                                            labels=val_paths["label"].to_numpy(),
                                            # access the paths to labels of validation images from val_paths df
                                            batch_size= batch_size,
                                            num_epochs=best_params["num_epochs"],
                                            training=False  # set to False for validation set
                                            )

    model, base_model =  design_model_transfer_phase1(best_params["img_size"],
                          best_params["num_channels"],
                          best_params["dropout_val"],
                          best_params["num_dense_units"],
                          best_params["activation_dense"],
                          best_params["l2_reg_dense"],
                          best_params["nodes_output"],
                          best_params["activation_output"],
                          best_params["learning_rate"],
                          best_params["alpha"],
                          best_params["gamma"])

    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

    model, history = fit_model(model,
                               train_dataset=train_dataset,
                               steps_per_epoch=train_steps,
                               validation_dataset=val_dataset,
                               validation_steps=val_steps,
                               num_epochs=best_params["num_epochs"],
                               weight_positive=best_params["weight_positive"],
                               callbacks = [early_stop],
                               verbose=1)

    print("Starting Phase 2 of Fine Tuning...")

    model =  design_model_transfer_phase2(model, base_model, best_params["learning_rate"] * 0.5, best_params["alpha"], best_params["gamma"])

    model, history = fit_model(model,
                               train_dataset=train_dataset,
                               steps_per_epoch=train_steps,
                               validation_dataset=val_dataset,
                               validation_steps=val_steps,
                               num_epochs= best_params["num_epochs"] // 2,
                               weight_positive = best_params["weight_positive"],
                               callbacks = [early_stop],
                               verbose=1)

    #val_loss, val_precision, val_recall, val_auc = model.evaluate(val_dataset)

    return model, history
