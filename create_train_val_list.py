import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def create_train_val_list(metadata_path, image_directory):

    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata.diagnosis != "unknown"]
    logger.info(metadata.diagnosis.value_counts(normalize=True))
    logger.info(metadata.diagnosis.value_counts())
    logger.info("Rations of benign vs malignant samples:")
    logger.info(metadata.benign_malignant.value_counts(normalize=True))

    # create image_path column that points to local directory
    metadata["image_path"] = image_directory + "/"  + metadata['image_name'] + '.jpg'

    # perform stratified train / val split on the image paths based on the target column
    train_files, val_files = train_test_split(metadata[["image_path", "target"]],
                                              test_size = 0.2,
                                              shuffle=True,
                                              random_state = 10,
                                              stratify=metadata["target"])

    num_train_samples = len(train_files)
    num_val_samples = len(val_files)
    logger.info(f"Number of train images: {num_train_samples}")
    logger.info(f"Number of val images: {num_val_samples}")

    # create csv files with train and validation image file paths
    pd.DataFrame({"image_path": train_files["image_path"], "label": train_files["target"]}).to_csv("train.csv", index=False)
    pd.DataFrame({"image_path": val_files["image_path"], "label": val_files["target"]}).to_csv("val.csv", index=False)

    logger.info("Images per class in training set (ratios and total number):")
    train_paths = pd.read_csv("train.csv")
    logger.info(train_paths.label.value_counts(normalize=True))
    logger.info(train_paths.label.value_counts())

    logger.info("Suggested class weight: ", len(train_paths) / (2 * len(train_paths[train_paths.label == 1])))

    logger.info("Images per class in val set (ratios and total number):")
    val_paths = pd.read_csv("val.csv")
    logger.info(val_paths.label.value_counts(normalize=True))
    logger.info(val_paths.label.value_counts())

    return train_paths, val_paths
