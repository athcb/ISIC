import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger("MainLogger")

def create_train_val_list(metadata_path, image_directory):

    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata.diagnosis != "unknown"]

    # convert sex column to binary
    metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1})

    # fill null age values
    median_age = round(metadata["age_approx"].median())
    metadata.fillna({"age_approx": median_age}, inplace=True)

    # create new category for null anatom_site values
    metadata.fillna({"anatom_site_general_challenge": "unknown"}, inplace=True)
    #print(metadata["anatom_site_general_challenge"].value_counts(normalize=True))

    logger.info(metadata.diagnosis.value_counts(normalize=True))
    logger.info(metadata.diagnosis.value_counts())
    logger.info("Rations of benign vs malignant samples:")
    logger.info(metadata.benign_malignant.value_counts(normalize=True))

    # create image_path column that points to local directory
    metadata["image_path"] = image_directory + "/"  + metadata['image_name'] + '.jpg'


    metadata["anatom_site_general_challenge"] = metadata["anatom_site_general_challenge"].replace(
        {"lower extremity": "lower_extremity",
         "upper extremity": "upper_extremity"})

    # keep columns of interest
    metadata = metadata[["image_path", "sex", "age_approx", "anatom_site_general_challenge", "target"]]

    # one hot encode categorical variables
    metadata = pd.get_dummies(metadata,
                              columns=["anatom_site_general_challenge"],
                              prefix=["site"],
                              drop_first=True)

    # perform stratified train / val split on the image paths based on the target column
    train_files, val_files = train_test_split(metadata[["image_path", "target"]],
                                              test_size = 0.2,
                                              shuffle=True,
                                              random_state = 10,
                                              stratify=metadata["target"])

    #smote = SMOTE(random_state=11)
    #train_files_resampled = smote.fit_resample(train_files)

    num_train_samples = len(train_files)
    num_val_samples = len(val_files)
    logger.info(f"Number of train images: {num_train_samples}")
    logger.info(f"Number of val images: {num_val_samples}")

    # create csv files with train and validation image file paths
    train_df = pd.DataFrame({"image_path": train_files["image_path"],
                  "label": train_files["target"],
                  "sex": train_files["sex"],
                  "age_approx": train_files["age_approx"],
                  "site_lower_extremity": train_files["site_lower_extremity"],
                  "site_oreal/genital": train_files["site_oreal/genital"],
                  "site_palms/soles": train_files["site_palms/soles"],
                  "site_torso": train_files["site_torso"],
                  "site_upper_extremity": train_files["site_upper_extremity"],
                  "site_unknown": train_files["site_unknown"]})

    val_df = pd.DataFrame({"image_path": val_files["image_path"],
                  "label": val_files["target"],
                  "sex": train_files["sex"],
                  "age_approx": val_files["age_approx"],
                  "site_lower_extremity": val_files["site_lower_extremity"],
                  "site_oreal/genital": val_files["site_oreal/genital"],
                  "site_palms/soles": val_files["site_palms/soles"],
                  "site_torso": val_files["site_torso"],
                  "site_upper_extremity": val_files["site_upper_extremity"],
                  "site_unknown": val_files["site_unknown"]
                  })

    # standardise columns
    ct = ColumnTransformer(transformers=[("scaler", StandardScaler(), "age_approx")],
                           remainder="passthrough")

    train_df = ct.fit_transform(train_df)
    val_df = ct.transform(val_df)

    logger.info(train_df.describe(include="all"))

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)

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
