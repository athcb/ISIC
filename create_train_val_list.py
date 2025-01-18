import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger("MainLogger")

def create_train_val_list(metadata_path, image_directory):

    logger.info("Staring creation of train and validation lists (csv) containing the image paths per split.")
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata.diagnosis != "unknown"]
    print(metadata.columns)

    # convert sex column to binary
    metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1})

    # fill null age values
    median_age = round(metadata["age_approx"].median())
    metadata.fillna({"age_approx": median_age}, inplace=True)
    print(metadata.age_approx.describe())

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
    train_files, val_files = train_test_split(metadata,
                                              test_size = 0.2,
                                              shuffle=True,
                                              random_state = 10,
                                              stratify=metadata["target"])

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
                  "site_oral_genital": train_files[r"site_oral/genital"],
                  "site_palms_soles": train_files[r"site_palms/soles"],
                  "site_torso": train_files["site_torso"],
                  "site_upper_extremity": train_files["site_upper_extremity"],
                  "site_unknown": train_files["site_unknown"]})

    val_df = pd.DataFrame({"image_path": val_files["image_path"],
                  "label": val_files["target"],
                  "sex": val_files["sex"],
                  "age_approx": val_files["age_approx"],
                  "site_lower_extremity": val_files["site_lower_extremity"],
                  "site_oral_genital": val_files[r"site_oral/genital"],
                  "site_palms_soles": val_files[r"site_palms/soles"],
                  "site_torso": val_files["site_torso"],
                  "site_upper_extremity": val_files["site_upper_extremity"],
                  "site_unknown": val_files["site_unknown"]
                  })
    print(train_df.info())
    # standardise columns
    ct = ColumnTransformer(transformers=[("scaler", StandardScaler(), ["age_approx"])],
                           remainder="passthrough")

    # standardise columns
    train_df_st = ct.fit_transform(train_df)
    val_df_st = ct.transform(val_df)

    train_df_st = pd.DataFrame(train_df_st, columns = ["age_approx"] + list(train_df.columns.drop("age_approx")))
    val_df_st = pd.DataFrame(val_df_st, columns= ["age_approx"] + list(train_df.columns.drop("age_approx")))

    correct_column_order = ["image_path",
                               "label",
                               "sex",
                               "age_approx",
                               "site_lower_extremity",
                               "site_oral_genital",
                               "site_palms_soles",
                               "site_torso",
                               "site_upper_extremity",
                               "site_unknown"]
    train_df_st = train_df_st[correct_column_order]
    val_df_st = val_df_st[correct_column_order]

    train_df_st["label"] = train_df_st["label"].astype(int)
    train_df_st["sex"] = train_df_st["sex"].astype(int)
    train_df_st["age_approx"] = train_df_st["age_approx"].astype(float)
    train_df_st["site_lower_extremity"] = train_df_st["site_lower_extremity"].astype(int)
    train_df_st["site_oral_genital"] = train_df_st["site_oral_genital"].astype(int)
    train_df_st["site_palms_soles"] = train_df_st["site_palms_soles"].astype(int)
    train_df_st["site_torso"] = train_df_st["site_torso"].astype(int)
    train_df_st["site_upper_extremity"] = train_df_st["site_upper_extremity"].astype(int)
    train_df_st["site_unknown"] = train_df_st["site_unknown"].astype(int)

    val_df_st["label"] = val_df_st["label"].astype(int)
    val_df_st["sex"] = val_df_st["sex"].astype(int)
    val_df_st["age_approx"] = val_df_st["age_approx"].astype(float)
    val_df_st["site_lower_extremity"] = val_df_st["site_lower_extremity"].astype(int)
    val_df_st["site_oral_genital"] = val_df_st["site_oral_genital"].astype(int)
    val_df_st["site_palms_soles"] = val_df_st["site_palms_soles"].astype(int)
    val_df_st["site_torso"] = val_df_st["site_torso"].astype(int)
    val_df_st["site_upper_extremity"] = val_df_st["site_upper_extremity"].astype(int)
    val_df_st["site_unknown"] = val_df_st["site_unknown"].astype(int)

    print(train_df_st.columns)
    print(train_df_st.info())

    train_df_st.to_csv("train.csv", index=False)
    val_df_st.to_csv("val.csv", index=False)


    train_paths = pd.read_csv("train.csv")

    print("Images per class in training set (ratios and total number):")
    print(train_paths.label.value_counts(normalize=True))
    print(train_paths.label.value_counts())

    minority_class_weight = len(train_paths) / (2 * len(train_paths[train_paths.label == 1]))
    print("Suggested class weight: ", {minority_class_weight})

    val_paths = pd.read_csv("val.csv")
    print("Images per class in val set (ratios and total number):")
    print(val_paths.head())
    print(val_paths.label.value_counts(normalize=True))
    print(val_paths.label.value_counts())
    logger.info("Successfully created the train and validation csv.")

    return train_paths, val_paths
