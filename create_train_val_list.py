import os
from importlib.metadata import metadata
from locale import normalize

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger("MainLogger")

def preprocess_data_2019(image_directory_2019, metadata_path_2019, groundtruth_path_2019):
    ground_truth_2019 = pd.read_csv(groundtruth_path_2019)
    metadata_2019 = pd.read_csv(metadata_path_2019)

    # remove _downsampled images
    ground_truth_2019 = ground_truth_2019[~ground_truth_2019.image.str.contains("_downsampled")]

    # create target column based on melanoma diagnosis
    ground_truth_2019["target"] = ground_truth_2019.MEL.apply(lambda x: 1 if x == 1 else 0)

    # merge dfs with ground truth and metadata
    gt_metadata_2019 = pd.merge(ground_truth_2019[["image", "target"]], metadata_2019, on="image", how="left")
    gt_metadata_2019["image_path"] = image_directory_2019 +  "/" + gt_metadata_2019["image"] + ".jpg"

    # clean column: anatom_site_general
    gt_metadata_2019.rename(columns={"anatom_site_general": "site"}, inplace=True)
    gt_metadata_2019.fillna({"site": "unknown"}, inplace=True)
    gt_metadata_2019["site"] = gt_metadata_2019["site"].replace(
            {"lower extremity": "lower_extremity",
             "upper extremity": "upper_extremity",
             "anterior torso": "torso",
             "posterior torso": "torso",
             "lateral torso": "torso"})

    # clean column: sex
    #print(gt_metadata_2019[(gt_metadata_2019["sex"].isna()) & (gt_metadata_2019["target"] == 1)].groupby("lesion_id").count())
    gt_metadata_2019 = gt_metadata_2019[~gt_metadata_2019.sex.isna()]
    gt_metadata_2019["sex"] = gt_metadata_2019["sex"].map({"male": 0, "female": 1}).astype(int)

    # clean column: age_approx
    median_age = round(gt_metadata_2019["age_approx"].median())
    gt_metadata_2019.fillna({"age_approx": median_age}, inplace=True)

    # final df
    gt_metadata_2019 = gt_metadata_2019[["image_path", "target", "sex", "age_approx", "site"]]

    # log overview
    total_samples = gt_metadata_2019.shape[0]
    melanoma_samples = sum(gt_metadata_2019[gt_metadata_2019.target ==1]["target"])
    ratio_melanoma = melanoma_samples/total_samples
    logger.info(f"Pre-processed images from 2019 dataset:")
    logger.info(f"Number of 2019 images: {total_samples}")
    logger.info(f"Number of 2019 melanoma samples: {melanoma_samples}")
    logger.info(f"Ratio of melanoma to benign samples (2019) - includes images of same lesions from different angles: {ratio_melanoma}")

    return gt_metadata_2019


def preprocess_data_2020(image_directory_2020, metadata_path_2020, duplicates_path_2020):
    metadata_2020 = pd.read_csv(metadata_path_2020)
    duplicates_2020 = pd.read_csv(duplicates_path_2020)

    # remove duplicates from dataset
    metadata_2020 = pd.merge(metadata_2020, duplicates_2020["image_name_1"], left_on="image_name", right_on="image_name_1", how="left")
    metadata_2020 = metadata_2020[metadata_2020.image_name_1.isna()]

    # create image_path column that points to local directory
    metadata_2020["image_path"] = image_directory_2020 + "/" + metadata_2020['image_name'] + '.jpg'

    # remove samples with unknown diagnosis (all benign)
    metadata_2020 = metadata_2020[metadata_2020.diagnosis != "unknown"]
    # metadata = metadata[~metadata.sex.isna()]

    # clean column: sex
    metadata_2020['sex'] = metadata_2020['sex'].map({'male': 0, 'female': 1}).astype(int)

    # clean column: age_approx
    median_age = round(metadata_2020["age_approx"].median())
    metadata_2020.fillna({"age_approx": median_age}, inplace=True)

    # clean column: anatom_site_general_challenge
    # rename column
    metadata_2020.rename(columns={"anatom_site_general_challenge": "site"}, inplace=True)
    # create new category for null anatom_site values
    metadata_2020.fillna({"site": "unknown"}, inplace=True)
    # rename variables
    metadata_2020["site"] = metadata_2020["site"].replace({"lower extremity": "lower_extremity",
                                                 "upper extremity": "upper_extremity"})

    # final df
    metadata_2020 = metadata_2020[["image_path", "target", "sex", "age_approx", "site"]]

    # log overview
    total_samples = metadata_2020.shape[0]
    melanoma_samples = sum(metadata_2020[metadata_2020.target ==1]["target"])
    ratio_melanoma = melanoma_samples/total_samples
    logger.info(f"Pre-processed images from 2020 dataset:")
    logger.info(f"Number of 2020 images that were kept (samples with non-null diagnosis): {total_samples}")
    logger.info(f"Number of 2020 melanoma samples: {melanoma_samples}")
    logger.info(f"Ratio of melanoma to benign samples (2020): {ratio_melanoma}")

    return metadata_2020

def combine_datasets(metadata_2019, metadata_2020):
    combined_metadata = pd.concat([metadata_2019, metadata_2020], axis = 0).reset_index(drop=True)

    total_samples = combined_metadata.shape[0]
    melanoma_samples = sum(combined_metadata[combined_metadata.target ==1]["target"])
    ratio_melanoma = melanoma_samples / total_samples
    logger.info(f"Combined dataset (2019 & 2020), nrows: {total_samples}, ratio of melanoma samples: {ratio_melanoma}, total melanoma samples: {melanoma_samples}, total benign samples: {total_samples - melanoma_samples}")

    return combined_metadata


def create_train_val_list(image_directory_2019, metadata_path_2019, groundtruth_path_2019,
                          image_directory_2020, metadata_path_2020, duplicates_path_2020,
                          features_output):
    logger.info("Staring creation of train and validation lists (csv) containing the image paths per split.")

    metadata_2019 = preprocess_data_2019(image_directory_2019, metadata_path_2019, groundtruth_path_2019)
    metadata_2020 = preprocess_data_2020(image_directory_2020, metadata_path_2020, duplicates_path_2020)
    combined_df = combine_datasets(metadata_2019, metadata_2020)

    # load features from simclr model
    features_df = pd.read_csv(features_output)

    # one hot encode categorical variables
    combined_df = pd.get_dummies(combined_df,
                              columns=["site"],
                              drop_first=True)
    print(combined_df.columns)

    # perform stratified train / val split on the image paths based on the target column
    train_files, val_files = train_test_split(combined_df,
                                              test_size=0.2,
                                              shuffle=True,
                                              random_state=10,
                                              stratify=combined_df["target"])

    num_train_samples = len(train_files)
    num_val_samples = len(val_files)
    logger.info(f"Number of train images: {num_train_samples}")
    logger.info(f"Number of val images: {num_val_samples}")
    print("check train files cols: ", train_files.columns)

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

    # standardise columns
    ct = ColumnTransformer(transformers=[("scaler", StandardScaler(), ["age_approx"])],
                           remainder="passthrough")

    # standardise columns
    train_df_st = ct.fit_transform(train_df)
    val_df_st = ct.transform(val_df)

    train_df_st = pd.DataFrame(train_df_st, columns=["age_approx"] + list(train_df.columns.drop("age_approx")))
    val_df_st = pd.DataFrame(val_df_st, columns=["age_approx"] + list(train_df.columns.drop("age_approx")))

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

    print(f"image path type in train df {train_df_st.image_path.dtype}")
    num_rows1 = len(train_df_st)
    train_df_st = pd.merge(train_df_st, features_df, on="image_path", how="left")
    num_rows2 = len(train_df_st)
    print(f"train df rows before merge {num_rows1} and after merge {num_rows2}")

    num_rows1 = len(val_df_st)
    val_df_st = pd.merge(val_df_st, features_df, on="image_path", how="left")
    num_rows2 = len(val_df_st)
    print(f"val df rows before merge {num_rows1} and after merge {num_rows2}")

    # standardize features
    features_scaler = StandardScaler()
    train_df_st.iloc[:, 10:] = features_scaler.fit_transform(train_df_st.iloc[:, 10:])
    val_df_st.iloc[:, 10:] = features_scaler.transform(val_df_st.iloc[:, 10:])

    print(f"train_df_st columns {train_df_st.columns}")

    print(f"val_df_st columns {val_df_st.columns}")
    print(val_df_st.head())

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
    print(val_paths.label.value_counts(normalize=True))
    print(val_paths.label.value_counts())
    logger.info("Successfully created the train and validation csv.")

    return train_paths, val_paths


def create_file_paths_simclr(image_directory_2019, metadata_path_2019, groundtruth_path_2019,
                             image_directory_2020, metadata_path_2020, duplicates_path_2020):

    metadata_2019 = preprocess_data_2019(image_directory_2019, metadata_path_2019, groundtruth_path_2019)
    metadata_2020 = preprocess_data_2020(image_directory_2020, metadata_path_2020, duplicates_path_2020)
    combined_df = combine_datasets(metadata_2019, metadata_2020)

    file_paths_simclr = combined_df["image_path"]
    file_paths_simclr.to_csv("file_paths_all.csv", index=False)

    return file_paths_simclr


def oversample_minority(train_paths, oversampling_factor):
    majority_samples = train_paths[train_paths.label == 0]
    minority_samples = train_paths[train_paths.label == 1]
    train_paths_new = pd.concat([majority_samples] + [minority_samples] * oversampling_factor, ignore_index=True)

    train_paths_new = train_paths_new.sample(frac=1).reset_index(drop=True)

    new_minority_ratio = train_paths_new[train_paths_new.label == 1].shape[0] / train_paths_new.shape[0]
    logger.info(f"Minority ratio in train list after oversampling: {new_minority_ratio}")
    logger.info(f"Total samples after oversampling: {train_paths_new.shape[0]}")
    logger.info(f"Total benign samples after oversampling: {train_paths_new[train_paths_new.label == 0].shape[0]}")
    logger.info(f"Total melanoma samples after oversampling: {train_paths_new[train_paths_new.label == 1].shape[0]}")

    minority_class_weight = len(train_paths_new) / (2 * len(train_paths_new[train_paths_new.label == 1]))
    print("Suggested class weight after oversampling: ", {minority_class_weight})

    train_paths_new.to_csv("train_oversampled.csv", index=False)

    return train_paths_new


def undersample_majority(train_paths, undersampling_factor):
    majority_samples = train_paths[train_paths.label == 0]
    maj_samples_before = majority_samples.shape[0]

    new_maj_samples = majority_samples.sample(frac=(1 - undersampling_factor), random_state=11)
    maj_samples_after = new_maj_samples.shape[0]

    logger.info(f"Undersampling factor: {undersampling_factor}")
    logger.info(f"Majority samples removed: {maj_samples_before - maj_samples_after}")

    train_paths_new = pd.concat([train_paths[train_paths.label == 1]] + [new_maj_samples], ignore_index=True)
    train_paths_new = train_paths_new.sample(frac=1).reset_index(drop=True)

    new_maj_ratio = train_paths_new[train_paths_new.label == 0].shape[0] / train_paths_new.shape[0]
    logger.info(f"Maj ratio in train list after undersampling: {new_maj_ratio}")
    logger.info(f"Total samples after undersampling: {train_paths_new.shape[0]}")
    logger.info(f"Total benign samples after undersampling: {maj_samples_after}")
    logger.info(f"Total melanoma samples after undersampling: {train_paths_new[train_paths_new.label == 1].shape[0]}")

    # minority_class_weight = len(train_paths_new) / (2 * len(train_paths_new[train_paths_new.label == 1]))
    # print("Suggested class weight after oversampling: ", {minority_class_weight})

    train_paths_new.to_csv("train_undersampled.csv", index=False)

    return train_paths_new
