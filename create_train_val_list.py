import os
from importlib.metadata import metadata
from locale import normalize

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import logging

from config import  image_directory_2020, image_directory_2019, metadata_path_2019, groundtruth_path_2019, metadata_path_2020, duplicates_path_2020, features_output

logger = logging.getLogger("MainLogger")

def preprocess_data_2019(image_directory_2019, metadata_path_2019, groundtruth_path_2019):
    ground_truth_2019 = pd.read_csv(groundtruth_path_2019)
    metadata_2019 = pd.read_csv(metadata_path_2019)

    metadata_2019 = metadata_2019.sort_values(by=["image", "lesion_id"])
    #metadata_2019["image_counter"] = metadata_2019.groupby("lesion_id").cumcount()
    #metadata_2019 = metadata_2019[(metadata_2019.image_counter == 0) | (metadata_2019.image_counter.isna()) ]

    lesion_classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    ground_truth_2019["lesion_type"] = ground_truth_2019[lesion_classes].idxmax(axis=1)

    grouped_by_lesion = metadata_2019.groupby("lesion_id").agg(num_lesion_images = ("image", "count"))
    #print(grouped_by_lesion[grouped_by_lesion.num_lesion_images >20])

    metadata_2019 = pd.merge(metadata_2019, grouped_by_lesion["num_lesion_images"], on = "lesion_id", how="left" )
    metadata_2019.fillna({"num_lesion_images": 1}, inplace=True)

    # remove _downsampled images
    ground_truth_2019 = ground_truth_2019[~ground_truth_2019.image.str.contains("_downsampled")]

    # create target column based on melanoma diagnosis
    ground_truth_2019["target"] = (ground_truth_2019.MEL == 1) | (ground_truth_2019.BCC == 1) | (ground_truth_2019.SCC == 1) | (ground_truth_2019.AK == 1)

    # merge dfs with ground truth and metadata
    gt_metadata_2019 = pd.merge(ground_truth_2019[["image", "target", "lesion_type"]], metadata_2019, on="image", how="left")
    gt_metadata_2019["image_path"] = image_directory_2019 +  "/" + gt_metadata_2019["image"] + ".jpg"
    #print(len(gt_metadata_2019))
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
    gt_metadata_2019 = gt_metadata_2019[~gt_metadata_2019.sex.isna()]
    gt_metadata_2019["sex"] = gt_metadata_2019["sex"].map({"male": 0, "female": 1}).astype(int)

    # clean column: age_approx
    median_age = round(gt_metadata_2019["age_approx"].median())
    gt_metadata_2019.fillna({"age_approx": median_age}, inplace=True)

    # final df
    gt_metadata_2019 = gt_metadata_2019[["image_path", "target", "sex", "age_approx", "site", "num_lesion_images", "lesion_id", "lesion_type"]]

    # log overview
    total_samples = gt_metadata_2019.shape[0]
    cancer_samples = sum(gt_metadata_2019[gt_metadata_2019.target ==1]["target"])
    ratio_cancer = cancer_samples/total_samples
    logger.info(f"Pre-processed images from 2019 dataset:")
    logger.info(f"Number of 2019 images (only one image per lesion was kept to remove bias): {total_samples}")
    logger.info(f"Number of 2019 cancer samples: {cancer_samples}")
    logger.info(f"Ratio of cancerous to benign samples (2019): {ratio_cancer}")

    return gt_metadata_2019

def preprocess_data_2020(image_directory_2020, metadata_path_2020, duplicates_path_2020):
    metadata_2020 = pd.read_csv(metadata_path_2020)
    duplicates_2020 = pd.read_csv(duplicates_path_2020)
    print(metadata_2020.diagnosis.unique())

    # map categpries to 2019 dataset
    category_mapping = {
        "melanoma": "MEL",
        "nevus": "NV",
        "seborrheic keratosis": "BKL",
        "lichenoid keratosis": "BKL",
        "solar lentigo": "BKL",
        "lentigo NOS": "AK",
        "atypical melanocytic proliferation": "AK",
        "cafe-au-lait macule": "unknown",
        "unknown": "unknown"}

    metadata_2020["lesion_type"] = metadata_2020["diagnosis"].map(category_mapping)

    # remove duplicates from dataset
    metadata_2020 = pd.merge(metadata_2020, duplicates_2020["image_name_1"], left_on="image_name", right_on="image_name_1", how="left")
    metadata_2020 = metadata_2020[metadata_2020.image_name_1.isna()]
    grouped_by_lesion = metadata_2020.groupby("lesion_id").agg(num_lesion_images=("image_name", "count"))
    metadata_2020 = pd.merge(metadata_2020, grouped_by_lesion["num_lesion_images"], on="lesion_id", how="left")

    # create image_path column that points to local directory
    metadata_2020["image_path"] = image_directory_2020 + "/" + metadata_2020['image_name'] + '.jpg'

    # remove samples with unknown diagnosis (all benign)
    metadata_2020 = metadata_2020[metadata_2020.diagnosis != "unknown"]

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
    metadata_2020 = metadata_2020[["image_path", "target", "sex", "age_approx", "site", "num_lesion_images", "lesion_id", "lesion_type"]]

    # log overview
    total_samples = metadata_2020.shape[0]
    cancer_samples = sum(metadata_2020[metadata_2020.target ==1]["target"])
    ratio_cancer = cancer_samples/total_samples
    logger.info(f"Pre-processed images from 2020 dataset:")
    logger.info(f"Number of 2020 images that were kept (samples with non-null diagnosis): {total_samples}")
    logger.info(f"Number of 2020 cancer samples: {cancer_samples}")
    logger.info(f"Ratio of cancerous to benign samples (2020): {ratio_cancer}")

    return metadata_2020

def combine_datasets(metadata_2019, metadata_2020):
    combined_metadata = pd.concat([metadata_2019, metadata_2020], axis = 0).reset_index(drop=True)
    print(combined_metadata.lesion_type.value_counts(normalize=True))

    total_samples = combined_metadata.shape[0]
    cancer_samples = sum(combined_metadata[combined_metadata.target ==1]["target"])
    ratio_cancer = cancer_samples / total_samples
    logger.info(f"Combined dataset (2019 & 2020), nrows: {total_samples}, ratio of cancer samples: {ratio_cancer}, total melanoma samples: {cancer_samples}, total benign samples: {total_samples - cancer_samples}")

    return combined_metadata


def create_train_val_list(image_directory_2019, metadata_path_2019, groundtruth_path_2019,
                          image_directory_2020, metadata_path_2020, duplicates_path_2020,
                          features_output):
    logger.info("Staring creation of train and validation lists (csv) containing the image paths per split.")

    metadata_2019 = preprocess_data_2019(image_directory_2019, metadata_path_2019, groundtruth_path_2019)
    metadata_2020 = preprocess_data_2020(image_directory_2020, metadata_path_2020, duplicates_path_2020)
    combined_df = combine_datasets(metadata_2019, metadata_2020)

    # load features from simclr model
    #features_df = pd.read_csv(features_output)

    # one hot encode categorical variables
    combined_df = pd.get_dummies(combined_df,
                              columns=["site"],
                              drop_first=True)

    lesion_grouped = combined_df.groupby("lesion_id").first().reset_index()

    train_lesions, val_lesions = train_test_split(
        lesion_grouped["lesion_id"],
        test_size=0.2,
        stratify=lesion_grouped["target"],  # Stratify based on lesion-level target
        random_state=10
    )

    train_files = combined_df[combined_df["lesion_id"].isin(train_lesions)]
    val_files = combined_df[combined_df["lesion_id"].isin(val_lesions)]

    # perform stratified train / val split on the image paths based on the target column
    #train_files, val_files = train_test_split(combined_df,
    #                                          test_size=0.2,
    #                                          shuffle=True,
    #                                          random_state=10,
    #                                          stratify=combined_df["target"])

    num_train_samples = len(train_files)
    num_val_samples = len(val_files)
    logger.info(f"Number of train images: {num_train_samples}")
    logger.info(f"Number of val images: {num_val_samples}")

    # create csv files with train and validation image file paths
    train_df = pd.DataFrame({"image_path": train_files["image_path"],
                             "label": train_files["target"],
                             "image_weight": 1./train_files["num_lesion_images"],
                             "lesion_type":train_files["lesion_type"],
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
                           "image_weight": 1. / val_files["num_lesion_images"],
                           "sex": val_files["sex"],
                           "age_approx": val_files["age_approx"],
                           "site_lower_extremity": val_files["site_lower_extremity"],
                           "site_oral_genital": val_files[r"site_oral/genital"],
                           "site_palms_soles": val_files[r"site_palms/soles"],
                           "site_torso": val_files["site_torso"],
                           "site_upper_extremity": val_files["site_upper_extremity"],
                           "site_unknown": val_files["site_unknown"]
                           })

    #train_df = pd.merge(train_df, features_df, on="image_path", how="left")
    #val_df = pd.merge(val_df, features_df, on="image_path", how="left")

    # standardise columns
    #ct = ColumnTransformer(transformers=[("scaler", StandardScaler(), ["age_approx"])],
    #                       remainder="passthrough")

    age_scaler = StandardScaler()
    train_df["age_approx"] = age_scaler.fit_transform(train_df[["age_approx"]])
    val_df["age_approx"] = age_scaler.transform(val_df[["age_approx"]])

    #features_scaler = StandardScaler()
    #train_df.iloc[:, 11:] = features_scaler.fit_transform(train_df.iloc[:, 11:])
    #val_df.iloc[:, 11:] = features_scaler.transform(val_df.iloc[:, 11:])


    # standardise columns
    #train_df_st = ct.fit_transform(train_df)
    #val_df_st = ct.transform(val_df)

    #train_df_st = pd.DataFrame(train_df_st, columns=["age_approx"] + list(train_df.columns.drop("age_approx")))
    #val_df_st = pd.DataFrame(val_df_st, columns=["age_approx"] + list(train_df.columns.drop("age_approx")))


    #train_df_st = train_df_st[correct_column_order]
    #val_df_st = val_df_st[correct_column_order]

    train_df["label"] = train_df["label"].astype(int)
    train_df["image_weight"] = train_df["image_weight"].astype(float)
    train_df["sex"] = train_df["sex"].astype(int)
    train_df["age_approx"] = train_df["age_approx"].astype(float)
    train_df["site_lower_extremity"] = train_df["site_lower_extremity"].astype(int)
    train_df["site_oral_genital"] = train_df["site_oral_genital"].astype(int)
    train_df["site_palms_soles"] = train_df["site_palms_soles"].astype(int)
    train_df["site_torso"] = train_df["site_torso"].astype(int)
    train_df["site_upper_extremity"] = train_df["site_upper_extremity"].astype(int)
    train_df["site_unknown"] = train_df["site_unknown"].astype(int)

    val_df["label"] = val_df["label"].astype(int)
    val_df["image_weight"] = val_df["image_weight"].astype(float)
    val_df["sex"] = val_df["sex"].astype(int)
    val_df["age_approx"] = val_df["age_approx"].astype(float)
    val_df["site_lower_extremity"] = val_df["site_lower_extremity"].astype(int)
    val_df["site_oral_genital"] = val_df["site_oral_genital"].astype(int)
    val_df["site_palms_soles"] = val_df["site_palms_soles"].astype(int)
    val_df["site_torso"] = val_df["site_torso"].astype(int)
    val_df["site_upper_extremity"] = val_df["site_upper_extremity"].astype(int)
    val_df["site_unknown"] = val_df["site_unknown"].astype(int)
    #print(f"image path type in train df {train_df_st.image_path.dtype}")
    #num_rows1 = len(train_df_st)
    #train_df_st = pd.merge(train_df_st, features_df, on="image_path", how="left")
    #num_rows2 = len(train_df_st)
    #print(f"train df rows before merge {num_rows1} and after merge {num_rows2}")

    #num_rows1 = len(val_df_st)
    #val_df_st = pd.merge(val_df_st, features_df, on="image_path", how="left")
    #num_rows2 = len(val_df_st)
    #print(f"val df rows before merge {num_rows1} and after merge {num_rows2}")

    # standardize features
    #features_scaler = StandardScaler()
    #train_df_st.iloc[:, 10:] = features_scaler.fit_transform(train_df_st.iloc[:, 10:])
    #val_df_st.iloc[:, 10:] = features_scaler.transform(val_df_st.iloc[:, 10:])



    train_df.to_csv("../ISIC_data/train.csv", index=False)
    val_df.to_csv("../ISIC_data/val.csv", index=False)

    train_paths = pd.read_csv("../ISIC_data/train.csv")

    print("Images per class in training set (ratios and total number):")
    print(train_paths.label.value_counts(normalize=True))
    print(train_paths.label.value_counts())

    minority_class_weight = len(train_paths) / (2 * len(train_paths[train_paths.label == 1]))
    logger.info("Suggested class weight (before under- or oversampling): ", {minority_class_weight})

    val_paths = pd.read_csv("../ISIC_data/val.csv")
    print("Images per class in val set (ratios and total number):")
    print(val_paths.label.value_counts(normalize=True))
    print(val_paths.label.value_counts())
    logger.info("Successfully created the train and validation csv.")

    return train_paths, val_paths


def create_file_paths_simclr(simclr_image_paths):

    train_paths = pd.read_csv("../ISIC_data/train.csv")
    file_paths_simclr = train_paths["image_path"]
    file_paths_simclr.to_csv(simclr_image_paths, index=False)

    logger.info(f"Created file with image paths for the simclr model at {simclr_image_paths}.")
    logger.info(f"Number of images for simclr model: {len(file_paths_simclr)}")

    return file_paths_simclr


def oversample_minority(train_paths, oversampling_factor):
    majority_samples = train_paths[train_paths.label == 0]

    #balance classes
    ak_samples = train_paths[train_paths.lesion_type == "AK"]
    scc_samples = train_paths[train_paths.lesion_type == "SCC"]
    mel_samples = train_paths[train_paths.lesion_type == "MEL"]
    bcc_samples = train_paths[train_paths.lesion_type == "BCC"]

    minority_samples = pd.concat([ak_samples] * 4 + [scc_samples] * 6 + [mel_samples] + [bcc_samples], ignore_index=True)
    print(minority_samples.lesion_type.value_counts(normalize=True))

    #minority_samples = train_paths_new[train_paths_new.label == 1]

    train_paths_new = pd.concat([majority_samples] + [minority_samples] * oversampling_factor, ignore_index=True)

    train_paths_new = train_paths_new.sample(frac=1).reset_index(drop=True)

    new_minority_ratio = train_paths_new[train_paths_new.label == 1].shape[0] / train_paths_new.shape[0]
    logger.info(f"Minority ratio in train list after oversampling: {new_minority_ratio}")
    logger.info(f"Total samples after oversampling: {train_paths_new.shape[0]}")
    logger.info(f"Total benign samples after oversampling: {train_paths_new[train_paths_new.label == 0].shape[0]}")
    logger.info(f"Total cancer samples after oversampling: {train_paths_new[train_paths_new.label == 1].shape[0]}")

    minority_class_weight = len(train_paths_new) / (2 * len(train_paths_new[train_paths_new.label == 1]))
    print("Suggested class weight after oversampling: ", {minority_class_weight})

    train_paths_new.to_csv("../ISIC_data/train_oversampled.csv", index=False)

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
    logger.info(f"Total cancer samples after undersampling: {train_paths_new[train_paths_new.label == 1].shape[0]}")

    # minority_class_weight = len(train_paths_new) / (2 * len(train_paths_new[train_paths_new.label == 1]))
    # print("Suggested class weight after oversampling: ", {minority_class_weight})

    train_paths_new.to_csv("../ISIC_data/train_undersampled.csv", index=False)

    return train_paths_new


preprocess_data_2019(image_directory_2019, metadata_path_2019, groundtruth_path_2019)
preprocess_data_2020(image_directory_2020, metadata_path_2020, duplicates_path_2020)
train_paths, val_paths = create_train_val_list(image_directory_2019, metadata_path_2019, groundtruth_path_2019,
                          image_directory_2020, metadata_path_2020, duplicates_path_2020,
                          features_output)
oversample_minority(train_paths, 2)