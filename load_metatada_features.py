import pandas as pd
import random

def load_metadata():
    """ Load training labels from csv with image metadata """
    metadata = pd.read_csv(metadata_directory)
    metadata = metadata[metadata.diagnosis != "unknown"]
    print(metedata.info())

    # Replace NAs in Sex columns with a random choice between male and female (only 65 out of 33127, should not bias model)
    metadata.sex = metadata.sex.apply(lambda x: random.choice(["male", "female"]) if pd.isna(x) else x)
    metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1})

    mean_age = round(metadata["age_approx"].mean())
    metadata.fillna({"age_approx": mean_age}, inplace=True)

    metadata["image_path"] = '../ISIC_data/ISIC_2020_Training_JPEG/train/' + metadata['image_name'] + '.jpg'

    metadata["anatom_site_general_challenge"] = metadata["anatom_site_general_challenge"].replace(
        {"lower extremity": "lower_extremity",
         "upper extremity": "upper_extremity"})
    metadata = pd.get_dummies(metadata,
                              columns=["diagnosis", "anatom_site_general_challenge"],
                              prefix=["diagnosis", "site"],
                              drop_first=True)

    print(metadata.image_path.head())
    return metadata[["image_path", "age_approx", "sex", "target"]]
