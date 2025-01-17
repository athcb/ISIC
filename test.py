import pandas as pd
from tensorflow.keras.backend import epsilon
from config import metadata_path
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import random


metadata = pd.read_csv(metadata_path)
metadata = metadata[metadata.diagnosis != "unknown"]
print(metadata.info())
print(metadata.groupby("sex").count())


# Replace NAs in Sex columns with a random choice between male and female (only 65 out of 33127, should not bias model)
#metadata.sex = metadata.sex.apply(lambda x: random.choice(["male", "female"]) if pd.isna(x) else x)
metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1})

median_age = round(metadata["age_approx"].median())
metadata.fillna({"age_approx": median_age}, inplace=True)

metadata.fillna({"anatom_site_general_challenge": "unknown"}, inplace=True)
print(metadata["anatom_site_general_challenge"].value_counts(normalize=True))

metadata["anatom_site_general_challenge"] = metadata["anatom_site_general_challenge"].replace(
        {"lower extremity": "lower_extremity",
         "upper extremity": "upper_extremity"})
metadata = pd.get_dummies(metadata,
                              columns=["anatom_site_general_challenge"],
                              prefix=["site"],
                              drop_first=True)

print(metadata.columns)
print(metadata.info())



#sns.countplot(x=metadata["anatom_site_general_challenge"], stat="percent", hue=metadata["benign_malignant"])
#plt.show()

#sns.countplot(x=metadata["diagnosis"], stat="percent", hue=metadata["benign_malignant"])
#plt.show()