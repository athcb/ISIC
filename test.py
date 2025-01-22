import pandas as pd
from tensorflow.keras.backend import epsilon
from config import metadata_path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


metadata = pd.read_csv(metadata_path)

grouped = metadata.groupby("patient_id").agg(total_cancer = ("target", "sum"), total_images = ("target", "count"))
print(grouped[grouped.total_cancer >= 1].total_images.sum())


print(metadata[metadata.patient_id == "IP_0038545"])

metadata = metadata[metadata.diagnosis != "unknown"]

metadata['sex'] = metadata['sex'].map({'male': 0, 'female': 1})
# fill null age values
median_age = round(metadata["age_approx"].median())
metadata.fillna({"age_approx": median_age}, inplace=True)

# create new category for null anatom_site values
metadata.fillna({"anatom_site_general_challenge": "unknown"}, inplace=True)
# print(metadata["anatom_site_general_challenge"].value_counts(normalize=True))

metadata["anatom_site_general_challenge"] = metadata["anatom_site_general_challenge"].replace(
    {"lower extremity": "lower_extremity",
     "upper extremity": "upper_extremity"})

# keep columns of interest
metadata = metadata[["sex", "age_approx", "anatom_site_general_challenge", "target"]]

# one hot encode categorical variables
metadata = pd.get_dummies(metadata,
                          columns=["anatom_site_general_challenge"],
                          prefix=["site"],
                          drop_first=True)

X = metadata[["age_approx", "sex", "site_lower_extremity", "site_oral/genital", "site_palms/soles", "site_palms/soles", "site_unknown"]]
y = metadata["target"]
print(X.columns)
print(y)
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                          test_size=0.2,
                                          shuffle=True,
                                          random_state=10,
                                          stratify=y)

print(x_train.head())
print(x_test.head())
# standardise columns
scaler = StandardScaler()

x_train["age_approx"] = scaler.fit_transform(x_train[["age_approx"]])
x_test["age_approx"] = scaler.transform(x_test[["age_approx"]])

#ct = ColumnTransformer(transformers=[("scaler", StandardScaler(), ["age_approx"])],
#                       remainder="passthrough")

# standardise columns
model = RandomForestClassifier()

#model.fit(x_train, y_train)

# Get feature importance
#feature_importances = model.feature_importances_


# Display feature importance
#importance_df = pd.DataFrame({
#    'Feature': x_train.columns,
#    'Importance': feature_importances
#}).sort_values(by='Importance', ascending=False)
#print(importance_df)


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