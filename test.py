import pandas as pd
from tensorflow.keras.backend import epsilon

precision =0.18
recall = 0.07
print(2 * (precision * recall) / (precision + recall + epsilon()))

print(2 * ((precision * recall) / (precision + recall + epsilon())))

metadata_path = "../ISIC_data/ISIC_2020_Training_GroundTruth.csv"



metadata = pd.read_csv(metadata_path)
metadata = metadata[metadata.diagnosis != "unknown" ]
print(metadata.diagnosis.value_counts(normalize=True))
print(metadata.diagnosis.value_counts())

print(metadata[metadata.diagnosis == "melanoma"].groupby("target").count())
