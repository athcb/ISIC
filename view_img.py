import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pygments.styles.dracula import background

metadata_path = "../ISIC_data/ISIC_2020_Training_GroundTruth.csv"
metadata = pd.read_csv(metadata_path)
metadata = metadata[metadata.diagnosis != "unknown" ]
metadata["image_path"] = '../ISIC_data/ISIC_2020_Training_JPEG/train/' + metadata['image_name'] + '.jpg'

#select random image from a given diagnosis:
for i in range(20):
    random_sample = metadata[metadata.diagnosis == "melanoma"].sample(1)
    diagnosis = random_sample["diagnosis"].item()
    benign_malignant = random_sample["benign_malignant"].item()
    image_path = random_sample["image_path"].item()
    img = mpimg.imread(image_path)
    img_height, img_width, _ = img.shape
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.text(img_width * 0.05, img_height * 0.05, diagnosis, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.text(img_width * 0.05, img_height * 0.10, benign_malignant, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.axis("off")
    plt.show()

#print(metadata.diagnosis.value_counts(normalize=True))
#print(metadata.diagnosis.value_counts())

