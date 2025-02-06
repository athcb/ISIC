import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pygments.styles.dracula import background
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import Unet
import tensorflow as tf

"""
image_directory_2019 = "../ISIC_data/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"

ground_truth_2019 = pd.read_csv("../ISIC_data/ISIC_2019_Training_GroundTruth.csv")
metadata_2019 = pd.read_csv("../ISIC_data/ISIC_2019_Training_Metadata.csv")
ground_truth_2019 = ground_truth_2019[~ground_truth_2019.image.str.contains("_downsampled")]
ground_truth_2019["target"] = ground_truth_2019.MEL.apply(lambda x: 1 if x == 1 else 0)

gt_metadata_2019 = pd.merge(ground_truth_2019[["image", "target", "BCC", "SCC", "AK", "MEL"]], metadata_2019, on="image", how="left")
gt_metadata_2019["image_path"] = image_directory_2019 + gt_metadata_2019["image"] + ".jpg"

image_paths = gt_metadata_2019[gt_metadata_2019.MEL == 1]
for i in range(len(image_paths)):
    path =image_paths.image_path.iloc[i]
    target = image_paths.target.iloc[i]
    img = mpimg.imread(path)
    img_height, img_width, _ = img.shape
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.text(img_width * 0.05, img_height * 0.05, target, fontsize=10, color="black", backgroundcolor="white", ha="center")
    #plt.text(img_width * 0.05, img_height * 0.10, benign_malignant, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.axis("off")
    plt.show()
"""

metadata_path = "../ISIC_data/ISIC_2020_Training_GroundTruth.csv"
metadata = pd.read_csv(metadata_path)
print(metadata.diagnosis.value_counts())
metadata = metadata[metadata.diagnosis == "atypical melanocytic proliferation" ]
metadata["image_path"] = '../ISIC_data/ISIC_2020_Training_JPEG/train/' + metadata['image_name'] + '.jpg'
print(len(metadata))
#select random image from a given diagnosis:
for i in range(len(metadata)):
    random_sample = metadata.sample(1)
    diagnosis = random_sample["diagnosis"].item()
    benign_malignant = random_sample["benign_malignant"].item()
    image_path = random_sample["image_path"].item()
    image_name = random_sample["image_name"].item()
    img = mpimg.imread(image_path)
    img_height, img_width, _ = img.shape
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.text(img_width * 0.05, img_height * 0.05, image_name, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.text(img_width * 0.05, img_height * 0.10, benign_malignant, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.axis("off")
    plt.show()

artefacts = ["ISIC_0504165", "ISIC_9052500", "ISIC_9596721", "ISIC_9863642",  "ISIC_9022005", "ISIC_0351666", "ISIC_3963183", "ISIC_4404772",
             "ISIC_3817719", "ISIC_7800750", "ISIC_6255113", "ISIC_3561065", "ISIC_9967383", "ISIC_9038318", "ISIC_2757355", "ISIC_8263489",
             "ISIC_2072219", "ISIC_1388552", "ISIC_4938994", "ISIC_3517311", "ISIC_8483382", "ISIC_4851366", "ISIC_2797353"]
# view images from a single patient:
"""
images_patient = metadata[metadata.patient_id == "IP_0038545"]
print(images_patient)
for i in range(len(images_patient)):
    diagnosis = images_patient["diagnosis"].iloc[i]
    benign_malignant = images_patient["benign_malignant"].iloc[i]
    image_path = images_patient["image_path"].iloc[i]
    img = mpimg.imread(image_path)
    img_height, img_width, _ = img.shape
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.text(img_width * 0.05, img_height * 0.05, diagnosis, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.text(img_width * 0.05, img_height * 0.10, benign_malignant, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.axis("off")
    plt.show()
"""
"""
# view images from a single patient:
images_patient = metadata[metadata.patient_id == "IP_0038545"]
print(images_patient)
for i in range(len(images_patient)):
    diagnosis = images_patient["diagnosis"].iloc[i]
    benign_malignant = images_patient["benign_malignant"].iloc[i]
    image_path = images_patient["image_path"].iloc[i]

    img = mpimg.imread(image_path)
    img_height, img_width, _ = img.shape
    img_resized_or = cv2.resize(img, (224, 224))
    img_resized = img_resized_or/255.
    img_resized = np.expand_dims(img_resized, axis = 0)


    mask = unet_model.predict(img_resized)[0]
    mask = np.squeeze(mask, axis = -1)
    mask_binary = mask > 0.5

    # original image
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized_or)
    plt.text(img_width * 0.05, img_height * 0.05, diagnosis, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.text(img_width * 0.05, img_height * 0.10, benign_malignant, fontsize=10, color="black", backgroundcolor="white", ha="center")
    plt.axis("off")

    # segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Segmentation mask")
    plt.show()
"""


"""
# Load a test image
#images_patient = metadata[metadata.patient_id == "IP_0038545"]
print(metadata.head())
images_patient = metadata[metadata.target == 1]
print(images_patient)
for i in range(len(images_patient)):
    diagnosis = images_patient["diagnosis"].iloc[i]
    benign_malignant = images_patient["benign_malignant"].iloc[i]
    image_path = images_patient["image_path"].iloc[i]

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blackhat-like operation: Morphological filtering to detect dark hair
    kernel_size = 17  # Adjust for hair thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create a binary mask for hairs
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)  # Binary mask (0 or 255)

    # Inpainting: Replace hair pixels with surrounding context
    inpainted_img = cv2.inpaint(img, hair_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    inpainted_img_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
    # Remove hair: mask out pixels
    #hair_removed_img = input_img_rgb * (1 - hair_mask)

    # Display original and hair-removed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(inpainted_img_rgb)
    #plt.imshow(hair_removed_img.astype(np.uint8))
    plt.title("Hair Removed Image")
    plt.axis("off")


    plt.show()"""