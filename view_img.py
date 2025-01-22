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


metadata_path = "../ISIC_data/ISIC_2020_Training_GroundTruth.csv"
metadata = pd.read_csv(metadata_path)
#metadata = metadata[metadata.diagnosis != "unknown" ]
metadata["image_path"] = '../ISIC_data/ISIC_2020_Training_JPEG/train/' + metadata['image_name'] + '.jpg'

#select random image from a given diagnosis:
#for i in range(30):
#    random_sample = metadata[metadata.diagnosis == "melanoma"].sample(1)
#    diagnosis = random_sample["diagnosis"].item()
#    benign_malignant = random_sample["benign_malignant"].item()
#    image_path = random_sample["image_path"].item()
#    img = mpimg.imread(image_path)
#    img_height, img_width, _ = img.shape
#    plt.figure(figsize=(10,10))
#    plt.imshow(img)
#    plt.text(img_width * 0.05, img_height * 0.05, diagnosis, fontsize=10, color="black", backgroundcolor="white", ha="center")
#    plt.text(img_width * 0.05, img_height * 0.10, benign_malignant, fontsize=10, color="black", backgroundcolor="white", ha="center")
#    plt.axis("off")
#    plt.show()


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



# Load a test image
#images_patient = metadata[metadata.patient_id == "IP_0038545"]
print(metadata.head())
images_patient = metadata[metadata.image_name == "ISIC_0080752"]
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


    plt.show()