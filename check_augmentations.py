import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

def load_image(file_path):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    print(img.dtype)

    # Normalize the pixel values
    img = tf.cast(img, tf.float32)
    #img = img / 255.0
    img = tf.image.resize(img, size=[224, 224])
    #img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    #img_test = img
    # img = tf.cast(img, tf.float32)
    img = vgg16_preprocess_input(img)
    return img


def augment_image(img):
    img_original = img
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    #img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    #img = tf.image.random_brightness(img, max_delta=0.1)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_crop(img, size=[200, 200, 3])
    img = tf.image.resize(img, size=[224, 224])


    #img = tf.image.resize_with_crop_or_pad(img, int(img.shape[0] * 0.8), int(img.shape[1] * 0.8))
    #img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    #img = tf.image.random_flip_left_right(img)
    #img = tf.image.random_flip_up_down(img)
    #img = tf.image.random_hue(img, max_delta=0.1)

    #img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

    #img = rotate_image(img, 30)
    #img = tf.image.random_crop(img, size = [180, 180, 3])
    #img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    #img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return img, img_original

def rescale_img(img):
    img_np = img.numpy()
    img = (img_np * 255.).astype(np.uint8)
    return img

def rescale_img_vgg(img):
    img = img.numpy() if hasattr(img, 'numpy') else img
    img[..., 0] += 103.939  # Add B mean
    img[..., 1] += 116.779  # Add G mean
    img[..., 2] += 123.68  # Add R mean

    print(img)
    #print(tf.reduce_max(img.numpy()))
    #print(tf.reduce_min(img.numpy()))
    print(np.max(img))
    print(np.min(img))

    img = img[..., ::-1]  # Convert BGR to RGB
    img = np.clip(img, 0, 255).astype(np.uint8)  # Cli
    return img


metadata_path = "../ISIC_data/ISIC_2020_Training_GroundTruth.csv"
metadata = pd.read_csv(metadata_path)
#metadata = metadata[metadata.diagnosis != "unknown" ]
metadata["image_path"] = '../ISIC_data/ISIC_2020_Training_JPEG/train/' + metadata['image_name'] + '.jpg'


for _ in range(5):
    num_images = 5
    img_samples = metadata[metadata.target==1]["image_path"].sample(num_images)
    print(img_samples)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
    for i in range(num_images):

        img = load_image(img_samples.iloc[i])
        img, img_original = augment_image(img)
        img = rescale_img_vgg(img)
        img_original = rescale_img_vgg(img_original)

        axes[i,0].imshow(img_original)
        axes[i,0].set_axis_off()
        axes[i,0].set_title("original image")

        axes[i,1].imshow(img)
        axes[i,1].set_title("augmented image")
        axes[i,1].set_axis_off()

    plt.tight_layout()
    plt.show()
