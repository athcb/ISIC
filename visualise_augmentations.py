import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
import albumentations as A
from tensorflow.keras.layers import RandomRotation


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
    #img = vgg16_preprocess_input(img)

    #img = vgg16_preprocess_input(img)
    #img_original = vgg16_preprocess_input(img_original)
    return img

random_rot = RandomRotation(factor=0.2)
def augment_image(img):
    img_original = img
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = random_rot(img, training=True)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_crop(img, size=[200, 200, 3])
    img = tf.image.resize(img, size=[224, 224])

    img = dropout(img, DIM=224, PROBABILITY = 0.6, CT = 6, SZ = 0.05)
    img = vgg16_preprocess_input(img)
    img_original = vgg16_preprocess_input(img_original)
    return img, img_original

def dropout(image, DIM=224, PROBABILITY = 1, CT = 5, SZ = 0.2):
    # input - one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1) < PROBABILITY, tf.int32)
    if (P == 0)|(CT == 0)|(SZ == 0): return image

    for k in range( CT ):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        # COMPUTE SQUARE
        WIDTH = tf.cast( SZ*DIM,tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.zeros([yb-ya,xb-xa,3])
        three = image[ya:yb,xb:DIM,:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    image = tf.reshape(image,[DIM,DIM,3])
    return image

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


metadata_path = "../ISIC_data/ISIC_2019_Training_Metadata.csv"
groundtruth_path_2019 = "../ISIC_data/ISIC_2019_Training_GroundTruth.csv"
ground_truth_2019 = pd.read_csv(groundtruth_path_2019)
metadata_2019 = pd.read_csv(metadata_path)
grouped_by_lesion = metadata_2019.groupby("lesion_id").agg(num_lesion_images=("image", "count"))
print(grouped_by_lesion[grouped_by_lesion.num_lesion_images >20])

metadata_2019 = pd.merge(metadata_2019, grouped_by_lesion["num_lesion_images"], on="lesion_id", how="left")
metadata_2019["image_path"] = "../ISIC_data/ISIC_2019_Training_Input/ISIC_2019_Training_Input" + "/" + metadata_2019["image"] + ".jpg"
metadata_2019 = metadata_2019[metadata_2019.lesion_type =="NV"]
num_images = 5
for i in range(len(metadata_2019)//num_images):

    img_samples = metadata_2019["image_path"].iloc[i*num_images :i*num_images+num_images]
    print(len(img_samples))
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))

    for j in range(len(img_samples)):
        img = load_image(img_samples.iloc[j])
        img, img_original = augment_image(img)
        img = rescale_img_vgg(img)
        img_original = rescale_img_vgg(img_original)

        axes[j,0].imshow(img_original)
        axes[j,0].set_axis_off()
        axes[j,0].set_title("original image")

        axes[j,1].imshow(img)
        axes[j,1].set_title("augmented image")
        axes[j,1].set_axis_off()

    plt.tight_layout()
    plt.show()
