from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from io import BytesIO
import pandas as pd
import random
import tensorflow as tf
import h5py
import numpy as np
from collections import Counter

image_directory = "../ISIC_data/ISIC_2020_Training_JPEG/train"
metadata_directory = "../ISIC_data/ISIC_2020_Training_GroundTruth_v2.csv"


def load_metadata():
    """ Load training labels from csv with image metadata """
    metadata = pd.read_csv(metadata_directory)

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
    return metadata[["image_path", "age_approx", "sex", "target"]].head(1000)


def load_image(file_path, label):
    #with fs.open(file_path, "rb") as f:
    #    img = f.read()  # reads image as byte data
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    # img = Image.open(f).convert("RGB")
    img = tf.image.resize(img, [224, 224])
    # Normalize the pixel values
    img = img / 255.0
    # img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img, label


def data_generator(metadata):
    for _, row in metadata.iterrows():
        file_path = row["image_path"]
        # print("file path in generator")
        # print(file_path)
        label = row["target"]
        yield load_image(file_path, label)


def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label


def create_train_val_datasets(metadata, train_percent, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(metadata),
        output_signature=(tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.int32))
    )

    # shuffle the images before splitting into training and validation sets
    dataset = dataset.shuffle(buffer_size=1000)

    # define size of training set
    train_size = int(train_percent * metadata.shape[0])

    # split datasets
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # apply augmentation to train dataset:
    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(3000)

    # train_dataset = train_dataset.cache()
    # val_dataset = val_dataset.cache()

    # batch the train and validation sets
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def save_as_hdf5(data, filename):
    images = []
    labels = []

    for image_batch, label_batch in data:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    with h5py.File(filename, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("labels", data=labels)


print("Loading metadata...")
metadata = load_metadata()
print(f"Metadata for {metadata.shape[0]} images")

print("Creating training and validation sets...")
train_dataset, val_dataset = create_train_val_datasets(metadata, train_percent=0.8, batch_size=16)
print("Training and validation sets created.")

# save datasets as h5 files
# save_as_hdf5(train_dataset, "train_dataset.h5")
# save_as_hdf5(val_dataset, "val_dataset.h5")
# print("Saved training and validation datasets locally")

# save datasets to an S3 bucket
# output_bucket = "isic-split-datasets"
# s3.upload_file("train_dataset.h5", output_bucket, "train/train_dataset.h5")
# s3.upload_file("val_dataset.h5", output_bucket, "val/val_dataset.h5")
# print("Uploaded training and validation datasets to S3")


for batch_img, batch_labels in train_dataset.take(1):
    print("shape of images and labels in first batch of training dataset:")
    print(batch_img.shape, batch_labels.shape)
    print(batch_labels.numpy())

for batch_img, batch_labels in val_dataset.take(1):
    print("shape of images and labels in first batch of val dataset:")
    print(batch_img.shape, batch_labels.shape)
    print(batch_labels.numpy())


def check_class_distribution(dataset, dataset_name):
    labels = []
    for img, label in dataset.unbatch():
        labels.append(label.numpy())
    # print(labels)

    label_counts = Counter(labels)
    print(label_counts)
    total_num = sum(label_counts.values())

    print(f"Class distribution in {dataset_name}:")
    for label, count in label_counts.items():
        print("Class ", label, " ratio: ", count / total_num)
    print("Total number of samples: ", total_num)


check_class_distribution(train_dataset, "train dataset")
check_class_distribution(val_dataset, "val dataset")