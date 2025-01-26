from collections import Counter
import logging
import cv2

## Import Tensorflow libraries
import tensorflow as tf
from tf.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tf.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tf.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

logger = logging.getLogger("MainLogger")

def load_image_metadata(file_path, label, metadata, features):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    # Normalize the pixel values
    #img = img / 255.0
    img = tf.image.resize(img, size=[224, 224])
    img = tf.cast(img, tf.float32)
    img = vgg16_preprocess_input(img)
    return {"input_image": img, "input_metadata": metadata, "input_features": features}, label


def augment_image(inputs, labels):
    img = inputs["input_image"]
    metadata = inputs["input_metadata"]
    features = inputs["input_features"]
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_crop(img, size=[210, 210, 3])
    img = tf.image.resize(img, size=[224, 224])
    return {"input_image": img, "input_metadata": metadata, "input_features": features}, labels


def load_augment_image_metadata_unlabeled(file_path):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # Normalize the pixel values
    #img = img / 255.0
    img = resnet50_preprocess_input(img) # scaling img to -1, 1

    # augmentation
    def augment_image2(img):
       img = tf.image.random_flip_left_right(img)
       img = tf.image.random_flip_up_down(img)
       img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
       img = tf.image.random_brightness(img, max_delta=0.1)
       img = tf.image.random_crop(img, size=[210, 210, 3])
       img = tf.image.resize(img, size=[224, 224])
       return img

    img1 = augment_image2(tf.identity(img))
    img2 = augment_image2(tf.identity(img))

    return {"input1": img1, "input2": img2}, tf.constant(0)


def create_train_val_datasets(file_paths, labels, metadata, features, batch_size, num_epochs, training):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels, metadata, features))
    dataset = dataset.map(load_image_metadata, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    logger.info("Loaded images.")
    if training == True:
        logger.info("Starting augmentation (training set)...")
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        logger.info("Starting shuffling and batching...")
        dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
        dataset_cardinality = dataset.cardinality().numpy()
        dataset_steps = len(file_paths) // batch_size
        logger.info(f"Cardinality train dataset: {dataset_cardinality}")
        logger.info(f"steps per epoch train dataset: {dataset_steps}")
    else:
        logger.info("Starting batching (validation set)...")
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset_cardinality = dataset.cardinality().numpy()
        dataset_steps = len(file_paths) // batch_size
        logger.info(f"Cardinality val dataset: {dataset_cardinality}")
        logger.info(f"steps per epoch val dataset: {dataset_steps}")
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, dataset_steps


def create_dataset_simclr(file_paths_all, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths_all)
    dataset = dataset.map(lambda img: load_augment_image_metadata_unlabeled(img), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size)
    return dataset

def check_class_distribution(dataset, dataset_name):
    labels = []
    for img, label in dataset.unbatch():
        labels.append(label.numpy())
    # print(labels)

    label_counts = Counter(labels)
    # print(label_counts)
    total_num = sum(label_counts.values())

    print(f"Class distribution in {dataset_name}:")
    for label, count in label_counts.items():
        print("Class ", label, " ratio: ", count / total_num)
    print("Total number of samples: ", total_num)
