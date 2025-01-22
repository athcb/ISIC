from collections import Counter
import logging
import cv2

## Import Tensorflow libraries
import tensorflow as tf

logger = logging.getLogger("MainLogger")

def load_image_metadata(file_path, label, metadata):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # Normalize the pixel values
    img = img / 255.0
    return {"input_image": img, "input_metadata": metadata}, label


def load_augment_image_metadata_unlabeled(file_path):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # Normalize the pixel values
    #img = img / 255.0
    img = preprocess_input(img) # scaling img to -1, 1

    # augmentation
    def augment_image(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, max_delta=0.1)
        img = tf.image.random_saturation(img, lower=0.95, upper=1.05)
        # img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_brightness(img, max_delta=0.1)
        return img

    img1 = augment_image(tf.identity(img))
    img2 = augment_image(tf.identity(img))

    return {"input1": img1, "input2": img2}, tf.constant(0)


def augment_image(inputs, labels):
    img = inputs["input_image"]
    metadata = inputs["input_metadata"]
    img = tf.image.random_flip_left_right(img)
    #img = tf.image.random_flip_up_down(img)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.95, upper=1.05)
    #img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_brightness(img, max_delta = 0.1)
    #img = tf.image.random_crop(img, size = [180, 180, 3])
    #img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    #img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return {"input_image": img, "input_metadata": metadata}, labels

def create_train_val_datasets(file_paths, labels, metadata, batch_size, num_epochs, training):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels, metadata))
    dataset = dataset.map(load_image_metadata, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
    return dataset, dataset_steps


def create_dataset_simclr(file_paths_all, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths_all)
    #dataset = dataset.map(lambda img: (load_augment_image_metadata_unlabeled(img),tf.constant(0)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
