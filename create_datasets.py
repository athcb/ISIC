from collections import Counter
import logging
import cv2

## Import Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

logger = logging.getLogger("MainLogger")


def load_image_metadata(file_path, label, image_weight, metadata, features, pretrained_model, img_size, num_channels):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=num_channels)
    img = tf.image.resize(img, size=[img_size, img_size])
    return {"input_image": img, "input_metadata": metadata, "input_features": features}, label, image_weight


def augment_image(inputs, labels, image_weight, img_size, crop_size):
    img = inputs["input_image"]
    metadata = inputs["input_metadata"]
    features = inputs["input_features"]
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, num_channels])
    img = tf.image.resize(img, size=[img_size, img_size])
    img = dropout(img, dim=img_size, probability=0.75, ct=6, sz=0.1)
    return {"input_image": img, "input_metadata": metadata, "input_features": features}, labels, image_weight


def preprocess_input_model(inputs, labels, image_weight, pretrained_model):
    img = inputs["input_image"]
    metadata = inputs["input_metadata"]
    features = inputs["input_features"]
    if pretrained_model == "vgg16":
        img = vgg16_preprocess_input(img)
    elif (pretrained_model == "densenet121") or (pretrained_model == "densenet169"):
        img = densenet_preprocess_input(img)
    elif pretrained_model == "efficientnetb4":
        img = efficientnet_preprocess_input(img)
    return {"input_image": img, "input_metadata": metadata, "input_features": features}, labels, image_weight


def create_train_val_datasets(file_paths, labels, image_weight, metadata, features, batch_size, num_epochs,
                              pretrained_model, img_size, num_channels, crop_size, training):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels, image_weight, metadata, features))
    dataset = dataset.map(
        lambda fp, lb, iw, md, ft: load_image_metadata(fp, lb, iw, md, ft, pretrained_model, img_size, num_channels),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    logger.info("Loaded images.")
    if training == True:
        logger.info("Starting augmentation (training set)...")
        # dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda inputs, labels, image_weight: augment_image(inputs, labels, image_weight, img_size, crop_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda inputs, labels, image_weight: preprocess_input_model(inputs, labels, image_weight, pretrained_model),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        logger.info("Starting shuffling and batching...")
        dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
        dataset_cardinality = dataset.cardinality().numpy()
        dataset_steps = len(file_paths) // batch_size
        logger.info(f"Cardinality train dataset: {dataset_cardinality}")
        logger.info(f"steps per epoch train dataset: {dataset_steps}")
    else:
        logger.info("Starting batching (validation set)...")
        dataset = dataset.map(
            lambda inputs, labels, image_weight: preprocess_input_model(inputs, labels, image_weight, pretrained_model),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset_cardinality = dataset.cardinality().numpy()
        dataset_steps = len(file_paths) // batch_size
        logger.info(f"Cardinality val dataset: {dataset_cardinality}")
        logger.info(f"steps per epoch val dataset: {dataset_steps}")
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, dataset_steps


def dropout(image, DIM=224, PROBABILITY=0.75, CT=8, SZ=0.2):
    # input - one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([], 0, 1) < PROBABILITY, tf.int32)
    if (P == 0) | (CT == 0) | (SZ == 0): return image

    for k in range(CT):
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        # COMPUTE SQUARE
        WIDTH = tf.cast(SZ * DIM, tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        # DROPOUT IMAGE
        one = image[ya:yb, 0:xa, :]
        two = tf.zeros([yb - ya, xb - xa, 3])
        three = image[ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        image = tf.concat([image[0:ya, :, :], middle, image[yb:DIM, :, :]], axis=0)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    image = tf.reshape(image, [DIM, DIM, 3])
    return image


def load_augment_image_metadata_unlabeled(file_path, img_size, num_channels, crop_size):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=num_channels)
    img = tf.image.resize(img, [img_size, img_size])

    # augmentation
    def augment_image2(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_crop(img, size=[crop_size, crop_size, num_channels])
        img = tf.image.resize(img, size=[img_size, img_size])
        img = dropout(img, dim=img_size, probability=0.75, ct=6, sz=0.1)
        img = resnet50_preprocess_input(img)
        return img

    img1 = augment_image2(tf.identity(img))
    img2 = augment_image2(tf.identity(img))

    return {"input1": img1, "input2": img2}, tf.constant(0)


def create_dataset_simclr(file_paths_all, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths_all)
    dataset = dataset.map(
        lambda img, img_size, num_channels, crop_size: load_augment_image_metadata_unlabeled(img, img_size,
                                                                                             num_channels, crop_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
