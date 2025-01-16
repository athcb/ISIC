from collections import Counter

## Import Tensorflow libraries
import tensorflow as tf


def load_image(file_path, label):
    img = tf.io.read_file(file_path)  # requires local paths
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # Normalize the pixel values
    img = img / 255.0
    return img, label


def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label


def create_train_val_datasets(file_paths, labels, batch_size, num_epochs, training):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training == True:
        print("Starting augmentation...")
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print("Starting shuffling and batching...")
        dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
        dataset_cardinality = dataset.cardinality().numpy()
        dataset_steps = len(file_paths) // batch_size
        print("Cardinality train dataset: ", dataset_cardinality)
        print("steps per epoch train dataset: ", len(file_paths) // batch_size)
        print("steps per epoch train dataset float: ", len(file_paths) / batch_size)
        dataset = dataset.repeat(num_epochs)
        # dataset = dataset.batch(batch_size)
    else:
        print("Starting batching...")
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset_cardinality = dataset.cardinality().numpy()
        dataset_steps = len(file_paths) // batch_size
        print("Cardinality val dataset: ", dataset_cardinality)
        print("steps per epoch val dataset: ", len(file_paths) // batch_size)
        print("steps per epoch val dataset float: ", len(file_paths) / batch_size)
    return dataset, dataset_steps


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
