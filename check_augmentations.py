import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
import albumentations as A


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
    return img


def augment_image(img):
    img_original = img
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    #img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.random_brightness(img, max_delta=0.1)
    #img = tf.clip_by_value(img, 0.0, 1.0)

    img = tf.image.random_crop(img, size=[200, 200, 3])
    img = tf.image.resize(img, size=[224, 224])

    img = dropout(img, DIM=224, PROBABILITY = 0.75, CT = 5, SZ = 0.15)

    img = vgg16_preprocess_input(img)
    img_original = vgg16_preprocess_input(img_original)


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

def coarse_dropout(images, square_size, num_squares, prob):
    ''' Coarse image dropout
    Coarse dropout masks are created by using tf.scatter_nd
    Which can update matrices according to coordinates specified
    by the user.

    Parameters
    ----------
    images : tf Tensor
        A batch of images of size [BS, H, W, C]
    square_size : float
        A float between [0,1], a percentage of the image size
    num_squares : int > 0
        The number of dropout squares per single image, must be >0
    prob : float
        A float between [0,1], probability that dropout is used for this batch

    Yields
    ------
    TYPE
        DESCRIPTION.

    '''
       # random dropout
    prob = tf.cast( tf.random.uniform([],0,1) < prob, tf.int32)
    if (prob == 0): return images

    img_shape = images.shape
    _, h, w, c = img_shape[0], img_shape[1], img_shape[2], img_shape[3]

    # For some odd reason, the batch size is lost in the processing pipeline
    # seriously, how can it get lost ._., it literally says ds.batch()...
    # tensorflow shenanigans...
    bs = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int32)


    # size of square in pixels (ssp), ASSUMING SQUARE IMAGES!
    ssp = tf.cast(tf.math.ceil(h * square_size), tf.int32)

    # Create random start x coordinates
    coords_x = tf.random.uniform([bs, num_squares],
                                 minval=0,
                                 maxval= (h - ssp),
                                 dtype=tf.int32)

    # Create ranges from start to the end of the line
    # [ssp, bs, num_squares]
    coords_x = tf.linspace(coords_x, coords_x + ssp - 1, ssp)

    # [num_squares, bs, ssp]
    coords_x = tf.cast(tf.transpose(coords_x), tf.int32)

    # Create random start y coordinates
    coords_y = tf.random.uniform([bs, num_squares],
                                 minval=0,
                                 maxval= (h - ssp),
                                 dtype=tf.int32)

    # Create ranges from start to the end of the line
    # [ssp, bs, num_squares]
    coords_y = tf.linspace(coords_y, coords_y + ssp - 1, ssp)

    # [num_squares, bs, ssp]
    coords_y = tf.cast(tf.transpose(coords_y), tf.int32)


    # Create coordinate range combinations
    # and reshape to [bs, num_squares, 1, ssp * ssp]
    grid_y = tf.reshape(tf.tile(coords_y, [1,1,ssp]),
                        (bs, num_squares, 1, ssp * ssp))

    # and reshape to [bs, num_squares, ssp, ssp], transpose the inner matrices
    grid_y = tf.transpose(tf.reshape(grid_y,
                                     (bs, num_squares, ssp, ssp)),
                          (0, 1, 3, 2))

    # Repeat for x coordinates
    grid_x = tf.reshape(tf.tile(coords_x, [1,1,ssp]),
                        (bs, num_squares, 1, ssp * ssp))

    grid_x = tf.reshape(grid_x, (bs, num_squares, ssp, ssp))

    # Stack the grids into a single matrix
    # grid is [2, bs, num_squares, ssp, ssp]
    grid = tf.stack([grid_y, grid_x], axis=0)

    # Transpose and reshape [ bs, ssp * ssp * num_squares, 2]
    # Creates an array of 2D coordinates ([[x1,y1], [x2, y2], ..., [xn, yn]])
    # over all squares (num_squares), for each combination (ssp*ssp) of coordinates
    # and each batch (bs)
    grid = tf.reshape(tf.transpose(grid, (1, 4, 3, 2, 0)),
                      (bs, ssp * ssp * num_squares, 2))

    # [bs, sz*sz*num_squares, 2]
    #grid = tf.reshape(grid, (bs, ssp*ssp*num_squares, 2))

    # create batch indices [0,..., bs] and reshape to [ssp * ssp * num_squares, bs]
    batch_indices = tf.reshape(tf.tile(tf.range(0, bs),
                                       [ssp * ssp * num_squares]),
                               (ssp * ssp * num_squares, bs))

    # Transpose to get the right order, and reshape to match grid shape
    batch_indices = tf.reshape(tf.transpose(batch_indices),
                               (bs, ssp * ssp * num_squares, 1))

    # concatenate batch indices with the grid
    # these yield 3D coordinates like e.g.
    # [[bs0, x1, y1], [bs0, x2, y1], ..., [bsn, xn, yn]]
    grid = tf.concat([batch_indices, grid], axis=2)

    # create a matrix of zeros, and update the matrix with the grid indices
    # this essentially creates a mask with coarse dropout squares
    masks = tf.scatter_nd(grid[tf.newaxis,...],
                            tf.ones([1,bs,ssp*ssp*num_squares]) * -1,
                            shape=(bs, h, w)) +1

    # Due to overlap of squares, some get coordinates get updated twice
    # and result in values < -1, clip these values
    masks = tf.clip_by_value(masks, 0, 1)

    return images * masks[..., tf.newaxis]


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
