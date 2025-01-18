import io
import logging

## Import Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, initializers, Input
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16

logger = logging.getLogger("MainLogger")

def design_model_transfer_phase1(img_size,
                      num_channels,
                      dropout_val,
                      num_dense_units,
                      activation_dense,
                      l2_reg_dense,
                      nodes_output,
                      activation_output,
                      learning_rate,
                      alpha,
                      gamma,
                      num_metadata_features,
                      num_dense_units_metadata):

    input_image = Input(shape=(img_size, img_size, num_channels), name="input_image")
    input_metadata = Input(shape=(num_metadata_features, ), name="input_metadata")

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, num_channels))

    for layer in base_model.layers:
        layer.trainable = False

    # image input
    x = base_model(input_image)
    x = GlobalMaxPooling2D()(x)
    #x = Flatten()(x)

    for j in range(len(num_dense_units)):
        x = Dense(num_dense_units[j],
                  kernel_initializer=initializers.HeNormal(),
                  kernel_regularizer=l2(l2_reg_dense))(x)
        # x = BatchNormalization()(x)
        x = Activation(activation_dense)(x)
        x = Dropout(dropout_val)(x)

    # metadata input
    y = input_metadata
    for m in range(len(num_dense_units_metadata)):
        y = Dense(num_dense_units_metadata[m],
                  kernel_initializer=initializers.HeNormal(),
                  kernel_regularizer=l2(l2_reg_dense))(y)
        y = Activation(activation_dense)(y)
        y = Dropout(dropout_val)(y)

    # combine image and metadata features
    xy = Concatenate()([x,y])

    # output layer
    output = Dense(nodes_output, activation=activation_output)(xy)

    model = Model(inputs= [input_image, input_metadata], outputs = output)

    for i, layer in enumerate(base_model.layers):
        logger.info(f"layer {i}: {layer.name}")

    #stream = io.StringIO()
    #model.summary(print_fn = lambda x: stream.write(f"{x}\n"))
    logger.info(model.summary())
    #summary_log = stream.getvalue()
    #logger.info(summary_log)

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy",
                  #loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR', name="auc_pr")], # area under precision-recall curve
                  optimizer=opt)

    return model, base_model

def design_model_transfer_phase2(model, base_model, learning_rate, alpha, gamma, num_unfrozen_layers, decay_steps, decay_rate):

    # define trainable layers
    for layer in base_model.layers[-num_unfrozen_layers:]:
        layer.trainable = True

    # exponential learning rate decay
    lr_schedule = ExponentialDecay(
        initial_learning_rate = learning_rate,
        decay_steps = decay_steps,
        decay_rate = decay_rate,
        staircase = True
    )

    opt = Adam(learning_rate=lr_schedule)
    model.compile(loss="binary_crossentropy",
                  #loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR', name="auc_pr")], # area under precision-recall curve
                  optimizer=opt)

    return model

