import io
import logging

## Import Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16

logger = logging.getLogger(__name__)

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
                      gamma):

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, num_channels))
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    for j in range(len(num_dense_units)):
        x = Dense(num_dense_units[j],
                  kernel_initializer=initializers.HeNormal(),
                  kernel_regularizer=l2(l2_reg_dense))(x)
        x = BatchNormalization()(x)
        x = Activation(activation_dense)(x)
        x = Dropout(dropout_val)(x)

    # output layer
    x = Dense(nodes_output, activation=activation_output)(x)

    model = Model(inputs= base_model.input, outputs = x)

    for layer in base_model.layers:
        layer.trainable = False

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
                           tf.keras.metrics.AUC(curve='PR')], # area under precision-recall curve
                  optimizer=opt)

    return model, base_model

def design_model_transfer_phase2(model, base_model, learning_rate, alpha, gamma):

    for layer in base_model.layers[-4:]:
        layer.trainable = True

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy",
                  #loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR')], # area under precision-recall curve
                  optimizer=opt)

    return model

