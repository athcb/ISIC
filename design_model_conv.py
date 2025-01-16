## Import Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def design_model_conv(img_size,
                      num_channels,
                      num_filters,
                      filter_size,
                      padding_type,
                      activation_fn,
                      l2_reg_conv,
                      stride_conv,
                      pool_s,
                      pool_stride,
                      pool_padding,
                      dropout_val,
                      num_dense_units,
                      activation_dense,
                      l2_reg_dense,
                      nodes_output,
                      activation_output,
                      learning_rate,
                      alpha,
                      gamma):
    # initialize model instance
    model = Sequential()
    # input layer
    model.add(layers.InputLayer(shape=(img_size, img_size, num_channels)))

    for i in range(1, len(num_filters) + 1):

        model.add(Conv2D(num_filters[i - 1],
                         (filter_size, filter_size),
                         strides= (stride_conv, stride_conv),
                         padding=padding_type,
                         # activation= activation_fn,
                         kernel_initializer=initializers.HeNormal(),
                         kernel_regularizer=l2(l2_reg_conv)))
        model.add(BatchNormalization())
        model.add(Activation(activation_fn))

        if (i % 2) == 0:
            model.add(MaxPooling2D(pool_size=(pool_s, pool_s),
                                   strides=(pool_stride, pool_stride),
                                   padding=pool_padding))

    # model.add(MaxPooling2D(pool_size=(pool_s, pool_s),
    #                       strides=(pool_stride, pool_stride),
    #                       padding=pool_padding))
    # model.add(Flatten())
    model.add(GlobalMaxPooling2D())
    # model.add(Dropout(dropout_val))

    for j in range(len(num_dense_units)):
        model.add(Dense(num_dense_units[j],
                        # activation = activation_dense,
                        kernel_initializer=initializers.HeNormal(),
                        kernel_regularizer=l2(l2_reg_dense)))
        model.add(BatchNormalization())
        model.add(Activation(activation_dense))
        model.add(Dropout(dropout_val))

    # output layer
    model.add(Dense(nodes_output, activation=activation_output))

    print(model.summary())

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy",
                  #loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR')], # area under precision-recall curve
                  optimizer=opt)

    return model

