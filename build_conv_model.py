import io
import logging

## Import Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, initializers, Input
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import DenseNet121, DenseNet169

logger = logging.getLogger("MainLogger")


def build_model_phase1(img_size,
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
                       num_dense_units_metadata,
                       num_dense_units_features,
                       num_dense_units_combined,
                       pooling_type,
                       batch_norm,
                       decay_steps,
                       decay_rate,
                       pretrained_model):
    input_image = Input(shape=(img_size, img_size, num_channels), name="input_image")
    input_metadata = Input(shape=(num_metadata_features,), name="input_metadata")
    input_features = Input(shape=(128,), name="input_features")
    # input_weight = Input(shape=(1, ), name="input_weight")

    if pretrained_model == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, num_channels))
    elif pretrained_model == "densenet121":
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size, img_size, num_channels))
    elif pretrained_model == "densenet169":
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(img_size, img_size, num_channels))
    elif pretrained_model == "efficientnetb4":
        base_model = EfficientNetB4(weights='imagenet', include_top=False,
                                    input_shape=(img_size, img_size, num_channels))

    logger.info(f"Base model: {base_model.name}")
    logger.info(f"Trainable & Non-Trainable Layers:")
    for layer in base_model.layers:
        layer.trainable = False
        logger.info(f"Non-Trainable layer: {layer.name}")

    x = base_model(input_image)
    if pooling_type == "flatten":
        x = Flatten()(x)
    elif pooling_type == "global_avg":
        x = GlobalAveragePooling2D()(x)
    elif pooling_type == "global_max":
        x = GlobalMaxPooling2D()(x)

    for j in range(len(num_dense_units)):
        x = Dense(num_dense_units[j],
                  kernel_initializer=initializers.HeNormal(),
                  kernel_regularizer=l2(l2_reg_dense))(x)
        if batch_norm == 1:
            x = BatchNormalization()(x)
        x = Activation(activation_dense)(x)
        x = Dropout(dropout_val)(x)

    # metadata input
    y = input_metadata
    for m in range(len(num_dense_units_metadata)):
        y = Dense(num_dense_units_metadata[m],
                  kernel_initializer=initializers.HeNormal(),
                  kernel_regularizer=l2(l2_reg_dense))(y)
        if batch_norm == 1:
            y = BatchNormalization()(y)
        y = Activation(activation_dense)(y)
        y = Dropout(dropout_val)(y)

    z = input_features
    for f in range(len(num_dense_units_features)):
        z = Dense(num_dense_units_features[f],
                  kernel_initializer=initializers.HeNormal(),
                  kernel_regularizer=l2(l2_reg_dense))(z)
        if batch_norm == 1:
            z = BatchNormalization()(z)
        z = Activation(activation_dense)(z)
        z = Dropout(dropout_val)(z)

    # combine image and metadata features
    xy = Concatenate()([x, y])  # removed z
    for c in range(len(num_dense_units_combined)):
        xy = Dense(num_dense_units_combined[c],
                   kernel_initializer=initializers.HeNormal(),
                   kernel_regularizer=l2(l2_reg_dense))(xy)
        if batch_norm == 1:
            xy = BatchNormalization()(xy)
        xy = Activation(activation_dense)(xy)
        xy = Dropout(dropout_val)(xy)

    # output layer
    output = Dense(nodes_output, activation=activation_output)(xy)

    model = Model(inputs={"input_image": input_image,
                          "input_metadata": input_metadata,
                          "input_features": input_features},
                  outputs=output)

    logger.info(model.summary())

    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    opt = Adam(learning_rate=lr_schedule)
    model.compile(loss="binary_crossentropy",
                  # loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR', name="auc_pr"),
                           tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.AUC(name="auc")],
                  optimizer=opt)

    return model, base_model


def build_model_phase2(model, base_model, learning_rate, alpha, gamma, decay_steps, decay_rate,
                       lr_scaling_factor_phase2, pretrained_model):
    logger.info(f"Phase 2:")
    logger.info(f"Trainable & Non-Trainable Layers:")
    if pretrained_model == "vgg16":
        for layer in base_model.layers:
            if layer.name in ("block5_conv1", "block5_conv2", "block5_conv3"):
                layer.trainable = True
                logger.info(f"Trainable layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")
    elif (pretrained_model == "densenet121") or (pretrained_model == "densenet169"):
        for layer in base_model.layers:
            if ("conv5_block" in layer.name):
                if not any(excluded in layer.name for excluded in ["bn", "relu", "pool"]):
                    layer.trainable = True
                    logger.info(f"Trainable layer: {layer.name}")
                else:
                    layer.trainable = False
                    logger.info(f"Excluded (non-trainable) transition layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")
    elif pretrained_model == "efficientnetb4":
        for layer in base_model.layers:
            if ("block7" in layer.name):
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = True
                    logger.info(f"Trainable layer: {layer.name}")
                else:
                    layer.trainable = False
                    logger.info(f"Excluded (non-trainable) transition layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")

    # exponential learning rate decay
    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate * lr_scaling_factor_phase2,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    if pretrained_model == "vgg16":
        opt = Adam(learning_rate=learning_rate * lr_scaling_factor_phase2)
    elif (pretrained_model == "densenet121") or (pretrained_model == "densenet169"):
        opt = Adam(learning_rate=learning_rate * lr_scaling_factor_phase2)
    elif pretrained_model == "efficientnetb4":
        opt = Adam(learning_rate=lr_schedule)
    model.compile(loss="binary_crossentropy",
                  # loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR', name="auc_pr"),
                           tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.AUC(name="auc")],  # area under precision-recall curve
                  optimizer=opt)
    return model


def build_model_phase3(model, base_model, learning_rate, alpha, gamma, decay_steps, decay_rate,
                       lr_scaling_factor_phase3, pretrained_model):
    logger.info(f"Phase 3:")
    logger.info(f"Trainable & Non-Trainable Layers:")
    if pretrained_model == "vgg16":
        for layer in base_model.layers:
            if layer.name in ("block4_conv1", "block4_conv2", "block4_conv3",
                              "block5_conv1", "block5_conv2", "block5_conv3"):
                layer.trainable = True
                logger.info(f"Trainable layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")
    elif (pretrained_model == "densenet121") or (pretrained_model == "densenet169"):
        for layer in base_model.layers:
            if ("conv4_block" in layer.name or "conv5_block" in layer.name):
                if not any(excluded in layer.name for excluded in ["bn", "relu", "pool"]):
                    layer.trainable = True
                    logger.info(f"Trainable layer: {layer.name}")
                else:
                    layer.trainable = False
                    logger.info(f"Excluded (non-trainable) transition layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")
    elif pretrained_model == "efficientnetb4":
        for layer in base_model.layers:
            if ("block6" in layer.name or "block7" in layer.name):
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = True
                    logger.info(f"Trainable layer: {layer.name}")
                else:
                    layer.trainable = False
                    logger.info(f"Excluded (non-trainable) transition layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")

    # exponential learning rate decay
    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate * lr_scaling_factor_phase3,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    if pretrained_model == "vgg16":
        opt = Adam(learning_rate=learning_rate * lr_scaling_factor_phase3)  # 0.1
    elif (pretrained_model == "densenet121") or (pretrained_model == "densenet169"):
        opt = Adam(learning_rate=learning_rate * lr_scaling_factor_phase3)  # 0.05
    elif pretrained_model == "efficientnetb4":
        opt = Adam(learning_rate=lr_schedule)
        # opt = SGD(learning_rate=learning_rate*1.5, momentum=0.90)
    model.compile(loss="binary_crossentropy",
                  # loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR', name="auc_pr"),
                           tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.AUC(name="auc")],  # area under precision-recall curve
                  optimizer=opt)
    return model


def build_model_phase4(model, base_model, learning_rate, alpha, gamma, decay_steps, decay_rate,
                       lr_scaling_factor_phase4, pretrained_model):
    logger.info(f"Phase 4:")
    logger.info(f"Trainable & Non-Trainable Layers:")
    if pretrained_model == "vgg16":
        for layer in base_model.layers:
            if layer.name in ("block4_conv1", "block4_conv2", "block4_conv3",
                              "block5_conv1", "block5_conv2", "block5_conv3"):
                layer.trainable = True
                logger.info(f"Trainable layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")
    elif (pretrained_model == "densenet121") or (pretrained_model == "densenet169"):
        for layer in base_model.layers:
            if ("conv3_block" in layer.name or "conv4_block" in layer.name or "conv5_block" in layer.name):
                if not any(excluded in layer.name for excluded in ["bn", "relu", "pool"]):
                    layer.trainable = True
                    logger.info(f"Trainable layer: {layer.name}")
                else:
                    layer.trainable = False
                    logger.info(f"Excluded (non-trainable) transition layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")
    elif pretrained_model == "efficientnetb4":
        for layer in base_model.layers:
            if ("block6" in layer.name or "block7" in layer.name):
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = True
                    logger.info(f"Trainable layer: {layer.name}")
                else:
                    layer.trainable = False
                    logger.info(f"Excluded (non-trainable) transition layer: {layer.name}")
            else:
                layer.trainable = False
                logger.info(f"Non-Trainable layer: {layer.name}")

    # exponential learning rate decay
    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate * lr_scaling_factor_phase4,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    if pretrained_model == "vgg16":
        opt = Adam(learning_rate=learning_rate * lr_scaling_factor_phase4)  # 0.1
    elif (pretrained_model == "densenet121") or (pretrained_model == "densenet169"):
        opt = Adam(learning_rate=learning_rate * lr_scaling_factor_phase4)  # 0.02
    elif pretrained_model == "efficientnetb4":
        opt = Adam(learning_rate=lr_schedule)
        # opt = SGD(learning_rate=learning_rate*1.5, momentum=0.90)
    model.compile(loss="binary_crossentropy",
                  # loss = focal_loss(alpha=alpha, gamma=gamma),
                  metrics=[tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall"),
                           tf.keras.metrics.AUC(curve='PR', name="auc_pr"),
                           tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.AUC(name="auc")],
                  optimizer=opt)
    return model

