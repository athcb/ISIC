from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute import batch_sizes_for_worker


class KerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn,
                 img_size=224,
                 num_channels=1,
                 num_epochs=10,
                 learning_rate=0.001,
                 num_conv_layers=3,
                 num_filters=[32, 64, 128],
                 filter_size=3,
                 stride_num=1,
                 l2_reg_conv=0.00001,
                 padding_type="same",
                 activation_fn="relu",
                 activation_dense="relu",
                 pool_s=2,
                 pool_stride=2,
                 pool_padding="valid",
                 num_dense_layers=2,
                 num_dense_units=[64, 32],
                 dropout_val=0.3,
                 l2_reg_dense=0.0001,
                 nodes_output=1,
                 activation_output="sigmoid" ,
                 weight_positive = 15.,
                 batch_size = 16,
                 task_type="image_classification"
                 ):
        self.build_fn = build_fn
        self.num_epochs = num_epochs
        self.num_filters = num_filters
        self.num_dense_units = num_dense_units
        self.learning_rate = learning_rate
        self.dropout_val = dropout_val
        self.task_type = task_type
        self.l2_reg_conv = l2_reg_conv
        self.l2_reg_dense = l2_reg_dense
        self.img_size = img_size
        self.num_channels = num_channels
        self.num_conv_layers = num_conv_layers
        self.filter_size = filter_size
        self.stride_num = stride_num
        self.pool_s = pool_s
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.nodes_output = nodes_output
        self.activation_output = activation_output
        self.padding_type = padding_type
        self.activation_fn = activation_fn
        self.activation_dense = activation_dense
        self.num_dense_layers = num_dense_layers
        self.weight_positive = weight_positive
        self.batch_size = batch_size

    def fit(self, X, y=None, callbacks=None):


        self.model_ = self.build_fn(img_size=self.img_size, num_channels=self.num_channels,
            num_conv_layers=self.num_conv_layers, num_filters=self.num_filters,
            filter_size=self.filter_size, padding_type= self.padding_type, activation_fn= self.activation_fn,
            l2_reg_conv=self.l2_reg_conv, stride_num=self.stride_num, pool_s=self.pool_s,
            pool_stride=self.pool_stride, pool_padding=self.pool_padding, dropout_val=self.dropout_val,
            num_dense_layers= self.num_dense_layers, num_dense_units=self.num_dense_units, activation_dense= self.activation_dense,
            l2_reg_dense=self.l2_reg_dense, nodes_output=self.nodes_output,
            activation_output=self.activation_output, learning_rate=self.learning_rate, weight_positive = self.weight_positive, steps_per_epoch = self.steps_per_epoch )

        class_weight_dict = {0: 1., 1: self.weight_positive}
        if callbacks is None:
            callbacks = []
        if isinstance(X, tf.data.Dataset):
            self.model_.fit(X,
                            epochs=self.num_epochs,
                            verbose=1,
                            callbacks=callbacks,
                            class_weight= class_weight_dict)
        else:
            self.model_.fit(X, y, epochs=self.num_epochs,verbose=1, callbacks=callbacks, class_weight= class_weight_dict)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y=None):
        if isinstance(X, tf.data.Dataset):
            score = self.model_.evaluate(X, class_weight = self.weight_positive, verbose=1)
        else:
            score = self.model_.evaluate(X, y, verbose=1)
        print(f"Evaluation score: {score}")
        return score


