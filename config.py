from scipy.stats import loguniform

# directory of training images
image_directory = "../ISIC_data/ISIC_2020_Training_JPEG/train"

# path to metadata file
metadata_path = "../ISIC_data/ISIC_2020_Training_GroundTruth_v2.csv"

# define paths for csv files containing the results of randomised hyperparameter search
output_best_params = "./search_results/search_best_params.csv"
output_mean_scores = "./search_results/search_mean_scores.csv"
output_val_scores = "./search_results/search_val_scores.csv"
all_itr_results = "./search_results/all_itr_results.csv"

# path to save model
output_model = "./training_results/model.keras"

# csv containing the training history on an epoch level for visualisations
output_training_history1 = "./training_results/training_history_phase1.csv"
output_training_history2 = "./training_results/training_history_phase2.csv"

# Parameter grid for model without transfer learning
param_grid = {"img_size": [224],
              "num_channels": [3],
              #"num_filters": [[32, 64, 128], [64, 128, 256], [64, 128, 256, 512]],  # number of filter in the conv layers (length of list also defines number of conv layers)
              #"num_filters": [[32, 64, 128], [64, 128, 256], [8, 32, 64, 128]],
              "num_filters": [[32, 64, 128, 256]],
              "filter_size": [3],
              "padding_type": ["same"],
              "activation_fn": ["relu"],
              "l2_reg_conv": loguniform(1e-4, 1e-3), #L2 regularization in the convolutional layers
              "stride_conv": [1],
              "pool_s": [2], # filter size for pooling layer
              "pool_stride": [2], # strides for pooling layer
              "pool_padding": ["valid"], # padding type in pooling layer (valid: no padding, same: pad to keep same image dimensions)
              "dropout_val": [0.2, 0.3, 0.4, 0.5], # dropout value for fully connected dense layers
              "num_dense_units": [ [64, 32], [128, 64], [256, 128]], # number of units in the FC layers after the conv layers (length of list also defines number of FC layers)
              "activation_dense": ["relu"], # activation function in the FC layers
              "l2_reg_dense": loguniform(1e-5, 1e-3), #L2 regularization in the FC layers
              "nodes_output": [1], # number of nodes in the output layer (1 for binary classification, else number of classes)
              "activation_output": ["sigmoid"], # activation function for the output layer (sigmoid for binary classification, softmax for multi-classification)
              "learning_rate": loguniform(1e-4, 1e-3), # learning rate for gradient descent
              "num_epochs": [15], # number of training epochs
              "weight_positive": [1., 1., 1.], # weight for the minority (positive) class in case of imbalanced datasets
              "alpha": [0.3],
              "gamma": [1.]
              }

# Parameter grid for model with transfer learning
param_grid_tl = {"img_size": [224],
              "num_channels": [3],
              "dropout_val": [0.3, 0.4, 0.5], # dropout value for fully connected dense layers
              "num_dense_units": [ [256, 128, 64], [256, 64],  [512, 256, 64], [512], [512, 128], [128, 64, 32]], # number of units in the FC layers after the conv layers (length of list also defines number of FC layers)
              "activation_dense": ["relu"], # activation function in the FC layers
              "l2_reg_dense": loguniform(1e-5, 1e-3), #L2 regularization in the FC layers
              "nodes_output": [1], # number of nodes in the output layer (1 for binary classification, else number of classes)
              "activation_output": ["sigmoid"], # activation function for the output layer (sigmoid for binary classification, softmax for multi-classification)
              #"learning_rate": loguniform(1e-4, 1e-3), # learning rate for gradient descent
              "learning_rate": loguniform(1e-5, 1e-4), # learning rate for gradient descent
              "num_epochs": [30], # number of training epochs
              "weight_positive": [1., 1.5], # weight for the minority (positive) class in case of imbalanced datasets
              "alpha": [1.],
              "gamma": [0.],
              "num_dense_units_metadata": [[256, 128], [128, 64], [64, 32], [512], [256], [128]],
              "num_dense_units_combined": [[256, 128], [128, 64], [64, 32], [512], [256], [128]],
              "num_unfrozen_layers":[3, 4, 6],
              "decay_rate": [0.9, 0.95],
              "pooling_type": ["flatten", "global_avg", "global_max"],
              "batch_norm": [1, 0]
              }
