from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from create_datasets import create_train_val_datasets
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
from config import image_directory_2019, image_directory_2020, metadata_path_2019, metadata_path_2020, groundtruth_path_2019, duplicates_path_2020, features_output
from create_train_val_list import create_train_val_list
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.utils import shuffle
from create_datasets import create_train_val_datasets

data_path = "val.csv"

model1_path = "./training_results/1/model.keras"
model1_name = "densenet121"

model2_path = "./training_results/2/model.keras"
model2_name = "efficientnetb4"

model3_path = "./training_results/3/model.keras"
model3_name = "densenet121"

model4_path = "./training_results/4/model.keras"
model4_name = "densenet169"

model5_path = "./training_results/model.keras"
model5_name = "vgg16"

# weights for model predictions
w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 1


class EvaluateModel:
    def __init__(self, model, model_name, data, batch_size):
        self.model = load_model(model)
        self.model_name = model_name
        self.data = data
        self.batch_size = batch_size

        self.val_dataset = self.preprocess_val_data()
        self.preds, self.true_labels = self.create_prediction_arrays()

    def preprocess_val_data(self):
        # image paths of validation dataset
        val_data = pd.read_csv(self.data)

        metadata = val_data.iloc[:, 3:11].to_numpy()
        features = val_data.iloc[:, 11:].to_numpy()
        val_dataset, _ = create_train_val_datasets(file_paths=val_data["image_path"].to_numpy(),
                                                   labels=val_data["label"].to_numpy(),
                                                   image_weight=val_data["image_weight"].to_numpy(),
                                                   metadata=metadata,
                                                   features=features,
                                                   batch_size=self.batch_size,
                                                   num_epochs=20,
                                                   pretrained_model=self.model_name,
                                                   img_size=224,
                                                   num_channels=3,
                                                   crop_size=200,
                                                   training=False)  # set to False for validation set
        return val_dataset

    def calculate_f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def calculate_scores(self):
        loss, precision, recall, auc_pr, accuracy, auc = self.model.evaluate(self.val_dataset)
        f1_score = self.calculate_f1_score(precision, recall)

        print(f"Results for model {self.model_name}:")
        print(f"F1 Score {f1_score}")
        print(f"AUC {auc}")
        print(f"AUC PR {auc_pr}")
        print(f"Accuracy {accuracy}")
        print(f"Precision {precision}")
        print(f"Recall {recall}")
        print(f"Loss {loss}")

    def create_prediction_arrays(self):
        preds = []
        true_labels = []

        for batch_inputs, batch_labels, batch_weights in self.val_dataset:
            batch_preds = self.model.predict(batch_inputs, verbose=0)
            preds.extend(batch_preds)
            true_labels.extend(batch_labels)

        preds = np.array(preds)
        true_labels = np.array(true_labels)

        return preds, true_labels

    def calculate_baseline_score(self):
        baseline_preds = np.zeros_like(self.true_labels)

        baseline_f1 = f1_score(self.true_labels, baseline_preds)
        baseline_accuracy = accuracy_score(self.true_labels, baseline_preds)
        baseline_precision = precision_score(self.true_labels, baseline_preds)
        baseline_recall = recall_score(self.true_labels, baseline_preds)
        baseline_auc = roc_auc_score(self.true_labels, baseline_preds)

        print(f"Results for Naive Classifier:")
        print(f"F1 Score {baseline_f1}")
        print(f"AUC {baseline_auc}")
        print(f"Accuracy {baseline_accuracy}")
        print(f"Precision {baseline_precision}")
        print(f"Recall {baseline_recall}")

    def calculate_random_score(self):

        minority_class_ratio = np.mean(self.true_labels)
        preds_raw = np.random.rand(len(self.true_labels))
        preds_raw = preds_raw * minority_class_ratio
        random_preds = np.random.choice([0, 1], size=len(self.true_labels),
                                        p=[1 - minority_class_ratio, minority_class_ratio])
        random_preds = shuffle(random_preds)

        random_f1 = f1_score(self.true_labels, random_preds)
        random_accuracy = accuracy_score(self.true_labels, random_preds)
        random_precision = precision_score(self.true_labels, random_preds)
        random_recall = recall_score(self.true_labels, random_preds)
        random_auc = roc_auc_score(self.true_labels, preds_raw)
        random_auc_pr = average_precision_score(self.true_labels, preds_raw)

        print(f"Results for Random Classifier (based on class distribution in validation or test set):")
        print(f"F1 Score {random_f1}")
        print(f"AUC {random_auc}")
        print(f"AUC PR {random_auc_pr}")
        print(f"Accuracy {random_accuracy}")
        print(f"Precision {random_precision}")
        print(f"Recall {random_recall}")

    def find_optimal_threshold(self, ensemble_preds=None, ensemble_labels=None):

        if ensemble_preds is not None:
            predictions = ensemble_preds
            labels = ensemble_labels
            name = "ensemble"
        else:
            predictions = self.preds
            labels = self.true_labels
            name = self.model_name

        threshold_range = np.arange(0.2, 1, 0.01)
        f1_values = []
        accuracy_values = []
        precision_values = []
        recall_values = []
        auc_values = []
        auc_pr_values = []

        for i in range(len(threshold_range)):
            y_pred = (predictions >= threshold_range[i]).astype(int)
            f1_values.append(f1_score(labels, y_pred))
            accuracy_values.append(accuracy_score(labels, y_pred))
            precision_values.append(precision_score(labels, y_pred))
            recall_values.append(recall_score(labels, y_pred))
            auc_values.append(roc_auc_score(labels, predictions))
            auc_pr_values.append(average_precision_score(labels, predictions))

        best_f1_idx = np.argmax(f1_values)
        final_threshold = threshold_range[best_f1_idx]
        final_f1_score = f1_values[best_f1_idx]
        final_accuracy_score = accuracy_values[best_f1_idx]
        final_precision_score = precision_values[best_f1_idx]
        final_recall_score = recall_values[best_f1_idx]
        final_auc_score = auc_values[best_f1_idx]
        final_auc_pr_score = auc_pr_values[best_f1_idx]

        print(f"Results for model {name} and threshold {final_threshold}:")
        print(f"F1 Score {final_f1_score}")
        print(f"Accuracy {final_accuracy_score}")
        print(f"Precision {final_precision_score}")
        print(f"Recall {final_recall_score}")
        print(f"AUC {final_auc_score}")
        print(f"AUC PR {final_auc_pr_score}")


class EvaluateEnsemble(EvaluateModel):
    def __init__(self,
                 model1_path, model1_name,
                 model2_path, model2_name,
                 model3_path, model3_name,
                 model4_path, model4_name,
                 model5_path, model5_name,
                 data_path, batch_size,
                 w1, w2, w3, w4, w5
                 ):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        self.model1 = EvaluateModel(model=model1_path, model_name=model1_name, data=data_path, batch_size=batch_size)
        self.model2 = EvaluateModel(model=model2_path, model_name=model2_name, data=data_path, batch_size=batch_size)
        self.model3 = EvaluateModel(model=model3_path, model_name=model3_name, data=data_path, batch_size=batch_size)
        self.model4 = EvaluateModel(model=model4_path, model_name=model4_name, data=data_path, batch_size=batch_size)
        self.model5 = EvaluateModel(model=model5_path, model_name=model5_name, data=data_path, batch_size=batch_size)

        assert np.array_equal(self.model1.true_labels, self.model2.true_labels), "True labels do not match.#1"
        assert np.array_equal(self.model1.true_labels, self.model3.true_labels), "True labels do not match.#2"

        self.labels = self.model1.true_labels

        self.ensemble_pred = self.combine_predictions()
        self.ensemble_pred_all = self.combine_predictions_all()

    def combine_predictions(self):
        ensemble_pred = (self.model1.preds * self.w1 + self.model3.preds * self.w3
                         # + self.model3.preds * self.w3 + self.model4.preds * self.w4
                         ) / (self.w1 + self.w3
                              # +  self.w3 +self.w4
                              )
        return ensemble_pred

    def combine_predictions_all(self):
        ensemble_pred_all = (self.model1.preds * self.w1 + self.model2.preds * self.w2
                             + self.model4.preds * self.w4 + self.model5.preds * self.w5
                             ) / (self.w1 + self.w2
                                  + self.w4 + self.w5
                                  )
        return ensemble_pred_all

    def calculate_ensemble_scores(self):
        print("Calculating ensemble predictions for 2 models")
        self.find_optimal_threshold(ensemble_preds=self.ensemble_pred, ensemble_labels=self.labels)

    def calculate_ensemble_scores_all(self):
        print("Calculating ensemble predictions for all models")
        self.find_optimal_threshold(ensemble_preds=self.ensemble_pred_all, ensemble_labels=self.labels)


# Evaluate individual model performance
model1 = EvaluateModel(model=model1_path, model_name=model1_name, data=data_path, batch_size=64)
# model1.calculate_scores()
# model1.find_optimal_threshold()
model1.calculate_baseline_score()
model1.calculate_random_score()

model2 = EvaluateModel(model=model2_path, model_name=model2_name, data=data_path, batch_size=64)
# model2.calculate_scores()
# model2.find_optimal_threshold()

model3 = EvaluateModel(model=model3_path, model_name=model3_name, data=data_path, batch_size=64)
# model3.calculate_scores()
# model3.find_optimal_threshold()


model4 = EvaluateModel(model=model4_path, model_name=model4_name, data=data_path, batch_size=64)
# model4.calculate_scores()
# model4.find_optimal_threshold()

model5 = EvaluateModel(model=model5_path, model_name=model5_name, data=data_path, batch_size=64)
# model5.calculate_scores()
# model5.find_optimal_threshold()


# Evaluate performace of ensemble model
ensemble = EvaluateEnsemble(model1_path, model1_name,
                            model2_path, model2_name,
                            model3_path, model3_name,
                            model4_path, model4_name,
                            model5_path, model5_name,
                            data_path, 64, w1, w4
                            , w3, w4, w5
                            )
print("using densenet121 and 169")
# ensemble.calculate_ensemble_scores()
print("excluding 2nd densenet121 and using 3 models instead")
# ensemble.calculate_ensemble_scores_all()
