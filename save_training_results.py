import tensorflow as tf
import pandas as pd
import os
import logging

logger = logging.getLogger("MainLogger")


def calculate_f1_score(precision, recall):
    f1_score = [2 * (prec * rec) / (prec + rec + tf.keras.backend.epsilon()) for prec, rec in zip(precision, recall)]
    return f1_score


def save_training_results(model, history_phase1, history_phase2, history_phase3,
                          output_training_history1, output_training_history2, output_training_history3,
                          output_model):
    if os.path.exists(output_training_history1):
        logger.info("Training history already exists.")
    else:
        model.save(output_model)
        logger.info(f"model saved to {output_model}")

        phase1_metrics = history_phase1.history
        phase2_metrics = history_phase2.history
        phase3_metrics = history_phase3.history

        data_phase1 = {"epoch": list(range(1, len(phase1_metrics["loss"]) + 1)),
                       "loss": phase1_metrics["loss"],
                       "val_loss": phase1_metrics["val_loss"],
                       "precision": phase1_metrics["precision"],
                       "val_precision": phase1_metrics["val_precision"],
                       "recall": phase1_metrics["recall"],
                       "val_recall": phase1_metrics["val_recall"],
                       "f1_score": calculate_f1_score(phase1_metrics["precision"], phase1_metrics["recall"]),
                       "val_f1_score": calculate_f1_score(phase1_metrics["val_precision"],
                                                          phase1_metrics["val_recall"]),
                       "auc_pr": phase1_metrics["auc_pr"],
                       "val_auc_pr": phase1_metrics["val_auc_pr"],
                       "auc": phase1_metrics["auc_pr"],
                       "val_auc": phase1_metrics["val_auc"],
                       "accuracy": phase1_metrics["accuracy"],
                       "val_accuracy": phase1_metrics["val_accuracy"]
                       }

        data_phase2 = {"epoch": list(range(1, len(phase2_metrics["loss"]) + 1)),
                       "loss": phase2_metrics["loss"],
                       "val_loss": phase2_metrics["val_loss"],
                       "precision": phase2_metrics["precision"],
                       "val_precision": phase2_metrics["val_precision"],
                       "recall": phase2_metrics["recall"],
                       "val_recall": phase2_metrics["val_recall"],
                       "f1_score": calculate_f1_score(phase2_metrics["precision"], phase2_metrics["recall"]),
                       "val_f1_score": calculate_f1_score(phase2_metrics["val_precision"],
                                                          phase2_metrics["val_recall"]),
                       "auc_pr": phase2_metrics["auc_pr"],
                       "val_auc_pr": phase2_metrics["val_auc_pr"],
                       "auc": phase2_metrics["auc_pr"],
                       "val_auc": phase2_metrics["val_auc"],
                       "accuracy": phase2_metrics["accuracy"],
                       "val_accuracy": phase2_metrics["val_accuracy"]
                       }

        data_phase3 = {"epoch": list(range(1, len(phase3_metrics["loss"]) + 1)),
                       "loss": phase3_metrics["loss"],
                       "val_loss": phase3_metrics["val_loss"],
                       "precision": phase3_metrics["precision"],
                       "val_precision": phase3_metrics["val_precision"],
                       "recall": phase3_metrics["recall"],
                       "val_recall": phase3_metrics["val_recall"],
                       "f1_score": calculate_f1_score(phase3_metrics["precision"], phase3_metrics["recall"]),
                       "val_f1_score": calculate_f1_score(phase3_metrics["val_precision"],
                                                          phase3_metrics["val_recall"]),
                       "auc_pr": phase3_metrics["auc_pr"],
                       "val_auc_pr": phase3_metrics["val_auc_pr"],
                       "auc": phase3_metrics["auc_pr"],
                       "val_auc": phase3_metrics["val_auc"],
                       "accuracy": phase3_metrics["accuracy"],
                       "val_accuracy": phase3_metrics["val_accuracy"]
                       }

        metrics_phase1 = pd.DataFrame(data_phase1)
        metrics_phase2 = pd.DataFrame(data_phase2)
        metrics_phase3 = pd.DataFrame(data_phase3)

        metrics_phase1.to_csv(output_training_history1, index=False)
        metrics_phase2.to_csv(output_training_history2, index=False)
        metrics_phase3.to_csv(output_training_history3, index=False)

        logger.info("Training history for phase 1, 2, 3 saved to file.")
