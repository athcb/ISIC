import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def save_training_results(model, history, output_training_history, output_model):
    if os.path.exists(output_training_history):
        logger.info("Training history already exists.")
    else:
        model.save(output_model)
        logger.info(f"model saved to {output_model}")

        data = {"epoch": list(range(1, len(history.history["loss"]) + 1)),
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "precision": history.history["precision"],
                "val_precision": history.history["val_precision"],
                "recall": history.history["recall"],
                "val_recall": history.history["val_recall"],
                "f1_score": history.history["f1_score"],
                "val_f1_score": history.history["val_f1_score"],
                "AUC": history.history["AUC"],
                "val_AUC": history.history["val_AUC"]}

        history_vals = pd.DataFrame(data)
        history_vals.to_csv(output_training_history, index=False)

        logger.info("Training history saved to file. History values: ")
        logger.info(history_vals)