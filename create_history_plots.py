import pandas as pd
from matplotlib import pyplot as plt


def create_history_plots(input_history_vals1, input_history_vals2):
    history_vals1 = pd.read_csv(input_history_vals1)
    history_vals2 = pd.read_csv(input_history_vals2)

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))

    # Phase 1: Log loss
    axes[0, 0].plot(history_vals1.epoch, history_vals1.loss, label="train")
    axes[0, 0].plot(history_vals1.epoch, history_vals1.val_loss, label="validation")
    # axes[0].plot(history_vals.epoch, [baseline_loss] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    # axes[0].plot(history_vals.epoch, [random_loss] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    # axes[0].plot(history_vals.epoch, [test_loss] * len(history_vals.epoch), label="test", color="green")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Log loss")
    axes[0, 0].set_title("Phase 1: Train & Validation Loss")
    axes[0, 0].legend()

    # Phase 1: F1 Score
    axes[0, 1].plot(history_vals1.epoch, history_vals1.f1_score, label="train")
    axes[0, 1].plot(history_vals1.epoch, history_vals1.val_f1_score, label="validation")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("f1 score")
    axes[0, 1].set_title("Phase 1: Train & Validation F1-Score")
    axes[0, 1].legend()

    # Phase 1: Precision
    axes[1, 0].plot(history_vals1.epoch, history_vals1.precision, label="train")
    axes[1, 0].plot(history_vals1.epoch, history_vals1.val_precision, label="validation")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Phase 1: Train & Validation Precision")
    axes[1, 0].legend()

    # Phase 1: Recall
    axes[1, 1].plot(history_vals1.epoch, history_vals1.recall, label="train")
    axes[1, 1].plot(history_vals1.epoch, history_vals1.val_recall, label="validation")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Phase 1: Train & Validation Recall")
    axes[1, 1].legend()

    # Phase 2: Log Loss
    axes[2, 0].plot(history_vals2.epoch, history_vals2.loss, label="train")
    axes[2, 0].plot(history_vals2.epoch, history_vals2.val_loss, label="validation")
    axes[2, 0].set_xlabel("Epochs")
    axes[2, 0].set_ylabel("Log loss")
    axes[2, 0].set_title("Phase 2: Train & Validation Loss")
    axes[2, 0].legend()

    # Phase 2: F1 Score
    axes[2, 1].plot(history_vals2.epoch, history_vals2.f1_score, label="train")
    axes[2, 1].plot(history_vals2.epoch, history_vals2.val_f1_score, label="validation")
    axes[2, 1].set_xlabel("Epochs")
    axes[2, 1].set_ylabel("f1 score")
    axes[2, 1].set_title("Phase 2: Train & Validation F1-Score")
    axes[2, 1].legend()

    # Phase 2: Precision
    axes[3, 0].plot(history_vals2.epoch, history_vals2.precision, label="train")
    axes[3, 0].plot(history_vals2.epoch, history_vals2.val_precision, label="validation")
    axes[3, 0].set_xlabel("Epochs")
    axes[3, 0].set_ylabel("Log loss")
    axes[3, 0].set_title("Phase 2: Train & Validation Precision")
    axes[3, 0].legend()

    # Phase 2: Recall
    axes[3, 1].plot(history_vals2.epoch, history_vals2.recall, label="train")
    axes[3, 1].plot(history_vals2.epoch, history_vals2.val_recall, label="validation")
    axes[3, 1].set_xlabel("Epochs")
    axes[3, 1].set_ylabel("Recall")
    axes[3, 1].set_title("Phase 2: Train & Validation Recall")
    axes[3, 1].legend()

    plt.tight_layout()
    plt.show()

