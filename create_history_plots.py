import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def create_history_plots(input_history_vals1, input_history_vals2):
    history_vals1 = pd.read_csv(input_history_vals1)
    history_vals2 = pd.read_csv(input_history_vals2)

    history_total = pd.concat([history_vals1] + [history_vals2], ignore_index=True)
    # print(history_total)
    epochs_phase1 = len(history_vals1)
    epochs_phase2 = len(history_vals2)
    print(f"epochs phase 1 {epochs_phase1}")
    print(f"epochs phase 2 {epochs_phase2}")
    all_epochs_list = np.arange(1, len(history_total) + 1)

    # print(list(history_total.index+1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].plot(list(history_total.index + 1), history_total.loss, label="train")
    axes[0, 0].plot(list(history_total.index + 1), history_total.val_loss, label="validation")
    axes[0, 0].plot([epochs_phase1, epochs_phase1], plt.ylim(), label="start phase 2")
    # axes[0].plot(history_vals.epoch, [baseline_loss] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    # axes[0].plot(history_vals.epoch, [random_loss] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    # axes[0].plot(history_vals.epoch, [test_loss] * len(history_vals.epoch), label="test", color="green")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Log loss")
    axes[0, 0].set_xticks(np.arange(1, len(history_total.epoch) + 1, step=2))
    axes[0, 0].set_title("Phase 1 & 2: Train & Validation Loss")
    axes[0, 0].legend()

    # Phase 1: F1 Score
    axes[0, 1].plot(list(history_total.index + 1), history_total.f1_score, label="train")
    axes[0, 1].plot(list(history_total.index + 1), history_total.val_f1_score, label="validation")
    axes[0, 1].plot([epochs_phase1, epochs_phase1], plt.ylim(), label="start phase 2")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("f1 score")
    axes[0, 1].set_title("Phase 1 & 2: Train & Validation F1-Score")
    axes[0, 1].legend()

    # Phase 1: Precision
    axes[1, 0].plot(list(history_total.index + 1), history_total.precision, label="train")
    axes[1, 0].plot(list(history_total.index + 1), history_total.val_precision, label="validation")
    axes[1, 0].plot([epochs_phase1, epochs_phase1], plt.ylim(), label="start phase 2")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Phase 1 & 2: Train & Validation Precision")
    axes[1, 0].legend()

    # Phase 1: Recall
    axes[1, 1].plot(list(history_total.index + 1), history_total.recall, label="train")
    axes[1, 1].plot(list(history_total.index + 1), history_total.val_recall, label="validation")
    axes[1, 1].plot([epochs_phase1, epochs_phase1], plt.ylim(), label="start phase 2")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Phase 1 & 2: Train & Validation Recall")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

