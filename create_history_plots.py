import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def create_history_plots(input_history_vals1, input_history_vals2, input_history_vals3, input_history_vals4):
    history_vals1 = pd.read_csv(input_history_vals1)
    history_vals2 = pd.read_csv(input_history_vals2)
    history_vals3 = pd.read_csv(input_history_vals3)
    history_vals4 = pd.read_csv(input_history_vals4)

    history_total = pd.concat([history_vals1] + [history_vals2] + [history_vals3] + [history_vals4], ignore_index=True)
    # print(history_total)
    epochs_phase1 = len(history_vals1)
    epochs_phase2 = len(history_vals2)
    epochs_phase3 = len(history_vals3)
    epochs_phase4 = len(history_vals4)
    print(f"epochs phase 1 {epochs_phase1}")
    print(f"epochs phase 2 {epochs_phase2}")
    print(f"epochs phase 3 {epochs_phase3}")
    print(f"epochs phase 4 {epochs_phase4}")

    all_epochs_list = np.arange(1, len(history_total) + 1)
    start_phase3 = epochs_phase1 + epochs_phase2
    start_phase4 = epochs_phase1 + epochs_phase2 + epochs_phase3
    xticks = np.arange(1, len(all_epochs_list) + 1, step=2)

    # print(list(history_total.index+1))

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    # Loss
    axes[0, 0].plot(all_epochs_list, history_total.loss, label="train")
    axes[0, 0].plot(all_epochs_list, history_total.val_loss, label="validation")
    axes[0, 0].plot([epochs_phase1, epochs_phase1], axes[0, 0].get_ylim(), label="start phase 2")
    axes[0, 0].plot([start_phase3, start_phase3], axes[0, 0].get_ylim(), label="start phase 3")
    axes[0, 0].plot([start_phase4, start_phase4], axes[0, 0].get_ylim(), label="start phase 4")
    # axes[0].plot(history_vals.epoch, [baseline_loss] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    # axes[0].plot(history_vals.epoch, [random_loss] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    # axes[0].plot(history_vals.epoch, [test_loss] * len(history_vals.epoch), label="test", color="green")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Log loss")
    axes[0, 0].set_xticks(xticks)
    # axes[0, 0].set_xticklabels([all_epochs_list[i] for i in xticks])
    axes[0, 0].set_title("Phase 1 & 2: Train & Validation Loss")
    axes[0, 0].legend()

    # Phase 1: F1 Score
    axes[0, 1].plot(list(history_total.index + 1), history_total.f1_score, label="train")
    axes[0, 1].plot(list(history_total.index + 1), history_total.val_f1_score, label="validation")
    axes[0, 1].plot([epochs_phase1, epochs_phase1], axes[0, 1].get_ylim(), label="start phase 2")
    axes[0, 1].plot([start_phase3, start_phase3], axes[0, 1].get_ylim(), label="start phase 3")
    axes[0, 1].plot([start_phase4, start_phase4], axes[0, 1].get_ylim(), label="start phase 4")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("f1 score")
    axes[0, 1].set_xticks(xticks)
    # axes[0, 1].set_xticklabels([all_epochs_list[i] for i in xticks])
    axes[0, 1].set_title("Phase 1 & 2: Train & Validation F1-Score")
    axes[0, 1].legend()

    # Phase 1: Precision
    axes[1, 0].plot(list(history_total.index + 1), history_total.precision, label="train")
    axes[1, 0].plot(list(history_total.index + 1), history_total.val_precision, label="validation")
    axes[1, 0].plot([epochs_phase1, epochs_phase1], axes[1, 0].get_ylim(), label="start phase 2")
    axes[1, 0].plot([start_phase3, start_phase3], axes[1, 0].get_ylim(), label="start phase 3")
    axes[1, 0].plot([start_phase4, start_phase4], axes[1, 0].get_ylim(), label="start phase 4")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_xticks(xticks)
    # axes[1, 0].set_xticklabels([all_epochs_list[i] for i in xticks])
    axes[1, 0].set_title("Phase 1 & 2: Train & Validation Precision")
    axes[1, 0].legend()

    # Phase 1: Recall
    axes[1, 1].plot(list(history_total.index + 1), history_total.recall, label="train")
    axes[1, 1].plot(list(history_total.index + 1), history_total.val_recall, label="validation")
    axes[1, 1].plot([epochs_phase1, epochs_phase1], axes[1, 1].get_ylim(), label="start phase 2")
    axes[1, 1].plot([start_phase3, start_phase3], axes[1, 1].get_ylim(), label="start phase 3")
    axes[1, 1].plot([start_phase4, start_phase4], axes[1, 1].get_ylim(), label="start phase 4")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_xticks(xticks)
    # axes[1, 1].set_xticklabels([all_epochs_list[i] for i in xticks])
    axes[1, 1].set_title("Phase 1 & 2: Train & Validation Recall")
    axes[1, 1].legend()

    # Phase 1: Accuracy
    axes[2, 0].plot(list(history_total.index + 1), history_total.accuracy, label="train")
    axes[2, 0].plot(list(history_total.index + 1), history_total.val_accuracy, label="validation")
    axes[2, 0].plot([epochs_phase1, epochs_phase1], axes[2, 0].get_ylim(), label="start phase 2")
    axes[2, 0].plot([start_phase3, start_phase3], axes[2, 0].get_ylim(), label="start phase 3")
    axes[2, 0].plot([start_phase4, start_phase4], axes[2, 0].get_ylim(), label="start phase 4")
    axes[2, 0].set_xlabel("Epochs")
    axes[2, 0].set_ylabel("Accuracy")
    axes[2, 0].set_xticks(xticks)
    # axes[2, 0].set_xticklabels([all_epochs_list[i] for i in xticks])
    axes[2, 0].set_title("Phase 1 & 2: Train & Validation Accuracy")
    axes[2, 0].legend()

    # Phase 1: AUC
    axes[2, 1].plot(list(history_total.index + 1), history_total.auc, label="train")
    axes[2, 1].plot(list(history_total.index + 1), history_total.val_auc, label="validation")
    axes[2, 1].plot([epochs_phase1, epochs_phase1], axes[2, 1].get_ylim(), label="start phase 2")
    axes[2, 1].plot([start_phase3, start_phase3], axes[2, 1].get_ylim(), label="start phase 3")
    axes[2, 1].plot([start_phase4, start_phase4], axes[2, 1].get_ylim(), label="start phase 4")
    axes[2, 1].set_xlabel("Epochs")
    axes[2, 1].set_ylabel("AUC")
    axes[2, 1].set_xticks(xticks)
    # axes[2, 1].set_xticklabels([all_epochs_list[i] for i in xticks])
    axes[2, 1].set_title("Phase 1 & 2: Train & Validation AUC")
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()

