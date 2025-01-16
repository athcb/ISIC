import pandas as pd
from matplotlib import pyplot as plt

def create_history_plots(input_history_vals):
    history_vals = pd.read_csv(input_history_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(history_vals.epoch, history_vals.loss, label="train")
    axes[0].plot(history_vals.epoch, history_vals.val_loss, label="validation")
    # axes[0].plot(history_vals.epoch, [baseline_loss] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    # axes[0].plot(history_vals.epoch, [random_loss] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    #axes[0].plot(history_vals.epoch, [test_loss] * len(history_vals.epoch), label="test", color="green")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Train & Validation Loss")
    axes[0].legend()
    # Plot the MAE on the training set and validation set
    axes[1].plot(history_vals.epoch, history_vals.f1_score, label="train")
    axes[1].plot(history_vals.epoch, history_vals.val_f1_score, label="validation")
    # axes[1].plot(history_vals.epoch, [baseline_accuracy] * len(history_vals.epoch), label="baseline", color="red", linestyle="--")
    # axes[1].plot(history_vals.epoch, [random_accuracy] * len(history_vals.epoch), label="random", color="orange", linestyle="-.")
    #axes[1].plot(history_vals.epoch, [test_f1_score] * len(history_vals.epoch), label="test", color="green")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("f1 score")
    axes[1].set_title("Train & Validation F1-Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

