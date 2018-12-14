import matplotlib.pyplot as plt
"""
Temporary script to monitor loss history retuned by keras fit_generator
"""

CMAP = plt.cm.Dark2.colors
FIGSCALE = 7
FONTSIZE = 14


def plot_loss(self, logs_history, dump_path, cmap=CMAP, fig_scale=FIGSCALE, fontsize=FONTSIZE):
    """
    Saves plot of loss history at specified path
    Args:
        logs_history (object): logs.history object returned by training
        dump_path (str): directory in which figured should be saved
    """
    nr_losses = len(logs_history)
    fig_scale = 7
    fig, ax = plt.subplots(1, nr_losses, figsize=(nr_losses * fig_scale, fig_scale - 2))
    epochs = range(1, len(logs_history['loss']) + 1)

    for i, loss in enumerate(logs_history.values()):
        ax[i].set_xlabel("epochs", fontsize=fontsize)
        ax[i].set_ylabel(loss[0], fontsize=fontsize)
        ax[i].plot(epochs, loss[1], '--o', color=CMAP[i % nr_losses])
        ax[i].tick_params(axis='both', labelsize=fontsize)
        ax[i].set_xticks(epochs)
        ax[i].grid(alpha=0.3)
        ax[i].set_title(loss[0], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(dump_path)
    plt.close()
