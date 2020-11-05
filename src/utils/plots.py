import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix0(y_true, y_pred,
                           classes=None,
                           normalize=False,
                           title=None,
                           save_to=None):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if classes is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(figsize=(32, 32), dpi=200)

    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)  # Spectral)
    axes.set_title(title)

    tick_marks = np.arange(len(classes))
    axes.set_xticks(tick_marks)
    axes.set_yticks(tick_marks)
    axes.set_xticklabels(classes, rotation=90)
    axes.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, '{:.02f}'.format(cm[i, j]),
                  fontsize='x-small',
                  horizontalalignment="center",
                  verticalalignment="center",
                  color="xkcd:midnight" if cm[i, j] > thresh else "white")

        if i == j:
            axes.add_patch(Rectangle((i - .5, j - .5), 1, 1, fill=False, edgecolor='black', lw=2))

    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')

    bottom, top = axes.get_ylim()
    axes.set_ylim(bottom + 0.5, top - 0.5)

    if save_to:
        plt.savefig(save_to)

    plt.close()

    return axes


def plot_confusion_matrix(y_true, y_pred,
                          classes=None,
                          normalize=False,
                          title=None,
                          save_to=None,
                          cmap=plt.cm.viridis):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if classes is None:
        classes = unique_labels(y_true, y_pred)
    # else:
    #     classes = classes[unique_labels(y_true, y_pred)]

    # classes = [c.split('__')[1] for c in classes]
    if normalize:
        cm = cm.astype('float')
        cm = np.divide(cm, cm.sum(axis=1)[:, np.newaxis], where=cm != 0)

    fig, ax = plt.subplots(figsize=(32, 32), dpi=200)
    plt.rcParams.update({'font.size': 36})

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    fontsize='x-small',
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="xkcd:midnight" if cm[i, j] > thresh else "white")

            if i == j:
                ax.add_patch(Rectangle((i - .5, j - .5), 1, 1, fill=False, edgecolor='black', lw=1))

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    fig.tight_layout()

    if save_to:
        plt.savefig(save_to)

    plt.close()

    return ax


if __name__ == '__main__':
    plot_confusion_matrix(
        np.random.randint(0, 10, size=1000),
        np.random.randint(0, 10, size=1000),
        save_to='/workspace/cm.png'
    )
