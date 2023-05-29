from matplotlib import pyplot as plt
import numpy as np


def print_stats(arr, title):
    """
    prints mean, min and max value to console
    :param arr: np.array([mean, min, max])
    :param title: some title
    """
    print("\n{}:\nmean: {}\nmin:  {}\nmax:  {}".format(title, arr[0], arr[1], arr[2]))


def plot_scott_histogram(data, show_legend=True, title='title missing!', data_label='data_label missing!',
                         x_label='x_label missing!'):
    """
    histogram plot according to Scott's rule
    :param data: 1D np.array
    :param show_legend: default True
    :param title: string
    :param data_label: string
    :param x_label: string
    """
    # data must be an 1D-array
    n_bins = int(np.round(min(np.sqrt(data.shape[0]), 20)))
    w_bin = 3.49 * data.std() / data.shape[0] ** (1 / 3)
    x_start = data.mean() - n_bins / 2 * w_bin
    x_end = data.mean() + n_bins / 2 * w_bin
    bins_opt = np.linspace(x_start, x_end, n_bins + 1)

    plt.figure(figsize=(8, 4))
    hist_opt = plt.hist(data, bins_opt, label=data_label, edgecolor='black', color='blue')

    x_outliers = data[np.logical_or(data > x_end, data < x_start)]
    y_outliers = np.ones_like(x_outliers)
    plt.plot(x_outliers, y_outliers, 'ro', label='outliers')
    plt.xlabel(x_label)
    plt.ylabel('abs. frequency')
    if show_legend:
        plt.legend(loc='upper right')
    plt.title(title)
    plt.show()


def box_plot(data, title='title missing!', data_labels='data_labels missing!',
             x_label='x_label missing!', y_label='y_label missing', whisker=[1, 99]):
    """
    box plot, can plot multiple boxes next to each other
    :param data: list containing 1D np.arrays [data1, data2, ...]
    :param title: string
    :param data_labels: list of strings, equals len(data)
    :param x_label: string
    :param y_label: string
    :param whisker: whisker interval, default [1, 99]
    """
    plt.figure(figsize=(8, 4))
    flierprops = dict(marker='o', markerfacecolor='r', markersize=4, markeredgecolor='none')  # props of outliers
    plt.boxplot(data, labels=data_labels, whis=whisker, flierprops=flierprops)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def violin_plot(data, title='title missing!', data_labels='data_labels missing!',
                x_label='x_label missing!', y_label='y_label missing'):
    """
    creates violin plot
    :param data: list containing 1D np.arrays [data1, data2, ...]
    :param title: string
    :param data_labels: list of strings, equals len(data)
    :param x_label: string
    :param y_label: string
    """
    data_labels.insert(0, '')
    plt.figure(figsize=(8, 4))
    plt.violinplot(data, showmedians=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(len(data) + 1), data_labels)
    plt.show()


def plot_stats(data, tick_label, data_label, title, x_label, y_label, show_legend=True):
    """
    creates an overlay line plot for mean, min, max
    :param data: np.array of size (n, 3) containing [mean, min, max]
    :param tick_label: list of strings with length n
    :param data_label: list of strings with length 3
    :param title: string
    :param x_label: string
    :param y_label: string
    :param show_legend: boolean default True
    """
    colors = ['crimson', 'cornflowerblue', 'royalblue']
    x_data = np.arange(data.shape[0])
    if len(data_label) != data.shape[1]:
        raise Exception("plot_stats ERROR: number of labels doesn't fit amount of data columns")
    plt.figure(figsize=(8, 4))
    plt.fill_between(x_data, data[:, 1], data[:, 2], alpha=0.2)
    for i in range(data.shape[1]):
        plt.plot(x_data, data[:, i], label=data_label[i], color=colors[i], marker='None', linestyle='-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show_legend:
        plt.legend(loc='upper right')
    plt.xticks(x_data, tick_label)
    plt.show()


def line_plot(x_data, data_list, data_label, title, x_label, y_label, save_fig=None, tick_label=None):
    plt.figure(figsize=(10, 5))
    for i, d in enumerate(data_list):
        plt.plot(x_data, d, label=data_label[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    if tick_label is not None:
        plt.xticks(x_data, tick_label)
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig, dpi=600, bbox_inches='tight')
