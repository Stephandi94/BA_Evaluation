import pylab as p
from matplotlib import pyplot as plt
import numpy as np
import os
import csv

default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']

def print_stats(arr, title):
    """
    prints mean, min and max value to console
    :param arr: np.array([mean, min, max])
    :param title: some title
    """
    print("\n{}:\nmean: {}\nmin:  {}\nmax:  {}".format(title, arr[0], arr[1], arr[2]))

def stats_1x3_to_csv(arr, dest_title, col_label=["Mean", "Min", "Max"]):
    cwd = os.getcwd()
    with open(os.path.join(cwd, "{}.csv".format(dest_title)), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([col_label[0], col_label[1], col_label[2]])
        writer.writerow([arr[0], arr[1], arr[2]])

def stats_3x3_to_csv(arr, dest_title, row_label=["Mean Values", "Min Values", "Max Values"]):
    """
    prints 3x3 stats array to csv
    :param arr: np.array of size (3, 3)
    :param dest_title: subdirectory and filename [w/o extension]; e.g., "subdirectory/title"
    """
    cwd = os.getcwd()
    output = [
        [row_label[0], arr[0, 0], arr[0, 1], arr[0, 2]],
        [row_label[1], arr[1, 0], arr[1, 1], arr[1, 2]],
        [row_label[2], arr[2, 0], arr[2, 1], arr[2, 2]],
    ]
    with open(os.path.join(cwd, "{}.csv".format(dest_title)), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Mean", "Min", "Max"])
        writer.writerows(output)


def plot_scott_histogram(data, show_legend=True, title='title missing!', data_label='data_label missing!',
                         x_label='x_label missing!', save_fig=None):
    """
    histogram plot according to Scott's rule
    :param data: 1D np.array
    :param show_legend: default True
    :param title: string
    :param data_label: string
    :param x_label: string
    :param save_fig: source path incl. filename [w/o extension]; if no path given, figure will be shown
    """
    # data must be an 1D-array
    n_bins = int(np.round(min(np.sqrt(data.shape[0]), 20)))
    w_bin = 3.49 * data.std() / data.shape[0] ** (1 / 3)
    x_start = data.mean() - n_bins / 2 * w_bin
    x_end = data.mean() + n_bins / 2 * w_bin
    bins_opt = np.linspace(x_start, x_end, n_bins + 1)

    plt.figure(figsize=(10, 5))
    hist_opt = plt.hist(data, bins_opt, label=data_label, edgecolor='black', color='blue')

    x_outliers = data[np.logical_or(data > x_end, data < x_start)]
    y_outliers = np.ones_like(x_outliers)
    plt.plot(x_outliers, y_outliers, 'ro', label='outliers')
    plt.xlabel(x_label)
    plt.ylabel('abs. frequency')
    if show_legend:
        plt.legend(loc='upper right')
    #plt.title(title)
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig, dpi=600, bbox_inches='tight')


def plot_histogram_comparison(data_list, show_legend=True, title='title missing!', data_label='data_label missing!',
                         x_label='x_label missing!', bins=None, save_fig=None, colors=default_colors):
    """
    prints a histogram comparison
    :param data_list: list of 1D np.arrays
    :param show_legend: default True
    :param title: string
    :param data_label: list of strings
    :param x_label: string
    :param bins: list with bin parameters: [number_of_bins, x_start, x_end]
    :param save_fig: source path incl. filename [w/o extension]; if no path given, figure will be shown
    """
    if bins is None:
        max_l = 0
        max_i = 0
        for i, d in enumerate(data_list):
            if max_l < d.size:
                max_l = d.size
                max_i = i
        bins = [int(np.round(min(np.sqrt(max_l), 20)))]
        w_bin = 3.49 * data_list[max_i].std() / max_l ** (1 / 3)
        x_start = data_list[max_i].mean() - bins[0] / 2 * w_bin
        x_end = data_list[max_i].mean() + bins[0] / 2 * w_bin
    else:
        x_start = bins[1]
        x_end = bins[2]
    bins_opt = np.linspace(x_start, x_end, bins[0] + 1)

    plt.figure(figsize=(10, 5))
    for i in range(len(data_list)):
        weights = np.ones_like(data_list[i]) / float(data_list[i].size)
        plt.hist(data_list[i], bins_opt, weights=weights, label=data_label[i], edgecolor='black', color=colors[i], alpha=1/len(data_list))

    plt.xlabel(x_label)
    plt.ylabel('counts normalized')
    if show_legend:
        plt.legend(loc='upper right')
    #plt.title(title)
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig, dpi=600, bbox_inches='tight')


def box_plot(data, title='title missing!', data_labels='data_labels missing!',
             x_label='x_label missing!', y_label='y_label missing', whisker=[1, 99], save_fig=None):
    """
    box plot, can plot multiple boxes next to each other
    :param data: list containing 1D np.arrays [data1, data2, ...]
    :param title: string
    :param data_labels: list of strings, equals len(data)
    :param x_label: string
    :param y_label: string
    :param whisker: whisker interval, default [1, 99]
    :param save_fig: source path incl. filename [w/o extension]; if no path given, figure will be shown
    """
    plt.figure(figsize=(8, 4))
    flierprops = dict(marker='o', markerfacecolor='r', markersize=4, markeredgecolor='none')  # props of outliers
    plt.boxplot(data, labels=data_labels, whis=whisker, flierprops=flierprops)
    #plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig, dpi=600, bbox_inches='tight')


def violin_plot(data, title='title missing!', data_labels='data_labels missing!',
                x_label='x_label missing!', y_label='y_label missing', save_fig=None):
    """
    creates violin plot
    :param data: list containing 1D np.arrays [data1, data2, ...]
    :param title: string
    :param data_labels: list of strings, equals len(data)
    :param x_label: string
    :param y_label: string
    :param save_fig: source path incl. filename [w/o extension]; if no path given, figure will be shown
    """
    data_labels.insert(0, '')
    plt.figure(figsize=(10, 5))
    plt.violinplot(data, showmedians=True)
    #plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(len(data) + 1), data_labels)
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig, dpi=600, bbox_inches='tight')


def plot_stats(data, tick_label, data_label, title, x_label, y_label, save_fig=None, show_legend=True, gt_stats=None):
    """
    creates an overlay line plot for mean, min, max
    :param data: np.array of size (n, 3) containing [mean, min, max]
    :param tick_label: list of strings with length n
    :param data_label: list of strings with length 3
    :param title: string
    :param x_label: string
    :param y_label: string
    :param save_fig: source path incl. filename [w/o extension]; if no path given, figure will be shown
    :param show_legend: boolean default True
    """
    colors = ['crimson', 'cornflowerblue', 'royalblue']
    x_data = np.arange(data.shape[0])
    if len(data_label) != data.shape[1]:
        raise Exception("plot_stats ERROR: number of labels doesn't fit amount of data columns")
    plt.figure(figsize=(10, 5))
    if gt_stats is not None:
        gt_label = ["GT mean", "GT min", "GT max"]
        gt_data = np.ones(data.shape)
        gt_data[:, 0] *= gt_stats[0]
        gt_data[:, 1] *= gt_stats[1]
        gt_data[:, 2] *= gt_stats[2]
        for i in range(gt_data.shape[1]):
            plt.plot(x_data, gt_data[:, i], label=gt_label[i], color=colors[i], marker='None', linestyle='--')

    plt.fill_between(x_data, data[:, 1], data[:, 2], alpha=0.2)
    for i in range(data.shape[1]):
        plt.plot(x_data, data[:, i], label=data_label[i], color=colors[i], marker='None', linestyle='-')
    #plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show_legend:
        plt.legend(loc='upper right')
    plt.xticks(x_data, tick_label)
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig, dpi=600, bbox_inches='tight')


def line_plot(x_data, data_list, data_label, title, x_label, y_label, save_fig=None, legend_pos='upper right',
              tick_label=None):
    """
    creates line plot for multiple data
    :param x_data: 1D np.array x axis values
    :param data_list: list of np.array [data1, data2, ...] containing data
    :param data_label: labels list, has to be equally long as data_list
    :param title: string
    :param x_label: string
    :param y_label: string
    :param save_fig: source path incl. filename [w/o extension]; if no path given, figure will be shown
    :param legend_pos: default 'upper right'
    :param tick_label: in case x axis should be denoted otherwise
    :return:
    """
    plt.figure(figsize=(10, 5))
    for i, d in enumerate(data_list):
        plt.plot(x_data, d, label=data_label[i])
    #plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_pos)
    if tick_label is not None:
        plt.xticks(x_data, tick_label)
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig, dpi=600, bbox_inches='tight')

class VisEval:
    def __init__(self):
        self.y_data = None
        self.x_data = None
        self.x_label = None
        self.y_label = None
        self.x_tick_label = None
        self.y_tick_label = None
        self.colors = default_colors
        self.data_labels = None
        self.fig_size = (10, 5)
        self.legend_location = 'upper right'
        self.show_legend = True
        self.file_destination = None
        self.dpi = 600
        self.bins = None
        self.bar_width = 1
        self.legend_outside = False
        self.legend_frame = False
        self.marker = 'none'
        self.linestyle = '-'
        self.xlim = None
        self.ylim = None
        self.fontsize = 14
        self.labelsize = 12
        self.legend_col = 1

    def set_x_label(self, x_label):
        self.x_label = x_label

    def set_y_label(self, y_label):
        self.y_label = y_label

    def set_x_ticklabel(self, x_ticklabel):
        self.x_tick_label = x_ticklabel

    def set_y_ticklabel(self, y_ticklabel):
        self.y_tick_label = y_ticklabel

    def set_fontsize(self, size):
        self.fontsize = size

    def set_labelsize(self, size):
        self.labelsize = size

    def set_xlim(self, xlim):
        self.xlim = xlim

    def set_ylim(self, ylim):
        self.ylim = ylim

    def set_colors(self, color_list):
        self.colors = color_list

    def set_data_labels(self, data_label_list):
        self.data_labels = data_label_list

    def set_legend_location(self, legend_loc):
        self.legend_location = legend_loc

    def set_legend_frame(self, boolean):
        self.legend_frame = boolean

    def set_legend_col(self, n_col):
        self.legend_col = n_col

    def to_file(self, path, filename):
        """
        :param path: destination directory
        :param filename: file name w/o extension
        """
        self.file_destination = os.path.join(path, r"{}.png".format(filename))

    def set_dpi(self, dpi):
        self.dpi = dpi

    def set_figsize(self, figsize):
        """
        :param figsize: shape parameter (w, h)
        """
        self.fig_size = figsize

    def set_show_legend(self, boolean):
        self.show_legend = boolean

    def set_legend_outside(self, boolean):
        self.legend_outside = boolean

    def set_x_data(self, x_data_list):
        """
        set x_data
        :param x_data_list: [data1, data2, ...]
        """
        self.x_data = x_data_list

    def set_y_data(self, y_data_list):
        """
        set y_data
        :param y_data_list: [data1, data2, ...]
        """
        self.y_data = y_data_list

    def define_bins(self, n_bins, x_start, x_end):
        """
        customize histogram bins, otherwise Scott's rule applies
        """
        self.bins = np.linspace(x_start, x_end, n_bins + 1)

    def set_bar_width(self, width):
        self.bar_width = width

    def set_marker(self, marker):
        self.marker = marker

    def set_linestyle(self, style):
        self.linestyle = style

    #### PLOT FUNCTIONS BELOW ###

    def histogram(self, show_outlier=False, outlier_style='ro', outlier_label='outliers'):
        """
        prints a single histogram for each element in y_data
        :param show_outlier: only makes sense using Scott's rule
        :param outlier_style: default 'ro'
        :param outlier_label: string
        :return:
        """
        if len(self.y_data) == 0:
            raise Exception("no y_data!")
        if len(self.y_data) > 1:
            print("INFO: histogram files cannot be saved! call this function with a single dataset to do so!")
        for i, d in enumerate(self.y_data()):
            if self.bins is None:
                n_bins = int(np.round(min(np.sqrt(d.shape[0]), 20)))
                w_bin = 3.49 * d.std() / d.shape[0] ** (1 / 3)
                x_start = d.mean() - n_bins / 2 * w_bin
                x_end = d.mean() + n_bins / 2 * w_bin
                bins_opt = np.linspace(x_start, x_end, n_bins + 1)
            else:
                bins_opt = self.bins

            plt.figure(figsize=self.fig_size)
            plt.hist(d, bins_opt, label=self.data_labels[i], edgecolor='black', color=self.colors[0])

            if show_outlier:
                x_outliers = d[np.logical_or(d > bins_opt[0], d < bins_opt[-1])]
                y_outliers = np.ones_like(x_outliers)
                plt.plot(x_outliers, y_outliers, outlier_style, label=outlier_label)
            plt.xlabel(self.x_label, fontsize=self.fontsize)
            plt.ylabel(self.y_label, fontsize=self.fontsize)
            plt.xticks(fontsize=self.labelsize)
            plt.yticks(fontsize=self.labelsize)
            if self.show_legend:
                plt.legend(loc=self.legend_location, frameon=self.legend_frame, fontsize=self.fontsize)
            if self.file_destination is None or len(self.y_data) > 1:
                plt.show()
            else:
                plt.savefig(self.file_destination, dpi=self.dpi, bbox_inches='tight')

    def histogram_comparison(self, normalized=True):
        """
        compares y_data in a histogram plot
        :param normalized: default True (recommended)
        """
        if len(self.y_data) == 0:
            raise Exception("no y_data!")
        if self.bins is None:
            raise Exception("bins must be defined!")
        plt.figure(figsize=self.fig_size)
        for i, d in enumerate(self.y_data):
            if normalized:
                weights = np.ones_like(d) / float(d.size)
                plt.hist(d, self.bins, weights=weights, label=self.data_labels[i], edgecolor='black', color=self.colors[i],
                     alpha=1 / len(self.y_data))
            else:
                plt.hist(d, self.bins, label=self.data_labels[i], edgecolor='black', color=self.colors[i],
                     alpha=1 / len(self.y_data))
        plt.xlabel(self.x_label, fontsize=self.fontsize)
        plt.ylabel(self.y_label, fontsize=self.fontsize)
        plt.xticks(fontsize=self.labelsize)
        plt.yticks(fontsize=self.labelsize)
        if self.show_legend:
            plt.legend(loc=self.legend_location, frameon=self.legend_frame, fontsize=self.fontsize)
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.file_destination is None:
            plt.show()
        else:
            plt.savefig(self.file_destination, dpi=self.dpi, bbox_inches='tight')

    def line_plot(self):
        """
        assumes x_data is the same for each y_data
        """
        plt.figure(figsize=self.fig_size)
        for i, d in enumerate(self.y_data):
            plt.plot(self.x_data[0], d, label=self.data_labels[i], color=self.colors[i], marker=self.marker, linestyle=self.linestyle) #TODO irgendwann x_data pro y_data möglich machen
        plt.xlabel(self.x_label, fontsize=self.fontsize)
        plt.ylabel(self.y_label, fontsize=self.fontsize)
        plt.legend(loc=self.legend_location, ncol=self.legend_col, frameon=self.legend_frame, fontsize=self.fontsize)
        plt.xticks(fontsize=self.labelsize)
        plt.yticks(fontsize=self.labelsize)
        if self.x_tick_label is not None:
            plt.xticks(self.x_data[0], self.x_tick_label, fontsize=self.labelsize)
        if self.file_destination is None:
            plt.show()
        else:
            plt.savefig(self.file_destination, dpi=self.dpi, bbox_inches='tight')

    def line_plot_two_y_axis(self):
        fig, ax1 = p.subplots(figsize=self.fig_size)

        # plot augmented data
        y_data = self.y_data[0]
        ax1.plot(self.x_data[0], y_data[:, 0], label=self.data_labels[0], color=self.colors[0], marker='o', linestyle=self.linestyle[0])
        ax1.plot(self.x_data[0], y_data[:, 1], label=self.data_labels[1], color=self.colors[1], marker='o', linestyle=self.linestyle[1])
        ax1.set_xlabel(self.x_label, fontsize=self.fontsize)
        ax1.set_ylabel(self.y_label[0], fontsize=self.fontsize)
        ax1.legend(loc=self.legend_location[0], frameon=self.legend_frame, fontsize=self.fontsize)
        ax1.tick_params(labelsize=self.labelsize)
        if self.ylim is not None:
            ax1.set_ylim(self.ylim[0])

        y_data = self.y_data[1]
        ax2 = ax1.twinx()
        ax2.plot(self.x_data[0], y_data[:, 0], label=self.data_labels[2], color=self.colors[2], marker='o',
                 linestyle=self.linestyle[2], alpha=0.7)
        ax2.plot(self.x_data[0], y_data[:, 1], label=self.data_labels[3], color=self.colors[3], marker='o',
                 linestyle=self.linestyle[3], alpha=0.7)
        ax2.set_ylabel(self.y_label[1], fontsize=self.fontsize)
        ax2.legend(loc=self.legend_location[1], frameon=self.legend_frame, fontsize=self.fontsize)
        ax2.tick_params(labelsize=self.labelsize)
        if self.ylim is not None:
            ax2.set_ylim(self.ylim[1])

        if self.x_tick_label is not None:
            plt.xticks(self.x_data[0], self.x_tick_label, fontsize=self.labelsize)
        if self.file_destination is None:
            plt.show()
        else:
            plt.savefig(self.file_destination, dpi=self.dpi, bbox_inches='tight')

    def plot_stats(self):
        colors = ['crimson', 'cornflowerblue', 'royalblue']
        data = self.y_data[0]
        if len(self.data_labels) != data.shape[1]:
            raise Exception("plot_stats ERROR: number of labels doesn't fit amount of data columns")
        plt.figure(figsize=self.fig_size)
        plt.fill_between(self.x_data, data[:, 1], data[:, 2], alpha=0.2)
        for i in range(data.shape[1]):
            plt.plot(self.x_data, data[:, i], label=self.data_labels[i], color=colors[i], marker=self.marker, linestyle=self.linestyle)
        # plt.title(title)
        plt.xlabel(self.x_label, fontsize=self.fontsize)
        plt.ylabel(self.y_label, fontsize=self.fontsize)
        plt.xticks(fontsize=self.labelsize)
        plt.yticks(fontsize=self.labelsize)
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)
        if self.legend_outside:
            plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=self.legend_frame, fontsize=self.fontsize)
        else:
            plt.legend(loc=self.legend_location, frameon=self.legend_frame, fontsize=self.fontsize)
        if self.x_tick_label is not None:
            plt.xticks(self.x_data, self.x_tick_label, fontsize=self.labelsize)
        if self.file_destination is None:
            plt.show()
        else:
            plt.savefig(self.file_destination, dpi=self.dpi, bbox_inches='tight')

    def bar_plot(self, set_mean=True, mirror=False):
        x_data = self.x_data[0]
        fig, ax = plt.subplots(figsize=self.fig_size)
        for i, d in enumerate(self.y_data):
            if mirror and i == 1:
                d *= -1
                plt.axhline(color='black', lw=0.5)
            ax.bar(x_data, d, label=self.data_labels[i], color=self.colors[i], edgecolor=self.colors[i], alpha=1/len(self.y_data), width=self.bar_width, align='edge')
            if set_mean:
                mean = np.mean(d)
                ax.plot(x_data, np.ones_like(x_data) * mean, label="{} mean".format(self.data_labels[i]), color=self.colors[i], linestyle='--')
        plt.xlabel(self.x_label, fontsize=self.fontsize)
        plt.ylabel(self.y_label, fontsize=self.fontsize)
        plt.xticks(fontsize=self.labelsize)
        plt.yticks(fontsize=self.labelsize)
        ax.tick_params(labelsize=self.labelsize)
        step_size = x_data[-1] - x_data[-2]
        plt.xlim([x_data[0], x_data[-1] + 2 * step_size])
        if self.legend_outside:
            plt.legend(bbox_to_anchor=self.legend_location, borderaxespad=0, frameon=self.legend_frame, fontsize=self.fontsize)
        else:
            plt.legend(loc=self.legend_location, ncol=self.legend_col, frameon=self.legend_frame, fontsize=self.fontsize)
        if self.x_tick_label is not None:
            ax.set_xticks(self.x_data[0], self.x_tick_label)
        if self.y_tick_label is not None:
            ax.set_yticks(self.y_tick_label)
            ax.set_yticklabels(abs(self.y_tick_label))
        if self.file_destination is None:
            plt.show()
        else:
            plt.savefig(self.file_destination, dpi=self.dpi, bbox_inches='tight')

    def stat_sequence_plot(self):
        # data is [ (n, 3) np.array, (n, 3) np.array]
        plt.figure(figsize=self.fig_size)
        for i, d in enumerate(self.y_data):
            plt.plot(self.x_data[0], d, color=self.colors[i], linestyle='--', alpha=0.3) # dashed lines
            for j in range(d.shape[0]):
                min_point = (self.data[j], d[j, 1])
                max_point = (self.data[j], d[j, 2])
                plt.plot(min_point, max_point, color=self.colors[i], label="range min to max value", linestyle='-', marker='_', alpha=0.8) # line from min to max
            plt.plot(self.x_data[0], d, color=self.colors[i], linestyle='none', marker='o', label=self.data_labels[i])
        plt.xlabel(self.x_label, fontsize=self.fontsize)
        plt.ylabel(self.y_label, fontsize=self.fontsize)
        plt.xticks(fontsize=self.labelsize)
        plt.yticks(fontsize=self.labelsize)
        plt.legend(loc=self.legend_location, frameon=self.legend_frame, fontsize=self.fontsize)
        if self.x_tick_label is not None:
            plt.xticks(self.x_data[0], self.x_tick_label, fontsize=self.labelsize)
        if self.file_destination is None:
            plt.show()
        else:
            plt.savefig(self.file_destination, dpi=self.dpi, bbox_inches='tight')