import os
import sys

import numpy as np
from matplotlib import pyplot as plt


def get_distance_info(pcl):
    """
    determines the min, max detection distance as well as mean detection distance
    :param pcl: a single point cloud
    :return: np.array([mean, min, max])
    """
    distances = get_distances(pcl)
    return np.array([np.mean(distances), np.min(distances), np.max(distances)])


def get_intensity_info(pcl):
    """
    determines the min, max detection intensity as well as mean detection intensity
    :param pcl: a single point cloud
    :return: np.array([mean, min, max])
    """
    return np.array([np.mean(pcl[:, 3]), np.min(pcl[:, 3]), np.max(pcl[:, 3])])


def get_augmentation_info(gt, pcl):
    """
    determines augmentation of a point cloud wrt. GT
    :param gt: (n, 4) np.array GT point cloud
    :param pcl: (m, 4) np.array point cloud
    :return: (1, 3) np.array ([similar points, lost points, added noise])
    """
    gt_indices, pcl_one_hot = get_differences_to_gt(gt, pcl)
    num_lost = np.count_nonzero(gt_indices == -1)
    return np.array([gt_indices.size - num_lost, num_lost, np.count_nonzero(pcl_one_hot == 0)])

def get_distances(pcl):
    """
    computes the distance of the points to the origin
    :param pcl: a single point cloud
    :return: 1D np.array with distance at index i according to point pcl[i, :]
    """
    distances = np.zeros(pcl.shape[0], dtype=np.float32)
    for i in range(distances.size):
        sys.stdout.write("\r{} of {}".format(i + 1, distances.size))
        sys.stdout.flush()
        distances[i] = np.linalg.norm(pcl[i, :3])
    return distances

def summarize_dataset_stats(data):
    """
    returns mean min and max over stats dataset
    :param data: (n, 3) array with [mean, min, max] in each row
    :return: (3, 3) stats array ([mean_mean, min_mean, max_mean], [mean_min, min_min, max_min], [mean_max, min_max, max_max])
    """
    return np.array([
            [np.mean(data[:, 0]), np.min(data[:, 0]), np.max(data[:, 0])],  # mean values
            [np.mean(data[:, 1]), np.min(data[:, 1]), np.max(data[:, 1])],  # min values
            [np.mean(data[:, 2]), np.min(data[:, 2]), np.max(data[:, 2])],  # max values
        ])


def compare_intensities_equal_sized(index_array, gt, pcl):
    """
    computes for each point in an augmented pcl, occurring in gt as well, the intensity difference I_aug - I_gt
    :param index_array: 1D np.array containing for each point in gt the corresponding index in pcl
    :param gt: np.array of size (n, 4) containing ground truth points
    :param pcl: np.array of size (n, 4) containing augmented point cloud data
    :return: np.array([mean_diff, min_diff, max_diff, num_of_points])
    """
    differences_array = np.zeros(np.count_nonzero(index_array != -1))
    i = 0
    for g, p in enumerate(index_array):
        if p == -1:  # skip if in pcl is no point corresponding to gt point
            continue
        differences_array[i] = pcl[p, 3] - gt[g, 3]  # negative value means weaker intensity in pcl
        i += 1
    return np.array(
        [np.mean(differences_array), np.min(differences_array), np.max(differences_array), differences_array.size])


def analyze_noise(have_gt_point, pcl):
    """
    collects some distance and intensity information about noise and provides a noise point cloud
    :param have_gt_point: np.array one-hot vector if point occurs in gt
    :param pcl: np.array of size (n, 4) containing augmented point cloud data
    :return: np.array([mean_dist, min_dist, max_dist]),
    np.array([mean_intensity, min_intensity, max_intensity]),
    np.array of size (n, 4) containing noise data points
    """
    false_point_idx = np.where(have_gt_point == 0)[0]
    if false_point_idx.size == 0:
        print("\n\n=============")
        print("  NO NOISE!  ")
        print("=============\n")
        return np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0, 0])
    noise_points = np.zeros((false_point_idx.size, 4))
    for i, n in enumerate(false_point_idx):
        noise_points[i] = pcl[n, :4]
    return get_distance_info(noise_points), get_intensity_info(noise_points), noise_points


def get_compare_distances_to_gt(gt, pcl):
    """
    determines the distance deviation of an augmented point cloud to the ground truth
    :param gt: a single point cloud containing the ground truth
    :param pcl: a single point cloud containing augmented data
    :return: np.array([dev_mean, dev_min, dev_max])
    """
    gt_dist_info = get_distance_info(gt)
    pcl_dist_info = get_distance_info(pcl)
    return np.array([
        pcl_dist_info[0] - gt_dist_info[0],  # mean
        pcl_dist_info[1] - gt_dist_info[1],  # min
        pcl_dist_info[2] - gt_dist_info[2]  # max
    ])


def get_compare_intensities_to_gt(gt, pcl):
    """
    determines the intensity deviation of an augmented point cloud to the ground truth
    :param gt: a single point cloud containing the ground truth
    :param pcl: a single point cloud containing augmented data
    :return: np.array([dev_mean, dev_min, dev_max])
    """
    gt_intensity_info = get_intensity_info(gt)
    pcl_intensity_info = get_intensity_info(pcl)
    return np.array([
        pcl_intensity_info[0] - gt_intensity_info[0],  # mean
        pcl_intensity_info[1] - gt_intensity_info[1],  # min
        pcl_intensity_info[2] - gt_intensity_info[2]  # max
    ])


def get_differences_to_gt(gt, pcl):
    """
    returns an index array which denotes to each gt point a point in the pcl. if it doesn't exist the index is -1.
    also returns a one-hot vector for the pcl, which denotes for each point if it has a gt point
    :param gt: a single point cloud containing the ground truth
    :param pcl: a single point cloud containing augmented point cloud data
    :return: 1D np.array gt point cloud indices, 1D np.array pcl one-hot vector
    """
    gt_p = gt[:, :3]
    pcl_p = pcl[:, :3]
    pcl_idx_corresponding_to_gt = np.ones(gt.shape[0], dtype=int) * -1
    duplicates = []
    for i in range(gt.shape[0]):
        sys.stdout.write("\r{} of {}".format(i + 1, gt.shape[0]))
        sys.stdout.flush()
        idx_in_pcl = np.where((pcl_p == gt_p[i, :]).all(axis=1))[0]
        if idx_in_pcl.size > 0:
            pcl_idx_corresponding_to_gt[i] = idx_in_pcl[0]  # store corresponding pcl point index at gt index
        if idx_in_pcl.size > 1:
            print("\nsimilar points in pcl!\n")
            duplicates.append([i, idx_in_pcl[1:]])

    # TODO falls da mal was auftaucht, irgendwie verarbeiten
    if len(duplicates) > 0:
        print("\nduplicates: {}".format(duplicates))

    # create a one-hot vector for the pcl
    pcl_one_hot = np.zeros(pcl.shape[0])
    for i in pcl_idx_corresponding_to_gt:
        if i > -1:
            pcl_one_hot[i] = 1

    return pcl_idx_corresponding_to_gt, pcl_one_hot


def pcl_to_numpy(pcl_path, cols=4):
    """
    converts a bin file to numpy array
    :param pcl_path: system path to binary file
    :param cols: amount of columns (at least 4)
    :return:
    """
    data = np.fromfile(pcl_path, dtype=np.float32)
    return data.reshape((-1, cols))


class DatasetEvaluation:
    """
    to process datasets of point clouds
    """

    def __init__(self, src_path, pcl_cols=4):
        """
        constructor for evaluation of a dataset
        :param src_path:
        :param pcl_cols:
        """
        self.src_path = src_path
        self.pcl_cols = pcl_cols
        self.pcl_files = []
        self.gt = None

        files = os.listdir(os.path.join(self.src_path))
        files.sort()
        for f in files:
            self.pcl_files.append(os.path.join(self.src_path, f))
        print("\n{} pcl files have been registered!\n".format(len(self.pcl_files)))

    def set_gt(self, gt_path, cols=4):
        """
        define a ground truth
        :param gt_path: path to a single point cloud file
        :param cols: number of columns (default 4)
        """
        self.gt = pcl_to_numpy(gt_path, cols)

    def points_per_pcl(self):
        """
        counts the points for each pcl
        :return: 1D np.array
        """
        points_per_pcl = np.zeros(len(self.pcl_files))
        for i, f in enumerate(self.pcl_files):
            points_per_pcl[i] = pcl_to_numpy(f, self.pcl_cols).shape[0]
        return points_per_pcl

    def compare_number_of_points(self):
        """
        compares the amount of points within one dataset
        :return: np.array([mean, min, max]), 1D np.array with points per point cloud in ascending time order
        """
        points_per_pcl = self.points_per_pcl()
        return np.array([np.mean(points_per_pcl), np.min(points_per_pcl), np.max(points_per_pcl)]), points_per_pcl

    def analyze_distance(self):
        """
        compares distance stats over a complete dataset
        :return: np.array([
            [mean_mean, min_mean, max_mean],
            [mean_min, min_min, max_min],
            [mean_max, min_max, max_max]
        ]), np.array of size (n, 3) containing distance stats for each point
        """
        distance_stats_per_pcl = np.zeros((len(self.pcl_files), 3))
        for i, f in enumerate(self.pcl_files):
            distance_stats_per_pcl[i, :] = get_distance_info(pcl_to_numpy(f, self.pcl_cols))
        return summarize_dataset_stats(distance_stats_per_pcl), distance_stats_per_pcl

    def analyze_distance_to_gt(self):
        """
        compares distance stats to GT over complete dataset
        :return: np.array([
            [mean_mean, min_mean, max_mean],
            [mean_min, min_min, max_min],
            [mean_max, min_max, max_max]
        ]), np.array of size (n, 3) containing distance stats to GT for each point
        """
        if self.gt is None:
            raise Exception("analyze_distance_to_gt ERROR: no GT defined!")
        distance_stats_per_pcl = np.zeros((len(self.pcl_files), 3))
        for i, f in enumerate(self.pcl_files):
            distance_stats_per_pcl[i, :] = get_compare_distances_to_gt(self.gt, pcl_to_numpy(f, self.pcl_cols))
        return summarize_dataset_stats(distance_stats_per_pcl), distance_stats_per_pcl

    def analyze_intensity(self):
        """
        compares intensity stats over a complete dataset
        :return: np.array([
            [mean_mean, min_mean, max_mean],
            [mean_min, min_min, max_min],
            [mean_max, min_max, max_max]
        ]), np.array of size (n, 3) containing intensity stats for each point
        """
        intensity_stats_per_pcl = np.zeros((len(self.pcl_files), 3))
        for i, f in enumerate(self.pcl_files):
            intensity_stats_per_pcl[i, :] = get_intensity_info(pcl_to_numpy(f, self.pcl_cols))
        return summarize_dataset_stats(intensity_stats_per_pcl), intensity_stats_per_pcl

    def analyze_intensity_to_gt(self):
        """
        compares intensity stats to GT over complete dataset
        :return: np.array([
            [mean_mean, min_mean, max_mean],
            [mean_min, min_min, max_min],
            [mean_max, min_max, max_max]
        ]), np.array of size (n, 3) containing intensity stats to GT for each point
        """
        if self.gt is None:
            raise Exception("analyze_distance_to_gt ERROR: no GT defined!")
        intensity_stats_per_pcl = np.zeros((len(self.pcl_files), 3))
        for i, f in enumerate(self.pcl_files):
            intensity_stats_per_pcl[i, :] = get_compare_intensities_to_gt(self.gt, pcl_to_numpy(f, self.pcl_cols))
        return summarize_dataset_stats(intensity_stats_per_pcl), intensity_stats_per_pcl

    def evaluate_augmentation(self):
        """
        compares similar and lost points as well as added noise over the dataset
        :return: np.array([
            [mean_similar_points, min_similar_points, max_similar_points],
            [mean_lost_points, min_lost_points, max_lost_points],
            [mean_added_noise, min_added_noise, max_added_noise]
        ]), np.array of size (n, 3) containing augmentation stats for each point
        """
        if self.gt is None:
            raise Exception("evaluate_augmentation ERROR: no GT defined!")
        augmentation = np.zeros((len(self.pcl_files), 3))   # similar points, lost points, noise
        for i, f in enumerate(self.pcl_files):
            print("\nevaluate file {} of {}\n".format(i+1, len(self.pcl_files)))
            gt_indices, pcl_one_hot = get_differences_to_gt(self.gt, pcl_to_numpy(f, self.pcl_cols))
            num_lost = np.count_nonzero(gt_indices == -1)
            augmentation[i, :] = np.array([gt_indices.size - num_lost, num_lost, np.count_nonzero(pcl_one_hot == 0)])
        return summarize_dataset_stats(augmentation), augmentation

    def accumulate_distances(self):
        """
        concatenates all distance values of the dataset
        :return: 1D np.array
        """
        distances_in_dataset = np.array([])

        for i, f in enumerate(self.pcl_files):
            print("\nadd pcl {} of {}".format(i+1, len(self.pcl_files)))
            distances_in_dataset = np.concatenate((distances_in_dataset, get_distances(pcl_to_numpy(f, self.pcl_cols))))
        return distances_in_dataset

    def accumulate_intensities(self):
        """
        concatenates all intensity values of the dataset
        :return: 1D np.array
        """
        intensities_in_dataset = np.array([])

        for i, f in enumerate(self.pcl_files):
            print("add pcl {} of {}".format(i + 1, len(self.pcl_files)))
            intensities_in_dataset = np.concatenate((intensities_in_dataset, pcl_to_numpy(f, self.pcl_cols)[:, 3]))
        return intensities_in_dataset




class SeriesEvaluation:
    def __init__(self):
        self.data_sequence = []

    def add_data(self, data):
        if len(self.data_sequence) > 0:
            if self.data_sequence[-1].shape != data.shape:
                raise Exception("add_data ERROR: added data fits not the snape of data in sequence")
        self.data_sequence.append(data)

    def get_data_sequence_as_np_array(self):
        return np.array(self.data_sequence)




    # TODO ergebnisse mit ändernden parametern auftragen (e.g. detektionsleistung mit sinkender sichtweite)
