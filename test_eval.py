import os

import numpy as np
import evaluation_tools as et
import evaluation_visualizer as ev
from evaluation_tools import DatasetEvaluation as DatasetEval
from evaluation_tools import SeriesEvaluation as SeriesEval
from test_data import TestData


def main():
    path = os.path.join(r"c:\Users\steph\Desktop\BA\Datasets\CARLA\5.0\VLP32")
    test2 = DatasetEval(path)
    comp, num_arr = test2.compare_number_of_points()
    ev.print_stats(comp, "VLP32 points")

    # path = os.path.join(r"c:\Users\steph\Desktop\BA_Auswertung\CARLA\5.4\5.4.1\100")
    # gt_path = os.path.join(r"c:\Users\steph\Desktop\BA_Auswertung\CARLA\5.4\5.4.1\0\09-33-46.748212.bin")
    # test = DatasetEval(path)
    # test.set_gt(gt_path)
    # intensities = test.accumulate_intensities()
    # distances = test.accumulate_distances()
    # print(distances.shape)

    #num_of_points_stats, arr_points_per_file = test.compare_number_of_points()

    # distance_stats, distance_stats_per_pcl = test.analyze_distance()
    # ev.print_stats(distance_stats[0, :], "stats of means over dataset")
    # ev.print_stats(distance_stats[1, :], "stats of min vals over dataset")
    # ev.print_stats(distance_stats[2, :], "stats of max vals over dataset")

    # intensity_stats, intensity_stats_per_pcl = test.analyze_intensity()
    # ev.print_stats(intensity_stats[0, :], "stats of means over dataset")
    # ev.print_stats(intensity_stats[1, :], "stats of min vals over dataset")
    # ev.print_stats(intensity_stats[2, :], "stats of max vals over dataset")

    # distance_gt_stats, intensity_gt_stats_per_pcl = test.analyze_distance_to_gt()
    # ev.print_stats(distance_gt_stats[0, :], "stats of means over dataset to gt")
    # ev.print_stats(distance_gt_stats[1, :], "stats of min vals over dataset to gt")
    # ev.print_stats(distance_gt_stats[2, :], "stats of max vals over dataset to gt")

    # intensity_gt_stats, intensity_gt_stats_per_pcl = test.analyze_intensity_to_gt()
    # ev.print_stats(intensity_gt_stats[0, :], "stats of means over dataset to gt")
    # ev.print_stats(intensity_gt_stats[1, :], "stats of min vals over dataset to gt")
    # ev.print_stats(intensity_gt_stats[2, :], "stats of max vals over dataset to gt")

    # augmentation_stats, augmentation_stats_per_pcl = test.evaluate_augmentation()
    # ev.print_stats(augmentation_stats[0, :], "similar points")
    # ev.print_stats(augmentation_stats[1, :], "lost points")
    # ev.print_stats(augmentation_stats[2, :], "added noise")
    # print(augmentation_stats_per_pcl)

    # ev.print_stats(num_of_points_stats, "amount of points")
    # print(arr_points_per_file)

    # ==============================================================
    # bsp. detection range with decreasing fog visibility
    test_series = SeriesEval()

    # distances = []
    # ranges = [40001, 10001, 4001, 2001, 1001, 201, 51, 0]
    #
    # for r in ranges:
    #     c_path = os.path.join(r"c:\Users\steph\Desktop\BA_Auswertung\CARLA\5.4\5.4.3\{}".format(r))
    #     files = os.listdir(c_path)
    #     files.sort()
    #     pcl = et.pcl_to_numpy(os.path.join(c_path, files[0]))
    #     #distance_stats = et.get_distance_info(pcl)
    #     distances.append(et.get_distances(pcl))
    #     #test_series.add_data(distance_stats)
    #
    # # ev.box_plot(
    # #     distances,
    # #     "detection distance with decreasing meteorological visibility",
    # #     ['clear', '7000', '3000', '1500', '600', '125', '25', '0'],
    # #     "meteorological visibility [m]",
    # #     "detection range [m]"
    # # )
    # ev.violin_plot(
    #     distances,
    #     "detection distance with decreasing meteorological visibility",
    #     ['clear', '7000', '3000', '1500', '600', '125', '25', '0'],
    #     "meteorological visibility [m]",
    #     "detection range [m]"
    # )
    # ev.plot_stats(
    #     test_series.get_data_sequence_as_np_array(),
    #     ['clear', '7000', '3000', '1500', '600', '125', '25', '0'],
    #     ['mean', 'min', 'max'],
    #     "bla",
    #     "x",
    #     "y"
    # )



    # gt_file = os.path.join(r"c:\Users\steph\Desktop\BA_Auswertung\CARLA\5.4\5.4.1\0\09-33-47.305285.bin")
    # noise_file = os.path.join(r"c:\Users\steph\Desktop\BA_Auswertung\CARLA\5.4\5.4.1\74\09-25-34.697316.bin")
    #
    # dummy1 = np.array([
    #     [1, 1, 1, 1],
    #     [2, 2, 2, 1],
    #     [3, 3, 3, 1],
    #     [4, 4, 4, 1],
    #     [5, 5, 5, 1],
    #     [6, 6, 6, 1],
    #     [7, 7, 7, 1],
    #     [8, 8, 8, 1],
    #     [9, 9, 9, 1]
    # ])
    # dummy2 = np.array([
    #     [1, 1, 1, 1],
    #     [2, 2, 2, 1],
    #     [3, 3, 3, 1],
    #     [0, 0, 0, 1],
    #     [7, 7, 7, 1],
    #     [5, 5, 5, 1],
    #     [6, 6, 6, 1],
    #     [10, 10, 10, 1],
    #     [8, 8, 8, 1],
    #     [4, 4, 4, 1]
    #     #[9, 9, 9, 1]
    # ])
    #
    # gt_pcl = et.pcl_to_numpy(gt_file)
    # noise_pcl = et.pcl_to_numpy(noise_file)
    #
    # # ev.print_stats(et.get_distance_info(gt_pcl), "GT detection distance")
    # # ev.print_stats(et.get_intensity_info(gt_pcl), "GT detection intensity")
    #
    # ev.print_stats(et.get_compare_distances_to_gt(gt_pcl, noise_pcl), "distance deviation from noise to GT")
    # ev.print_stats(et.get_compare_intensities_to_gt(gt_pcl, noise_pcl), "intensity deviation from noise to GT")
    #
    # gt_indices, pcl_one_hot = et.get_differences_to_gt(gt_pcl, noise_pcl)
    # print(gt_indices)
    # print(pcl_one_hot)
    # dist_stats, int_stats, noise_data = et.analyze_noise(pcl_one_hot, noise_pcl)
    #
    # ev.print_stats(dist_stats, "noise detection distance")
    # ev.print_stats(int_stats, "noise detection intensity")
    # print("noise points:\n{}".format(noise_data))
    #
    # stats = et.compare_intensities_equal_sized(gt_indices, gt_pcl, noise_pcl)
    # print(
    #     "\nthere are {} equal points in this comparison.\n- so {} points have been lost\n- {} noise points are added".format(
    #         stats[-1], gt_pcl.shape[0] - stats[-1], noise_data.shape[0]))
    # ev.print_stats(stats[:3], "noise stats")


if __name__ == "__main__":
    main()
