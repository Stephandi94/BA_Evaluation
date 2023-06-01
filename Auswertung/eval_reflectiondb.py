import os
import math

import numpy as np

import evaluation_tools as et
import evaluation_visualizer as ev

def main():
    cwd = os.getcwd()
    # GT
    src_vlp32_gt = os.path.join(r"../Datasets/CARLA/5.3/special_scene/gt/VLP32/23-56-15.936035.bin")
    vlp32_gt = et.pcl_to_numpy(src_vlp32_gt)
    ev.stats_1x3_to_csv(et.get_intensity_info(vlp32_gt), r"ReflectionDatabase/gt_intensity_stats")
    print("num detections in GT: {}".format(vlp32_gt.shape[0]))

    dir_vlp32_db = os.path.join(r"../Datasets/CARLA/5.3/special_scene/no_noise/VLP32")

    # how strong is the intensity deviation in a measurement row
    db = et.DatasetEvaluation(dir_vlp32_db)
    db.set_gt(src_vlp32_gt)
    db_intensity_stats, db_intensity = db.analyze_intensity()
    db_intensity_stats_gt, db_intensity_gt = db.analyze_intensity_to_gt()
    db_num_points_stats, db_num_points = db.compare_number_of_points()
    ev.stats_1x3_to_csv(db_num_points_stats, r"ReflectionDatabase/db_num_points")

    # plot normalized histogram comparison
    acc_intensities = db.accumulate_intensities()
    gt_intensities = vlp32_gt[:, 3]
    # ev.plot_histogram_comparison(
    #     [gt_intensities, acc_intensities],
    #     True,
    #     "Intensity Distribution Comparison with and without Reflection Database",
    #     ["w/o reflection database", "w reflection database"],
    #     "intensities",
    #     100,
    #     os.path.join(r"ReflectionDatabase/intensity_comparison"),
    #     ['#1f77b4', '#ff7f0e']
    # )

    vis = ev.VisEval()
    vis.set_y_data([gt_intensities, acc_intensities])
    vis.set_data_labels(["w/o reflection database", "w reflection database"])
    vis.set_x_label("intensities")
    vis.set_y_label("frequency")
    vis.define_bins(100, 0, 1)
    vis.set_xlim([0, 1.05])
    vis.set_legend_location('upper left')
    vis.set_colors(['#1f77b4', '#ff7f0e'])
    vis.to_file(r"ReflectionDatabase", "intensity_comparison")
    vis.histogram_comparison()



    acc_int_wo_theshold = (acc_intensities * ((10 * math.log10(3444 * 9)) - (-70)) + (-70)) / (10 * math.log10(3444 * 9))
    ev.stats_1x3_to_csv(
        [np.mean(acc_int_wo_theshold), np.min(acc_int_wo_theshold), np.max(acc_int_wo_theshold)],
        os.path.join(r"ReflectionDatabase/custom_reflections_wo_threshold")
    )

    # ev.plot_histogram_comparison(
    #     [acc_intensities, acc_int_wo_theshold],
    #     True,
    #     "Intensity Distribution Depending on Sensor Intensity Threshold",
    #     ["with threshold", "without threshold"],
    #     "intensities",
    #     100,
    #     os.path.join(r"ReflectionDatabase/intensity_comparison_threshold"),
    #     ['#ff7f0e', '#2ca02c']
    # )

    vis = ev.VisEval()
    vis.set_y_data([acc_intensities, acc_int_wo_theshold])
    vis.set_data_labels(["with threshold", "without threshold"])
    vis.set_x_label("intensities")
    vis.set_y_label("frequency")
    vis.define_bins(100, 0, 1)
    vis.set_xlim([0, 1.05])
    vis.set_legend_location('upper left')
    vis.set_colors(['#ff7f0e', '#2ca02c'])
    vis.to_file(r"ReflectionDatabase", "intensity_comparison_threshold")
    vis.histogram_comparison()

    ev.stats_3x3_to_csv(db_intensity_stats, os.path.join(r"ReflectionDatabase/custom_reflections"))
    ev.stats_3x3_to_csv(db_intensity_stats_gt, os.path.join(r"ReflectionDatabase/custom_reflections_gt_dev"))

if __name__=='__main__':
    main()