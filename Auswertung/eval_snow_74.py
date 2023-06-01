import numpy as np
import os
import evaluation_tools as et
import evaluation_visualizer as ev
import evaluation_data as ed

def main():
    """
    evaluation of snow at 21.0 mm/h (74% in CARLA)
    """
    sub_dir = 74 # TODO ANPASSEN
    dest_path = os.path.join(r"Weather_Snow/{}".format(74)) # TODO ANPASSEN

    src_vlp16_snow = os.path.join(r"../Datasets/CARLA/5.4/5.4.2/{}".format(sub_dir))
    src_vlp16_gt = os.path.join(r"../Datasets/CARLA/5.4/5.4.2/0/09-55-10.889485.bin")

    vlp16_snow = et.DatasetEvaluation(src_vlp16_snow)
    vlp16_snow.set_gt(src_vlp16_gt)

    # gt data
    vlp16_gt = et.pcl_to_numpy(src_vlp16_gt)
    gt_intensity_stats = et.get_intensity_info(vlp16_gt)
    gt_distance_info = et.get_distance_info(vlp16_gt)

    # get augmentation
    if os.path.exists(os.path.join(r"Weather_Snow/{}/augmentation_per_pcl.csv".format(sub_dir))):
        print("augmentation data already exists!")
        augmentation_stats_per_pcl = ed.read_csv_to_array(os.path.join(r"Weather_Snow/{}/augmentation_per_pcl.csv".format(sub_dir)), True)
        augmentation_stats = et.summarize_dataset_stats(augmentation_stats_per_pcl)
    else:
        print("augmentation data must be generated")
        augmentation_stats, augmentation_stats_per_pcl = vlp16_snow.evaluate_augmentation()
        ed.nd_array_to_csv(augmentation_stats_per_pcl, dest_path, "augmentation_per_pcl", ["similar", "lost", "noise"])
    ed.stats_3x3_to_csv(augmentation_stats, dest_path, "augmentation_stats", ["similar", "lost", "noise"],
                        ["mean", "min", "max"])

    # get intensity stats
    print("get intensity stats...")
    intensity_stats, intensity_stats_per_pcl = vlp16_snow.analyze_intensity()
    acc_intensities = vlp16_snow.accumulate_intensities()

    # get distance stats
    print("get distance stats...")
    distance_stats, distance_stats_per_pcl = vlp16_snow.analyze_distance()

    # get point stats
    print("get point stats...")
    point_stats, points_per_pcl = vlp16_snow.compare_number_of_points()

    # write stats to csv
    ed.stats_to_csv(point_stats, distance_stats, intensity_stats, dest_path, "snow_stats")

    # plot intensities
    intensity_vis = ev.VisEval()
    intensity_vis.set_y_data([acc_intensities, vlp16_gt[:, 3]])
    intensity_vis.set_x_label("intensity")
    intensity_vis.set_y_label("frequency")
    intensity_vis.set_data_labels(["21.0 mm/h snow", "no snow"]) # TODO REGENRATE ANPASSEN
    intensity_vis.define_bins(100, 0, 1)
    intensity_vis.set_legend_location('upper left')
    intensity_vis.to_file(dest_path, "snow_intensities")
    intensity_vis.set_xlim([0, 1.05])
    intensity_vis.histogram_comparison()

    # plot augmentation
    aug_vis = ev.VisEval()
    aug_vis.set_x_data([np.arange(augmentation_stats_per_pcl.shape[0])])
    aug_vis.set_y_data([augmentation_stats_per_pcl[:, 1], augmentation_stats_per_pcl[:, 2]])
    aug_vis.set_x_label("frames")
    aug_vis.set_y_label("# points")
    aug_vis.set_data_labels(["lost points", "noise"])
    aug_vis.set_y_ticklabel(np.arange(
        (np.max(augmentation_stats_per_pcl[:, 2]) + 97) * -1,
        np.max(augmentation_stats_per_pcl[:, 1]) + 526, 200
    ))
    aug_vis.set_legend_col(2)
    aug_vis.set_legend_location((0.43, 0.83))
    aug_vis.to_file(dest_path, "snow_augmenatation")
    aug_vis.bar_plot(True, True)

    # comparison physical weather model
    src_phys_snow = os.path.join(r"..\Datasets\BA_AdverseWeather\data\VLP16_snow_21.0.bin") # TODO REGENRATE ANPASSEN
    src_phys_gt = os.path.join(r"..\Datasets\BA_AdverseWeather\data\VLP16_clear_no-noise.bin")

    phys_snow = et.pcl_to_numpy(src_phys_snow)
    phys_gt = et.pcl_to_numpy(src_phys_gt)
    print("points physical GT: {}\npoints physical snow: {}".format(phys_gt.shape[0], phys_snow.shape[0]))

    phys_distance_stats = et.get_distance_info(phys_snow)
    phys_intensity_stats = et.get_intensity_info(phys_snow)
    phys_distance_stats_gt = et.get_compare_distances_to_gt(phys_gt, phys_snow)
    phys_intensity_stats_gt = et.get_compare_intensities_to_gt(phys_gt, phys_snow)
    phys_augmentation = et.get_augmentation_info(phys_gt, phys_snow)
    phys_distance_gt = et.get_distances(phys_gt)
    phys_distance = et.get_distances(phys_snow)

    ed.stats_1x3_to_csv(phys_distance_stats, dest_path, "snow_distance_stats_physical_{}".format(sub_dir))
    ed.stats_1x3_to_csv(phys_intensity_stats, dest_path, "snow_intensity_stats_physical_{}".format(sub_dir))
    ed.stats_1x3_to_csv(phys_distance_stats_gt, dest_path, "snow_distance_stats_GT_physical_{}".format(sub_dir))
    ed.stats_1x3_to_csv(phys_intensity_stats_gt, dest_path, "snow_intensity_stats_GT_physical_{}".format(sub_dir))
    ed.stats_1x3_to_csv(phys_augmentation, dest_path, "snow_augmentation_stats_physical_{}".format(sub_dir), ["similar", "lost", "noise"])

    # plot physical intensities
    phys_int = ev.VisEval()
    phys_int.set_y_data([phys_snow[:, 3], phys_gt[:, 3]])
    phys_int.set_x_label("intensity")
    phys_int.set_y_label("frequency")
    phys_int.set_data_labels(["21.0 mm/h snow", "no snow"])  # TODO REGENRATE ANPASSEN
    phys_int.define_bins(100, 0, 1)
    phys_int.set_legend_location('upper right')
    phys_int.to_file(dest_path, "phys_snow_intensities")
    phys_int.set_xlim([0, 1.05])
    phys_int.histogram_comparison()

    # plot physical distances
    phys_int = ev.VisEval()
    phys_int.set_y_data([phys_distance, phys_distance_gt])
    phys_int.set_x_label("distance [m]")
    phys_int.set_y_label("frequency")
    phys_int.set_data_labels(["21.0 mm/h snow", "no snow"])  # TODO REGENRATE ANPASSEN
    phys_int.define_bins(int(np.max(phys_distance_gt)), 0, int(np.max(phys_distance_gt)))
    phys_int.set_legend_location('upper right')
    phys_int.to_file(dest_path, "phys_snow_distances")
    phys_int.set_xlim([0, 102])
    phys_int.histogram_comparison()

if __name__=='__main__':
    main()