import os
import evaluation_tools as et
import evaluation_visualizer as ev
import numpy as np

def main():
    src_vlp16_gt = os.path.join(r"../Datasets/CARLA/5.0/VLP16/06-11-54.834863.bin")
    src_vlp16_atm = os.path.join(r"../Datasets/CARLA/5.4/5.4.1/0/09-34-10.186099.bin") # static scene --> GT

    vlp16_gt = et.pcl_to_numpy(src_vlp16_gt)
    vlp16_atm = et.pcl_to_numpy(src_vlp16_atm)

    ev.stats_1x3_to_csv(et.get_distance_info(vlp16_atm), os.path.join("Weather_Clear/distance_stats"))
    ev.stats_1x3_to_csv(et.get_intensity_info(vlp16_atm), os.path.join("Weather_Clear/intensity_stats"))

    print("GT points: {}\natm points: {}".format(vlp16_gt.shape[0], vlp16_atm.shape[0]))

    gt_distances = et.get_distances(vlp16_gt)
    atm_distances = et.get_distances(vlp16_atm)

    distance_dev = et.get_compare_distances_to_gt(vlp16_gt, vlp16_atm)
    intensity_dev = et.get_compare_intensities_to_gt(vlp16_gt, vlp16_atm)
    ev.stats_1x3_to_csv(distance_dev, os.path.join(r"Weather_Clear/distance_deviation"))
    ev.stats_1x3_to_csv(intensity_dev, os.path.join(r"Weather_Clear/intensity_deviation"))

    ev.plot_histogram_comparison(
        [vlp16_gt[:, 3], vlp16_atm[:, 3]],
        True,
        "",
        ["w/o attenuation", "w attenuation"],
        "intensity",
        [100, 0, 1],
        os.path.join("Weather_Clear/clear_weather_intensity_attenuation")
    )

    ev.plot_histogram_comparison(
        [gt_distances, atm_distances],
        True,
        "",
        ["w/o attenuation", "w attenuation"],
        "distance [m]",
        [100, 0, 100],
        os.path.join("Weather_Clear/clear_weather_distance_attenuation")
    )

if __name__=='__main__':
    main()
