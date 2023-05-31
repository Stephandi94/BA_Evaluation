import os
import numpy as np
import csv
import evaluation_tools as et
import evaluation_visualizer as ev

def stats_to_csv(points, distances, intensities, title):
    cwd = os.getcwd()
    dest = os.path.join(cwd, "VLP32vsCARLA")
    output = [
        ["Points Per PCL", points[0], points[1], points[2]],
        ["Distances Mean Values", distances[0, 0], distances[0, 1], distances[0, 2]],
        ["Distances Min Values", distances[1, 0], distances[1, 1], distances[1, 2]],
        ["Distances Max Values", distances[2, 0], distances[2, 1], distances[2, 2]],
        ["Intensities Mean Values", intensities[0, 0], intensities[0, 1], intensities[0, 2]],
        ["Intensities Min Values", intensities[1, 0], intensities[1, 1], intensities[1, 2]],
        ["Intensities Max Values", intensities[2, 0], intensities[2, 1], intensities[2, 2]]
    ]
    with open(os.path.join(dest, "{}.csv".format(title)), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Measurement", "Mean", "Min", "Max"])
        writer.writerows(output)

def main():
    cwd = os.getcwd()
    src_carla_no_noise = os.path.join(r"..\Datasets\CARLA\5.2\no_noise\10-33-43.199828.bin") # 120 pcls
    dir_carla_noise = os.path.join(r"..\Datasets\CARLA\5.2\noise_no_dropoff") # 120 pcls
    src_vlp32_no_noise = os.path.join(r"..\Datasets\CARLA\5.0\VLP32\06-07-31.626189.bin") # GT
    dir_vlp32_noise = os.path.join(r"..\Datasets\CARLA\5.1\VLP32") # 120 pcls

    # CARLA noise analysis
    carla_noise = et.DatasetEvaluation(dir_carla_noise)
    carla_noise.set_gt(src_carla_no_noise)

    # 1. analyse points, distances, intensities
    #carla_noise_num_point_stats, carla_noise_num_points = carla_noise.compare_number_of_points()
    carla_noise_distance_stats, carla_noise_distances = carla_noise.analyze_distance()
    #carla_noise_intensity_stats, carla_noise_intensities = carla_noise.analyze_intensity()

    # 2. analyse deviation to gt
    #carla_noise_distance_gt_stats, carla_noise_distance_gt = carla_noise.analyze_distance_to_gt()
    #carla_noise_aug_stats, carla_noise_aug = carla_noise.evaluate_augmentation()


    # vlp32 noise
    vlp32_noise = et.DatasetEvaluation(dir_vlp32_noise)
    vlp32_noise.set_gt(src_vlp32_no_noise)

    # 1. analyse points, distances, intensities
    #vlp32_noise_num_point_stats, vlp32_noise_num_points = vlp32_noise.compare_number_of_points()
    vlp32_noise_distance_stats, vlp32_noise_distances = vlp32_noise.analyze_distance()
    #vlp32_noise_intensity_stats, vlp32_noise_intensities = vlp32_noise.analyze_intensity()

    # 2. analyse deviation to gt
    #vlp32_noise_distance_gt_stats, vlp32_noise_distances_gt = vlp32_noise.analyze_distance_to_gt()
    #vlp32_noise_aug_stats, vlp32_noise_aug = vlp32_noise.evaluate_augmentation()

    # to csv
    # stats_to_csv(carla_noise_num_point_stats, carla_noise_distance_stats, carla_noise_intensity_stats, "carla_noise")
    # stats_to_csv(vlp32_noise_num_point_stats, vlp32_noise_distance_stats, vlp32_noise_intensity_stats, "vlp32_noise")
    # ev.stats_3x3_to_csv(carla_noise_distance_gt_stats, r"VLP32vsCARLA/carla_distance_to_gt")
    # ev.stats_3x3_to_csv(vlp32_noise_distance_gt_stats, r"VLP32vsCARLA/vlp32_distance_to_gt")
    # ev.stats_3x3_to_csv(carla_noise_aug_stats, r"VLP32vsCARLA/carla_aug_to_gt", ["similar points", "lost points", "added points"])
    # ev.stats_3x3_to_csv(vlp32_noise_aug_stats, r"VLP32vsCARLA/vlp32_aug_to_gt", ["similar points", "lost points", "added points"])

    # plots
    # noise deviation in carla
    ev.plot_stats(
        carla_noise_distances,
        np.arange(0, carla_noise_distances.shape[0]),
        ["mean", "min", "max"],
        "Detection Distance CARLA LiDAR with Noise Compared to GT",
        "frame number",
        "distance [m]",
        os.path.join(cwd, r"VLP32vsCARLA/carla_distance"),
        True,
        et.get_distance_info(et.pcl_to_numpy(src_carla_no_noise))
    )

    # plots
    # noise deviation in vlp32
    ev.plot_stats(
        vlp32_noise_distances,
        np.arange(0, vlp32_noise_distances.shape[0]),
        ["mean", "min", "max"],
        "Detection Distance VLP32 LiDAR with Noise Compared to GT",
        "frame number",
        "distance [m]",
        os.path.join(cwd, r"VLP32vsCARLA/vlp32_distance"),
        True,
        et.get_distance_info(et.pcl_to_numpy(src_vlp32_no_noise))
    )





if __name__=='__main__':
    main()