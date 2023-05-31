import os
import csv
import numpy as np

import evaluation_tools as et
import evaluation_visualizer as ev


def stats_to_csv(points, distances, intensities, title):
    cwd = os.getcwd()
    dest = os.path.join(cwd, "Determinism")
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
    dest = os.path.join(cwd, "Determinism")

    path_vlp16 = os.path.join("../Verwendet_Fuer_Auswertung/5.0/VLP16")
    path_vlp32 = os.path.join("../Verwendet_Fuer_Auswertung/5.0/VLP32")
    path_carla = os.path.join("../Datasets/CARLA/5.2/no_noise")

    vlp16 = et.DatasetEvaluation(path_vlp16)
    vlp32 = et.DatasetEvaluation(path_vlp32)
    carla = et.DatasetEvaluation(path_carla)

    # VLP16 analysis
    vlp16_num_point_stats, vlp16_num_points = vlp16.compare_number_of_points()
    vlp16_distance_stats, vlp16_distances = vlp16.analyze_distance()
    vlp16_intensity_stats, vlp16_intensities = vlp16.analyze_intensity()

    stats_to_csv(vlp16_num_point_stats, vlp16_distance_stats, vlp16_intensity_stats, "determinism_vlp16")

    # VLP32 analysis
    vlp32_num_point_stats, vlp32_num_points = vlp32.compare_number_of_points()
    vlp32_distance_stats, vlp32_distances = vlp32.analyze_distance()
    vlp32_intensity_stats, vlp32_intensities = vlp32.analyze_intensity()

    stats_to_csv(vlp32_num_point_stats, vlp32_distance_stats, vlp32_intensity_stats, "determinism_vlp32")

    # VLP32 analysis
    carla_num_point_stats, carla_num_points = carla.compare_number_of_points()
    carla_distance_stats, carla_distances = carla.analyze_distance()
    carla_intensity_stats, carla_intensities = carla.analyze_intensity()

    stats_to_csv(carla_num_point_stats, carla_distance_stats, carla_intensity_stats, "determinism_carla")

    # plot point distribution over frames
    ev.line_plot(
        np.arange(0, vlp16_num_points.size),
        [vlp16_num_points, vlp32_num_points, carla_num_points],
        ["VLP16", "VLP32", "CARLA"],
        "Number of Detected Points in Static Scene",
        "frame number",
        "# points",
        os.path.join(dest, "determinism_points"),
        'center right'
    )


if __name__ == '__main__':
    main()
