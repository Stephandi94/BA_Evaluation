import os
import csv
import numpy as np

import evaluation_tools as et
import evaluation_visualizer as ev
import evaluation_data as ed
from matplotlib import pyplot as plt


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
    # ev.line_plot(
    #     np.arange(0, vlp16_num_points.size),
    #     [vlp16_num_points, vlp32_num_points, carla_num_points],
    #     ["VLP16", "VLP32", "CARLA"],
    #     "Number of Detected Points in Static Scene",
    #     "frame number",
    #     "# points",
    #     os.path.join(dest, "determinism_points"),
    #     'center right'
    # )

    vis = ev.VisEval()
    vis.set_x_data([np.arange(0, vlp16_num_points.size)])
    vis.set_y_data([vlp16_num_points, vlp32_num_points, carla_num_points])
    vis.set_data_labels(["VLP16", "VLP32", "CARLA"])
    vis.set_x_label("frame number")
    vis.set_y_label("# points")
    vis.to_file(dest, "determinism_points")
    vis.set_legend_location('lower right')
    vis.set_xlim([0, vlp16_num_points.size])
    vis.set_ylim([0, np.max(carla_num_points) + 1000])
    vis.line_plot()

    vis = ev.VisEval()
    vis.set_x_data([np.arange(0, vlp16_num_points.size)])
    vis.set_y_data([vlp16_num_points, vlp32_num_points])
    vis.set_data_labels(["VLP16", "VLP32"])
    vis.set_x_label("frame number")
    vis.set_y_label("# points")
    vis.to_file(dest, "determinism_points_vlp")
    vis.set_legend_location('lower right')
    vis.set_xlim([0, vlp16_num_points.size])
    vis.set_ylim([0, np.max(vlp32_num_points) + 1000])
    vis.line_plot()


def det_plot():
    carla_src = os.path.join(r"Determinism/determinism_carla.csv")
    vlp16_src = os.path.join(r"Determinism/determinism_vlp16.csv")
    vlp32_src = os.path.join(r"Determinism/determinism_vlp32.csv")
    dest = os.path.join(r"Determinism")

    carla = ed.read_csv_to_array(carla_src, True, True)
    vlp16 = ed.read_csv_to_array(vlp16_src, True, True)
    vlp32 = ed.read_csv_to_array(vlp32_src, True, True)

    y_data = [carla, vlp16, vlp32]
    x_data = np.arange(120)
    label = ["CARLA LiDAR", "VLP16", "VLP32"]
    colors = ['crimson', 'cornflowerblue', 'royalblue']



    for i, d in enumerate(y_data):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        # distances
        ax1.plot(x_data, np.ones_like(x_data) * d[1, 0], label="mean", linestyle='-', color=colors[0], marker='none')
        ax1.plot(x_data, np.ones_like(x_data) * d[2, 1], label="min", linestyle='--', color=colors[1], marker='none')
        ax1.plot(x_data, np.ones_like(x_data) * d[3, 2], label="max", linestyle='--', color=colors[2], marker='none')
        ax1.set_xlabel("frame", fontsize=14)
        ax1.set_ylabel("distance [m]", fontsize=14)
        ax1.tick_params(labelsize=12)
        ax1.set_xlim([0, 120])
        ax1.set_ylim([0, 103])
        ax1.legend(loc='best', frameon=False, fontsize=14)

        # intensities
        ax2.plot(x_data, np.ones_like(x_data) * d[4, 0], label="mean", linestyle='-', color=colors[0], marker='none')
        ax2.plot(x_data, np.ones_like(x_data) * d[5, 1], label="min", linestyle='--', color=colors[1], marker='none')
        ax2.plot(x_data, np.ones_like(x_data) * d[6, 2], label="max", linestyle='--', color=colors[2], marker='none')
        ax2.set_xlabel("frame", fontsize=14)
        ax2.set_ylabel("intensity", fontsize=14)
        ax2.tick_params(labelsize=12)
        ax2.set_xlim([0, 120])
        ax2.set_ylim([0, 1.03])
        ax2.legend(loc='best', frameon=False, fontsize=14)

        # points
        ax3.plot(x_data, np.ones_like(x_data) * d[0, 0], label="", linestyle='-', color=colors[0], marker='none')
        ax3.set_xlabel("frame", fontsize=14)
        ax3.set_ylabel("# detected points", fontsize=14)
        ax3.tick_params(labelsize=12)
        ax3.set_xlim([0, 120])
        ax3.set_ylim([0, 55500])
        #ax3.legend(loc='best', frameon=False, fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(dest, "eval_{}_determinism.png".format(label[i])), dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    #main()
    det_plot()