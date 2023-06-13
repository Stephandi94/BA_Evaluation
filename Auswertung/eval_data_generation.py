import os
import numpy as np
import evaluation_tools as et
import evaluation_data as ed
import evaluation_visualizer as ev
from matplotlib import rc

def comparison_carla_vlp32():
    src = os.path.join(r"..\Auswertung_Materialien\carla_vs_vlp32\pcl_for_fig")
    dest = os.path.join(r"VLP32vsCARLA")
    names = ["carla", "vlp32"]
    data = np.zeros((5, 2)) # em. points, det. points, loss, min dist, max dist
    data[0, 0] = 1157380 / 20
    data[0, 1] = 32 * (360 / 0.2)

    for i, n in enumerate(names):
        pcl = et.pcl_to_numpy(os.path.join(src, "5-2_{}_lidar_noise.bin".format(n)))
        data[1, i] = pcl.shape[0]
        data[2, i] = 1 - (data[1, i] / data[0, i])
        stats = et.get_distance_info(pcl)
        data[3, i] = stats[1]
        data[4, i] = stats[2]

    print(data)
    ed.nd_array_to_csv(data, dest, "sensor_comparison", names, ["em.points", "det. points", "loss", "min dist", "max dist"])

def kitti_stats():
    src = os.path.join(r"../Auswertung_Materialien/real_world_pcls/KITTI_01.bin")
    dest = os.path.join(r"ReflectionDatabase")
    pcl = et.pcl_to_numpy(src)
    stats = et.get_intensity_info(pcl)
    ed.stats_1x3_to_csv(stats, dest, "kitty_intensities")

def formula(i, t):
    mp = 10 * np.log10(3444 * 9)
    ep = mp - 10 * np.log10(9)
    return (i * mp - t) / (mp - t)

def fzd_intensity_spectra():
    dest = os.path.join(r"ReflectionDatabase")
    #x_data = np.linspace(0, 1, 0.1)
    data = np.zeros((3, 2)) #vlp16, vlp32, vlp16 wo, vlp32 wo
    data_labels = ["VLP16", "VLP32", "VLP16 & VLP32 w/o thresh"]

    for i in range(2):
        data[0, i] = formula(i, -48.8)
        data[1, i] = formula(i, -70.0)
        data[2, i] = formula(i, 0)
    #ed.nd_array_to_csv(data, dest, "min_max_intensities", ["0", "1"], data_labels)

    vis = ev.VisEval()
    vis.set_y_data([data[0, :], data[1, :], data[2, :]])
    vis.set_x_data([np.array([0, 1])])
    vis.set_x_label(r"$P_0 \cdot \rho \cdot \cos(\theta)$")
    vis.set_y_label("intensity")
    vis.set_data_labels(data_labels)
    vis.set_xlim([0, 1])
    vis.set_ylim([0, 1.1])
    vis.set_legend_location('lower right')
    vis.set_colors(['#1f77b4', '#ff7f0e', '#d62728'])
    vis.to_file(dest, "intensities_w_and_wo_threshold")
    vis.line_plot()

if __name__=='__main__':
    #comparison_carla_vlp32()
    #kitti_stats()
    fzd_intensity_spectra()