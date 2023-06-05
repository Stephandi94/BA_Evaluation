import os
import numpy as np
import evaluation_tools as et
import evaluation_visualizer as ev
import evaluation_data as ed

def remove_intensity_threshold(intensities):
    max_i = 10 * np.log10(3444 * 9)
    t = -48.8
    return (intensities * (max_i - t) + t) / max_i

def clear_weather():
    src = os.path.join(r"Weather_Clear/5-1_vlp16_no-noise.bin")
    pcl = et.pcl_to_numpy(src)
    tmp_intensities = pcl[:, 3]
    intensities = remove_intensity_threshold(tmp_intensities)
    distances = et.get_distances(pcl)

    x_data = np.arange(0, np.max(distances), 0.01)
    y_data = np.zeros_like(x_data)
    for i, v in enumerate(x_data):
        y_data[i] = np.exp(-0.002 * v)
    vis = ev.VisEval()
    vis.set_x_data([distances, x_data])
    vis.set_y_data([intensities, y_data])
    vis.set_y_label("intensity")
    vis.set_x_label("distance [m]")
    vis.set_data_labels(["attenuation by distance"])
    vis.scatter_plot(True)

def rain(remove_th=False):
    clear_src = os.path.join(r"Weather_Clear/5-1_vlp16_no-noise.bin")
    clear_pcl = et.pcl_to_numpy(clear_src)
    clear_int = et.get_intensity_info(clear_pcl)
    clear_dist = et.get_distance_info(clear_pcl)

    sub_dir = [29, 44, 59, 74, 89, 100]
    tick_labels = [0.0, 0.3, 1.2, 5.0, 21.0, 91.5, 150.0]
    rain_dist_int = np.zeros((len(tick_labels), 6))  # [distance_stats, intensity_stats]
    rain_dist_int[0, :] = np.array(np.concatenate((clear_dist, clear_int)))

    for i, l in enumerate(sub_dir):
        rain_stats = ed.read_csv_to_array(os.path.join(r"Weather_Rain/{}/rain_stats.csv".format(sub_dir[i])), True, True)  # TODO
        rain_dist_int[i + 1, :] = np.array([rain_stats[1, 0], rain_stats[2, 1], rain_stats[3, 0], rain_stats[4, 0], rain_stats[5, 1], rain_stats[6, 0]])  # distance max_max, intensity mean_mean
    inty = np.zeros((rain_dist_int.shape[0], 2))
    if remove_th:
        inty[:, 0] = remove_intensity_threshold(rain_dist_int[:, 3])
        inty[:, 1] = remove_intensity_threshold(rain_dist_int[:, 5])
        inty[inty < 0] = 0
    else:
        inty[:, 0] = rain_dist_int[:, 3]
        inty[:, 1] = rain_dist_int[:, 5]
    dist = np.zeros((rain_dist_int.shape[0], 2))
    dist[:, 0] = rain_dist_int[:, 0]
    dist[:, 1] = rain_dist_int[:, 2]

    vis = ev.VisEval()
    vis.set_x_data([np.arange(len(tick_labels))])
    vis.set_y_data([inty, dist]) # intensity, distance
    vis.set_data_labels(["mean intensity", "max intensity", "mean distance", "max distance"])
    vis.set_x_label("precipitation rate [mm/h]")
    vis.set_y_label(["intensity", "distance [m]"])
    vis.set_x_ticklabel(tick_labels)
    vis.set_legend_location([(0.02, 0.4), (0.02, 0.25)])
    vis.set_colors(['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e'])
    vis.set_linestyle(['-', '--', '-', '--'])
    vis.set_marker('.')
    vis.set_xlim([0, len(tick_labels)-1])
    if remove_th:
        vis.to_file(os.path.join(r"Weather_Rain"), r"eval_rain_intensity_and_distance_removed_th")
        vis.set_ylim([[0.0, 1.01], [0, 102]])
    else:
        vis.to_file(os.path.join(r"Weather_Rain"), r"eval_rain_intensity_and_distance")
        vis.set_ylim([[0.5, 1.01], [0, 102]])
    vis.line_plot_two_y_axis()

def snow(remove_th=False):
    clear_src = os.path.join(r"Weather_Clear/5-1_vlp16_no-noise.bin")
    clear_pcl = et.pcl_to_numpy(clear_src)
    clear_int = et.get_intensity_info(clear_pcl)
    clear_dist = et.get_distance_info(clear_pcl)

    sub_dir = [29, 44, 59, 74, 89, 100]
    tick_labels = [0.0, 0.3, 1.2, 5.0, 21.0, 91.5, 150.0]
    snow_dist_int = np.zeros((len(tick_labels), 6))  # [distance_stats, intensity_stats]
    snow_dist_int[0, :] = np.array(np.concatenate((clear_dist, clear_int)))

    for i, l in enumerate(sub_dir):
        snow_stats = ed.read_csv_to_array(os.path.join(r"Weather_Snow/{}/snow_stats.csv".format(sub_dir[i])), True, True)  # TODO
        snow_dist_int[i + 1, :] = np.array([snow_stats[1, 0], snow_stats[2, 1], snow_stats[3, 0], snow_stats[4, 0], snow_stats[5, 1], snow_stats[6, 0]])  # distance max_max, intensity mean_mean
    inty = np.zeros((snow_dist_int.shape[0], 2))
    if remove_th:
        inty[:, 0] = remove_intensity_threshold(snow_dist_int[:, 3])
        inty[:, 1] = remove_intensity_threshold(snow_dist_int[:, 5])
        inty[inty < 0] = 0
    else:
        inty[:, 0] = snow_dist_int[:, 3]
        inty[:, 1] = snow_dist_int[:, 5]
    dist = np.zeros((snow_dist_int.shape[0], 2))
    dist[:, 0] = snow_dist_int[:, 0]
    dist[:, 1] = snow_dist_int[:, 2]

    vis = ev.VisEval()
    vis.set_x_data([np.arange(len(tick_labels))])
    vis.set_y_data([inty, dist]) # intensity, distance
    vis.set_data_labels(["mean intensity", "max intensity", "mean distance", "max distance"])
    vis.set_x_label("precipitation rate [mm/h]")
    vis.set_y_label(["intensity", "distance [m]"])
    vis.set_x_ticklabel(tick_labels)
    vis.set_legend_location([(0.02, 0.35), (0.02, 0.20)])
    vis.set_colors(['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e'])
    vis.set_linestyle(['-', '--', '-', '--'])
    vis.set_marker('.')
    vis.set_xlim([0, len(tick_labels)-1])
    if remove_th:
        #vis.to_file(os.path.join(r"Weather_Snow"), r"eval_snow_intensity_and_distance_removed_th")
        vis.set_ylim([[0.0, 1.01], [0, 102]])
    else:
        vis.to_file(os.path.join(r"Weather_Snow"), r"eval_snow_intensity_and_distance")
        vis.set_ylim([[0.5, 1.01], [0, 102]])
    vis.line_plot_two_y_axis()

def fog(remove_th=False):
    clear_src = os.path.join(r"Weather_Clear/5-1_vlp16_no-noise.bin")
    clear_pcl = et.pcl_to_numpy(clear_src)
    clear_int = et.get_intensity_info(clear_pcl)
    clear_dist = et.get_distance_info(clear_pcl)

    sub_dir = [2001, 1001, 201, 51, 0]
    tick_labels = [3000, 1500, 600, 125, 25]
    fog_dist_int = np.zeros((len(tick_labels), 6))  # [distance_stats, intensity_stats]
    fog_dist_int[0, :] = np.array(np.concatenate((clear_dist, clear_int)))

    for i, l in enumerate(sub_dir):
        fog_stats = ed.read_csv_to_array(os.path.join(r"Weather_Fog/{}/fog_stats.csv".format(sub_dir[i])), True, True)  # TODO
        fog_dist_int[i, :] = np.array([fog_stats[1, 0], fog_stats[2, 1], fog_stats[3, 0], fog_stats[4, 0], fog_stats[5, 1], fog_stats[6, 0]])  # distance max_max, intensity mean_mean
    inty = np.zeros((fog_dist_int.shape[0], 2))
    if remove_th:
        inty[:, 0] = remove_intensity_threshold(fog_dist_int[:, 3])
        inty[:, 1] = remove_intensity_threshold(fog_dist_int[:, 5])
        inty[inty < 0] = 0
    else:
        inty[:, 0] = fog_dist_int[:, 3]
        inty[:, 1] = fog_dist_int[:, 5]
    dist = np.zeros((fog_dist_int.shape[0], 2))
    dist[:, 0] = fog_dist_int[:, 0]
    dist[:, 1] = fog_dist_int[:, 2]

    vis = ev.VisEval()
    vis.set_x_data([np.arange(len(tick_labels))])
    vis.set_y_data([inty, dist]) # intensity, distance
    vis.set_data_labels(["mean intensity", "max intensity", "mean distance", "max distance"])
    vis.set_x_label("meteorological visibility [m]")
    vis.set_y_label(["intensity", "distance [m]"])
    vis.set_x_ticklabel(tick_labels)
    vis.set_legend_location([(0.02, 0.4), (0.02, 0.25)])
    vis.set_colors(['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e'])
    vis.set_linestyle(['-', '--', '-', '--'])
    vis.set_marker('.')
    vis.set_xlim([0, len(tick_labels)-1])
    if remove_th:
        #vis.to_file(os.path.join(r"Weather_Fog"), r"eval_fog_intensity_and_distance_removed_th")
        vis.set_ylim([[0.0, 1.01], [0, 102]])
    else:
        vis.to_file(os.path.join(r"Weather_Fog"), r"eval_fog_intensity_and_distance")
        vis.set_ylim([[0.0, 1.01], [0, 102]])
    vis.line_plot_two_y_axis()


if __name__=='__main__':
    #clear_weather()
    rain()
    snow()
    fog()