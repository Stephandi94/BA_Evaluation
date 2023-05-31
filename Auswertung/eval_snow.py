import os
import numpy as np
import evaluation_tools as et
import evaluation_visualizer as ev
import evaluation_data as ed

def read_augmentation(src):
    data = ed.read_csv_to_array(src, True, True)
    return np.array([data[1, 0], data[2, 0]])

def read_augmentation_phys(src):
    data = ed.read_csv_to_array(src, True, False).reshape(-1)
    return np.array([data[1], data[2]])

def main():
    """
    comparison of the different snow rates
    """
    sub_dir = [29, 44, 59, 74, 89, 100]
    tick_labels = [0.3, 1.2, 5.0, 21.0, 91.5, 150.0]
    snow_augmentation = np.zeros((len(sub_dir), 2))
    snow_augmentation_physical = np.zeros((len(sub_dir), 2))
    for i, s in enumerate(sub_dir):
        snow_augmentation[i, :] = read_augmentation(os.path.join(r"Weather_Snow\{}\augmentation_stats.csv".format(s))) #TODO
        snow_augmentation_physical[i, :] = read_augmentation_phys(os.path.join(r"Weather_Snow\{}\snow_augmentation_stats_physical_{}.csv".format(s, s))) #TODO

    # only model values
    vis_aug = ev.VisEval()
    vis_aug.set_x_data([np.arange(len(sub_dir))])
    vis_aug.set_y_data([snow_augmentation[:, 0], snow_augmentation[:, 1]])
    vis_aug.set_x_ticklabel(tick_labels)
    vis_aug.set_x_label("precipitation rate [mm/h]")
    vis_aug.set_y_label("# points")
    vis_aug.set_marker('o')
    vis_aug.set_legend_location('upper left')
    vis_aug.set_data_labels(["lost points", "added noise"])
    vis_aug.to_file(os.path.join(r"Weather_Snow"), r"snow_augmentation_comparison") #TODO
    vis_aug.line_plot()

    # compared to phys model
    vis_aug_c = ev.VisEval()
    vis_aug_c.set_x_data([np.arange(len(sub_dir))])
    vis_aug_c.set_y_data([snow_augmentation, snow_augmentation_physical])
    vis_aug_c.set_x_ticklabel(tick_labels)
    vis_aug_c.set_x_label("precipitation rate [mm/h]")
    vis_aug_c.set_y_label(["# points by FZD Model", "# points by Physical Model"])
    vis_aug_c.set_data_labels(["lost points [FZD Model]", "added noise [FZD Model]", "lost points [Phys. Model]", "added noise [Phys. Model]"])
    vis_aug_c.set_legend_location([(0.02, 0.85), (0.02, 0.7)])
    vis_aug_c.to_file(os.path.join(r"Weather_Snow"), r"snow_augmentation_comparison_physical") #TODO
    vis_aug_c.set_colors(['#1f77b4', '#ff7f0e', '#1f77b4', '#ff7f0e'])
    vis_aug_c.set_linestyle(['-', '-', '--', '--'])
    vis_aug_c.line_plot_two_y_axis()

    # get max detection distance and intensity mean
    sub_dir = [29, 44, 59, 74, 89, 100]
    tick_labels = [0.0, 0.3, 1.2, 5.0, 21.0, 91.5, 150.0]
    snow_dist_int = np.zeros((len(tick_labels), 6))     # [distance_stats, intensity_stats]
    snow_dist_int_phys = np.zeros((len(tick_labels), 6))
    clear_dist = ed.read_csv_to_array(os.path.join(r"Weather_Clear/distance_stats.csv")).reshape(-1)
    clear_int = ed.read_csv_to_array(os.path.join(r"Weather_Clear/intensity_stats.csv")).reshape(-1)
    snow_dist_int[0, :] = np.array(np.concatenate((clear_dist, clear_int)))

    clear_dist_phys = ed.read_csv_to_array(os.path.join(r"Weather_Snow/clean_distance_stats_physical.csv")).reshape(-1) #TODO
    clear_int_phys = ed.read_csv_to_array(os.path.join(r"Weather_Snow/clean_intensity_stats_physical.csv")).reshape(-1) #TODO
    snow_dist_int_phys[0, :] = np.array(np.concatenate((clear_dist_phys, clear_int_phys)))
    for i, l in enumerate(sub_dir):
        snow_stats = ed.read_csv_to_array(os.path.join(r"Weather_Snow/{}/snow_stats.csv".format(sub_dir[i])), True, True) #TODO
        snow_dist_int[i+1, :] = np.array([snow_stats[1, 0], snow_stats[2, 1], snow_stats[3, 2], snow_stats[4, 0], snow_stats[5, 1], snow_stats[6, 2]]) # distance max_max, intensity mean_mean
        phys_dist = ed.read_csv_to_array(os.path.join(r"Weather_Snow/{}/snow_distance_stats_physical_{}.csv".format(l, l))).reshape(-1) #TODO
        phys_int = ed.read_csv_to_array(os.path.join(r"Weather_Snow/{}/snow_intensity_stats_physical_{}.csv".format(l, l))).reshape(-1) #TODO
        snow_dist_int_phys[i+1, :] = np.array(np.concatenate((phys_dist, phys_int)))

    # plot max detection range and mean intensity compared with physical model
    data_dist = np.zeros((len(tick_labels), 2))
    data_int = np.zeros((len(tick_labels), 2))
    data_dist[:, 0] = snow_dist_int[:, 2]
    data_dist[:, 1] = snow_dist_int_phys[:, 2]
    data_int[:, 0] = snow_dist_int[:, 3]
    data_int[:, 1] = snow_dist_int_phys[:, 3]
    vis_att_c = ev.VisEval()
    vis_att_c.set_x_data([np.arange(len(tick_labels))])
    vis_att_c.set_y_data([data_dist, data_int])
    vis_att_c.set_x_ticklabel(tick_labels)
    vis_att_c.set_colors(['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e'])
    vis_att_c.set_linestyle(['-', '--', '-', '--'])
    vis_att_c.set_x_label("precipitation rate [mm/h]")
    vis_att_c.set_y_label(["distance [m]", "intensity"])
    vis_att_c.set_data_labels(["max detection range [FZD Model]", "max detection range [Phys. Model]", "mean detection intensity [FZD Model]",
                               "mean detection intensity [Phys. Model]"])
    vis_att_c.set_legend_location([(0.02, 0.2), (0.02, 0.05)])
    vis_att_c.to_file(os.path.join(r"Weather_Snow"), r"snow_attenuation_comparison_physical") #TODO
    vis_att_c.set_ylim([[95, 105], [0.4, 1]])
    vis_att_c.line_plot_two_y_axis()

    # just intensities
    vis_int = ev.VisEval()
    vis_int.set_x_data(np.arange(len(tick_labels)))
    vis_int.set_y_data([snow_dist_int[:, 3:]])
    vis_int.set_data_labels(["mean", "min", "max"])
    vis_int.set_x_label("precipitation rate [mm/h]")
    vis_int.set_y_label("intensity")
    vis_int.set_legend_frame(True)
    vis_int.set_legend_location((0.875, 0.1))
    vis_int.to_file(os.path.join(r"Weather_Snow"), r"snow_intensities_comparison") #TODO
    vis_int.plot_stats()

    # just distances
    vis_dist = ev.VisEval()
    vis_dist.set_x_data(np.arange(len(tick_labels)))
    vis_dist.set_y_data([snow_dist_int[:, :3]])
    vis_dist.set_data_labels(["mean", "min", "max"])
    vis_dist.set_x_label("precipitation rate [mm/h]")
    vis_dist.set_y_label("detection distance [m]")
    vis_dist.set_legend_frame(True)
    vis_dist.set_legend_location('center right')
    vis_dist.to_file(os.path.join(r"Weather_Snow"), r"snow_distances_comparison") #TODO
    vis_dist.plot_stats()

if __name__=='__main__':
    main()